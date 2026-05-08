import os
import os.path as osp
from typing import Optional, Union, List, Callable, Any

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion
from torch.optim import Optimizer

from gp.lightning.metric import EvalKit


def _safe_debug_write_line(instance, message):
    rank = getattr(instance, "global_rank", 0)
    log_dir = os.path.join(os.getcwd(), "tmp")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"gofa_stage3_rank_{rank}.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


class ExpConfig:
    def __init__(self, name, optimizer, opt_params=None, lr_scheduler=None, dataset_callback=None, acc_grad_step=1, ):
        self.name = name
        self.optimizer = optimizer
        self.lr = optimizer.param_groups[0]['lr']
        self.T_max = lr_scheduler["scheduler"].T_max
        self.opt_params = opt_params
        self.lr_scheduler = lr_scheduler
        self.train_state_name = ["train_eval"]
        self.val_state_name = ["valid"]
        self.test_state_name = ["test"]
        self.acc_grad_step = acc_grad_step
        self.dataset_callback = dataset_callback

    @property
    def train_state_name(self):
        return self._train_state_name

    @property
    def val_state_name(self):
        return self._val_state_name

    @property
    def test_state_name(self):
        return self._test_state_name

    @train_state_name.setter
    def train_state_name(self, value: Union[str, List[str]]):
        if isinstance(value, str):
            self._train_state_name = [value]
        else:
            self._train_state_name = value

    @val_state_name.setter
    def val_state_name(self, value: Union[str, List[str]]):
        if isinstance(value, str):
            self._val_state_name = [value]
        else:
            self._val_state_name = value

    @test_state_name.setter
    def test_state_name(self, value: Union[str, List[str]]):
        if isinstance(value, str):
            self._test_state_name = [value]
        else:
            self._test_state_name = value

    def get_optimizer(self):
        return self.optimizer

    def get_scheduler(self):
        return self.lr_scheduler


class BaseTemplate(LightningModule):
    def __init__(self, exp_config: ExpConfig, model: torch.nn.Module, eval_kit: Optional[EvalKit] = None,
                 name: str = "", ):

        super().__init__()

        self.exp_config = exp_config

        self.model = model
        self.name = name

        self.op_step = 0
        self.eval_kit = eval_kit

    def _debug_eval_print(self, stage, step_name, batch_idx):
        if batch_idx > 2:
            return
        rank = getattr(self, "global_rank", 0)
        local_rank = getattr(self.trainer, "local_rank", 0) if getattr(self, "trainer", None) is not None else 0
        message = f"[DEBUG-EVAL] rank={rank} local_rank={local_rank} stage={stage} step={step_name} batch_idx={batch_idx}"
        print(message, flush=True)
        _safe_debug_write_line(self, message)

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    def configure_optimizers(self):
        optimizer = self.exp_config.get_optimizer()
        optimizer_dict = {"optimizer": optimizer}
        if self.exp_config.get_scheduler() is not None:
            optimizer_dict["lr_scheduler"] = self.exp_config.get_scheduler()
        return optimizer_dict

    def compute_results(self, batch, batch_idx, step_name, log_loss=True, *args):
        if any(tag in step_name for tag in ("_val", "_test", "valid", "test")):
            self._debug_eval_print("before_forward", step_name, batch_idx)
        score = self(batch, *args)
        if any(tag in step_name for tag in ("_val", "_test", "valid", "test")):
            self._debug_eval_print("after_forward", step_name, batch_idx)
        loss = self.eval_kit.compute_loss(score, batch)
        if any(tag in step_name for tag in ("_val", "_test", "valid", "test")):
            self._debug_eval_print("after_loss", step_name, batch_idx)
        if any(tag in step_name for tag in ("_val", "_test", "valid", "test")):
            self._debug_eval_print("before_log", step_name, batch_idx)
        self.log(osp.join(self.name, step_name, "loss"), loss, on_step=True, on_epoch=False, prog_bar=log_loss,
                 batch_size=batch.batch_size if hasattr(batch, "batch_size") else len(batch), sync_dist=True, )
        if any(tag in step_name for tag in ("_val", "_test", "valid", "test")):
            self._debug_eval_print("after_log", step_name, batch_idx)
        with torch.no_grad():
            if self.eval_kit.has_eval_state(step_name):
                if any(tag in step_name for tag in ("_val", "_test", "valid", "test")):
                    self._debug_eval_print("before_eval_step", step_name, batch_idx)
                self.eval_kit.eval_step(score, batch, step_name)
                if any(tag in step_name for tag in ("_val", "_test", "valid", "test")):
                    self._debug_eval_print("after_eval_step", step_name, batch_idx)
        return score, loss

    def epoch_post_process(self, epoch_name):
        if self.eval_kit.has_eval_state(epoch_name):
            metric = self.eval_kit.eval_epoch(epoch_name)
            self.log(self.eval_kit.get_metric_name(epoch_name), metric, prog_bar=True, sync_dist=True)
            self.eval_kit.eval_reset(epoch_name)
            return metric

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        try:
            score, loss = self.compute_results(batch, batch_idx, self.exp_config.train_state_name[dataloader_idx])
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("Ignoring OOM batch")
                loss = torch.tensor([0])
            else:
                raise e
        return loss

    def on_train_epoch_end(self):
        for name in self.exp_config.train_state_name:
            self.epoch_post_process(name)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self._debug_eval_print("validation_step_enter", self.exp_config.val_state_name[dataloader_idx], batch_idx)
        self.compute_results(batch, batch_idx, self.exp_config.val_state_name[dataloader_idx], log_loss=False, )
        self._debug_eval_print("validation_step_exit", self.exp_config.val_state_name[dataloader_idx], batch_idx)

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self._debug_eval_print("on_validation_batch_end", self.exp_config.val_state_name[dataloader_idx], batch_idx)

    def on_validation_epoch_end(self):
        rank = getattr(self, "global_rank", 0)
        local_rank = getattr(self.trainer, "local_rank", 0) if getattr(self, "trainer", None) is not None else 0
        message = f"[DEBUG-EPOCH] rank={rank} local_rank={local_rank} stage=enter_on_validation_epoch_end"
        print(message, flush=True)
        _safe_debug_write_line(self, message)
        cur_metric = []
        for name in self.exp_config.val_state_name:
            message = f"[DEBUG-EPOCH] rank={rank} local_rank={local_rank} stage=before_epoch_post_process name={name}"
            print(message, flush=True)
            _safe_debug_write_line(self, message)
            metric = self.epoch_post_process(name)
            message = f"[DEBUG-EPOCH] rank={rank} local_rank={local_rank} stage=after_epoch_post_process name={name}"
            print(message, flush=True)
            _safe_debug_write_line(self, message)
            if metric is not None:
                cur_metric.append(metric.cpu())
        if self.exp_config.dataset_callback is not None:
            self.exp_config.dataset_callback(cur_metric)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self._debug_eval_print("test_step_enter", self.exp_config.test_state_name[dataloader_idx], batch_idx)
        self.compute_results(batch, batch_idx, self.exp_config.test_state_name[dataloader_idx], log_loss=False, )

    def on_test_epoch_end(self):
        for name in self.exp_config.test_state_name:
            self.epoch_post_process(name)
