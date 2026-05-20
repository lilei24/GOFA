import os
import time
from typing import Any, Optional, Dict, Union, Callable

from gp.lightning.module_template import BaseTemplate
import torch

class GraphPredLightning(BaseTemplate):
    def forward(self, batch):
        return self.model(batch)

    def on_train_start(self) -> None:
        torch.cuda.empty_cache()
        self.optimizers().param_groups[0]['lr'] = self.exp_config.lr
        self.lr_schedulers().last_epoch = -1
        self.lr_schedulers().T_max = self.exp_config.T_max

class GraphTextPredLightning(BaseTemplate):
    def forward(self, batch):
        # print(batch)
        return self.model(batch)

    @staticmethod
    def _debug_rank():
        return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))

    @staticmethod
    def _sync_cuda():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _log_train_profile(self, batch_idx, stage, elapsed=None):
        elapsed_text = f" elapsed={elapsed:.2f}s" if elapsed is not None else ""
        print(
            f"[GOFA train profile] rank={self._debug_rank()} batch_idx={batch_idx} "
            f"stage={stage}{elapsed_text}",
            flush=True,
        )

    def on_train_start(self) -> None:
        torch.cuda.empty_cache()
        self.optimizers().param_groups[0]['lr'] = self.exp_config.lr
        self.lr_schedulers().last_epoch = -1
        self.lr_schedulers().T_max = self.exp_config.T_max

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:
        self._current_train_batch_idx = batch_idx
        self._train_batch_start_time = time.perf_counter()
        self._log_train_profile(batch_idx, "batch_start")

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.trainer.local_rank == 0:
            print("save pth model")
            self.model.save_partial(os.path.join(self.model.save_dir, "mem_ckpt.pth"))
        for k in list(checkpoint["state_dict"].keys()):
            if "g_layers" not in k:
                del checkpoint["state_dict"][k]

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        if self.trainer.local_rank == 0:
            print("save last epoch ckpt")
            self.model.save_partial(os.path.join(self.model.save_dir, "last_epoch_ckpt.pth"))

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        self._sync_cuda()
        step_start = time.perf_counter()
        self._log_train_profile(batch_idx, "training_step_start")
        try:
            score, loss = self.compute_results(batch, batch_idx, self.exp_config.train_state_name[dataloader_idx])
        except RuntimeError as e:
            if "out of memory" in str(e):
                self._log_train_profile(batch_idx, "oom_raise")
                torch.cuda.empty_cache()
            raise e
        self._sync_cuda()
        self._log_train_profile(batch_idx, "training_step_end", time.perf_counter() - step_start)
        return loss

    def on_before_backward(self, loss):
        self._sync_cuda()
        self._backward_start_time = time.perf_counter()
        batch_idx = getattr(self, "_current_train_batch_idx", "unknown")
        self._log_train_profile(batch_idx, "before_backward")

    def on_after_backward(self):
        self._sync_cuda()
        elapsed = time.perf_counter() - getattr(self, "_backward_start_time", time.perf_counter())
        batch_idx = getattr(self, "_current_train_batch_idx", "unknown")
        self._log_train_profile(batch_idx, "after_backward", elapsed)

    def on_before_optimizer_step(self, optimizer):
        self._sync_cuda()
        batch_idx = getattr(self, "_current_train_batch_idx", "unknown")
        self._log_train_profile(batch_idx, "before_optimizer_step")

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        self._sync_cuda()
        elapsed = time.perf_counter() - getattr(self, "_train_batch_start_time", time.perf_counter())
        self._log_train_profile(batch_idx, "batch_end", elapsed)

    # def on_validation_epoch_start(self) -> None:
    #     super().on_validation_epoch_start()
    #     self.old_decode = self.model.decode
    #     self.model.decode = self.model.generate
    #
    # def on_validation_epoch_end(self):
    #     super().on_validation_epoch_end()
    #     self.model.decode = self.old_decode
