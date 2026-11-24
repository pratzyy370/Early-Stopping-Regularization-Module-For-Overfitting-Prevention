from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #

class EarlyStoppingException(Exception):
    pass


# --------------------------------------------------------------------------- #
# EarlyStopping
# --------------------------------------------------------------------------- #

@dataclass
class EarlyStopping:
    patience: int = 10
    min_delta: float = 0.0
    min_delta_rel: float = 1e-4
    mode: str = "min"
    warmup_epochs: int = 0
    restore_best: bool = True
    state_getter: Optional[Callable[[], Any]] = None
    state_setter: Optional[Callable[[Any], None]] = None
    verbose: bool = True

    # -- internal state (not passed in) --
    best_score: float = field(init=False, default=math.inf)
    best_epoch: int = field(init=False, default=-1)
    num_bad_epochs: int = field(init=False, default=0)
    should_stop: bool = field(init=False, default=False)
    _best_state: Any = field(init=False, default=None, repr=False)
    history: List[Dict[str, Any]] = field(init=False, default_factory=list)

    def __post_init__(self):
        if self.mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.best_score = math.inf if self.mode == "min" else -math.inf

    def _is_improvement(self, current: float) -> bool:
        threshold = max(self.min_delta, self.min_delta_rel * abs(self.best_score)
                         if math.isfinite(self.best_score) else self.min_delta)
        if self.mode == "min":
            return current < (self.best_score - threshold)
        return current > (self.best_score + threshold)

    def step(self, current_score: float, epoch: Optional[int] = None) -> bool:
        epoch = epoch if epoch is not None else len(self.history)
        in_warmup = epoch < self.warmup_epochs

        improved = (not in_warmup) and self._is_improvement(current_score)

        if improved or (epoch == 0 and not in_warmup):
            self.best_score = current_score
            self.best_epoch = epoch
            self.num_bad_epochs = 0
            if self.restore_best and self.state_getter is not None:
                self._best_state = self.state_getter()
            if self.verbose:
                print(f"[EarlyStopping] epoch {epoch}: new best "
                      f"{current_score:.6g} (mode={self.mode})")
        elif not in_warmup:
            self.num_bad_epochs += 1

        self.history.append({
            "epoch": epoch,
            "score": current_score,
            "best_score": self.best_score,
            "num_bad_epochs": self.num_bad_epochs,
            "in_warmup": in_warmup,
        })

        if not in_warmup and self.num_bad_epochs >= self.patience:
            self.should_stop = True
            if self.verbose:
                print(f"[EarlyStopping] stopping at epoch {epoch}: "
                      f"no improvement for {self.num_bad_epochs} epochs "
                      f"(best={self.best_score:.6g} @ epoch {self.best_epoch})")

        return self.should_stop

    def restore_best_weights(self) -> bool:
        if self.restore_best and self._best_state is not None and self.state_setter:
            self.state_setter(self._best_state)
            if self.verbose:
                print(f"[EarlyStopping] restored weights from epoch {self.best_epoch}")
            return True
        return False

    def reset(self):
        self.best_score = math.inf if self.mode == "min" else -math.inf
        self.best_epoch = -1
        self.num_bad_epochs = 0
        self.should_stop = False
        self._best_state = None
        self.history.clear()

    def summary(self) -> str:
        return (f"EarlyStopping summary: best_score={self.best_score:.6g} "
                f"at epoch {self.best_epoch}, stopped={self.should_stop}, "
                f"total_epochs_seen={len(self.history)}")


# --------------------------------------------------------------------------- #
# TrainingMonitor
# --------------------------------------------------------------------------- #

class TrainingMonitor:
    def __init__(self, gap_smoothing: int = 3, gap_patience: int = 5,
                 gap_min_delta: float = 1e-3):
        self.gap_smoothing = max(1, gap_smoothing)
        self.gap_patience = max(1, gap_patience)
        self.gap_min_delta = gap_min_delta
        self.epochs: List[int] = []
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.timestamps: List[float] = []
        self.extra: List[Dict[str, float]] = []
        self._overfit_onset_epoch: Optional[int] = None
        self._rising_streak = 0
        self._min_gap = math.inf
        self._min_gap_epoch: Optional[int] = None

    def log(self, epoch: int, train_loss: float, val_loss: float, **extra_metrics):
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.timestamps.append(time.time())
        self.extra.append(extra_metrics)
        self._update_overfit_onset()

    def _smoothed_gap(self) -> List[float]:
        gaps = [v - t for t, v in zip(self.train_loss, self.val_loss)]
        w = self.gap_smoothing
        if w <= 1:
            return gaps
        out = []
        for i in range(len(gaps)):
            lo = max(0, i - w + 1)
            out.append(sum(gaps[lo:i + 1]) / (i - lo + 1))
        return out

    def _update_overfit_onset(self):
        if self._overfit_onset_epoch is not None:
            return
        gaps = self._smoothed_gap()
        # Require enough points for the moving average to have "settled"
        # before trusting it, otherwise early noise looks like a trend.
        if len(gaps) < max(2 * self.gap_smoothing, self.gap_patience + 1):
            return

        current_gap = gaps[-1]
        if current_gap > self._min_gap + self.gap_min_delta:
            self._rising_streak += 1
        else:
            self._rising_streak = 0
            if current_gap < self._min_gap:
                self._min_gap = current_gap
                self._min_gap_epoch = self.epochs[-1]

        if self._rising_streak >= self.gap_patience:
            self._overfit_onset_epoch = self._min_gap_epoch if self._min_gap_epoch is not None \
                else self.epochs[max(0, len(gaps) - self.gap_patience)]

    @property
    def overfitting_onset_epoch(self) -> Optional[int]:
        return self._overfit_onset_epoch

    def best_val_epoch(self) -> Optional[int]:
        if not self.val_loss:
            return None
        idx = min(range(len(self.val_loss)), key=lambda i: self.val_loss[i])
        return self.epochs[idx]

    def generalization_gap(self, epoch_idx: int = -1) -> float:
        if not self.val_loss:
            return math.nan
        return self.val_loss[epoch_idx] - self.train_loss[epoch_idx]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epochs": self.epochs,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "overfitting_onset_epoch": self._overfit_onset_epoch,
            "best_val_epoch": self.best_val_epoch(),
        }

    def save_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def plot(self, path: Optional[str] = None, show: bool = False):
        """Plot train/val curves with the best-epoch and overfitting-onset
        markers. Requires matplotlib. Saves to `path` if given."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.epochs, self.train_loss, label="train loss", color="#4C72B0")
        ax.plot(self.epochs, self.val_loss, label="val loss", color="#DD8452")

        best_epoch = self.best_val_epoch()
        if best_epoch is not None:
            ax.axvline(best_epoch, color="#55A868", linestyle="--",
                        label=f"best val (epoch {best_epoch})")

        if self._overfit_onset_epoch is not None:
            ax.axvline(self._overfit_onset_epoch, color="#C44E52", linestyle=":",
                        label=f"overfit onset (epoch {self._overfit_onset_epoch})")

        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("Training vs. validation loss")
        ax.legend()
        fig.tight_layout()

        if path:
            fig.savefig(path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)
        return path


# --------------------------------------------------------------------------- #
# Convenience: a single object that wraps both, for the common case
# --------------------------------------------------------------------------- #

class EarlyStoppingTrainer:

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        min_delta_rel: float = 1e-4,
        mode: str = "min",
        warmup_epochs: int = 0,
        restore_best: bool = True,
        state_getter: Optional[Callable[[], Any]] = None,
        state_setter: Optional[Callable[[Any], None]] = None,
        gap_smoothing: int = 3,
        gap_patience: int = 5,
        verbose: bool = True,
    ):
        self.stopper = EarlyStopping(
            patience=patience,
            min_delta=min_delta,
            min_delta_rel=min_delta_rel,
            mode=mode,
            warmup_epochs=warmup_epochs,
            restore_best=restore_best,
            state_getter=state_getter,
            state_setter=state_setter,
            verbose=verbose,
        )
        self.monitor = TrainingMonitor(gap_smoothing=gap_smoothing, gap_patience=gap_patience)

    def step(self, epoch: int, train_loss: float, val_loss: float, **extra_metrics) -> bool:
        self.monitor.log(epoch, train_loss, val_loss, **extra_metrics)
        monitored = val_loss if self.stopper.mode == "min" else -val_loss if False else val_loss
        return self.stopper.step(val_loss, epoch=epoch)

    def restore_best_weights(self) -> bool:
        return self.stopper.restore_best_weights()

    def plot(self, path: Optional[str] = None, show: bool = False):
        return self.monitor.plot(path=path, show=show)

    def summary(self) -> str:
        onset = self.monitor.overfitting_onset_epoch
        onset_str = f", overfitting onset ~ epoch {onset}" if onset is not None else ""
        return self.stopper.summary() + onset_str