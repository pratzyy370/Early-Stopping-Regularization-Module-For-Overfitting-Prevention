import math
import random
from early_stopping import EarlyStoppingTrainer

random.seed(7)

def synthetic_losses(epoch: int):
    train = 1.2 * math.exp(-0.15 * epoch) + 0.05 + random.gauss(0, 0.01)
    val_decay = 1.2 * math.exp(-0.13 * epoch) + 0.15
    overfit_term = 0.0
    if epoch > 18:
        overfit_term = 0.004 * (epoch - 18) ** 1.3
    val = val_decay + overfit_term + random.gauss(0, 0.015)
    return max(train, 1e-4), max(val, 1e-4)

def main():
    current_epoch = [0]
    trainer = EarlyStoppingTrainer(
        patience=8,
        min_delta=1e-4,
        min_delta_rel=1e-3,
        mode="min",
        warmup_epochs=3,
        restore_best=True,
        state_getter=lambda: {"epoch": current_epoch[0]},
        state_setter=lambda state: print(f"  (restored simulated state: {state})"),
        gap_smoothing=3,
        gap_patience=5,
        verbose=True,
    )
    max_epochs = 60
    for epoch in range(max_epochs):
        current_epoch[0] = epoch
        train_loss, val_loss = synthetic_losses(epoch)
        stop = trainer.step(epoch, train_loss, val_loss)
        print(f"epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}")
        if stop:
            print("\n>>> Early stopping triggered.")
            break
    trainer.restore_best_weights()
    print()
    print(trainer.summary())
    out_path = "loss_curves.png"
    trainer.plot(path=out_path)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()