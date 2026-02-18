"""Thin wrapper: run GraviQ training (delegates to graviq.training)."""
from graviq.training import train

if __name__ == "__main__":
    train(
        data_dir='training_data',
        num_epochs=100,
        batch_size=8,
        learning_rate=1e-3,
    )
