"""
Minimal example of a Convolutional Neural Network (CNN) for classifying MNIST
hand-written digits using PyTorch.

CNN = Convolutional Neural Network
SGD = Stochastic Gradient Descent
"""

import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── 1. Hyper-parameters ─────────────────────────────────────────────────────────
# Default values, will be overridden by argparse
BATCH_SIZE = 64  # How many images the model sees before its weights update
EPOCHS = 5  # Full passes through the training set
LEARNING_RATE = 1e-2  # Step size for weight updates (high → faster, but riskier)


# ── 2. Data pipeline ────────────────────────────────────────────────────────────
def get_data_loaders(batch_size):
    """Loads MNIST dataset and returns DataLoader objects for train and test sets."""
    # Transform: convert images to tensors & scale pixel values to [0,1]
    transform = transforms.ToTensor()

    train_ds = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    return train_loader, test_loader


# ── 3. Network definition ──────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    """Two convolution blocks → flatten → two fully connected layers."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1: conv → activation → downsample
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=3, padding=1
            ),  # 28×28 → 28×28
            nn.ReLU(),  # add non-linearity
            nn.MaxPool2d(kernel_size=2),  # 28×28 → 14×14
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),  # 14×14 → 14×14
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14×14 → 7×7
            nn.Flatten(),  # 64×7×7 → 3136
            nn.Linear(64 * 7 * 7, 128),  # dense layer
            nn.ReLU(),
            nn.Linear(128, 10),  # 10 digits → logits
        )

    def forward(self, x):
        return self.net(x)


def create_model():
    """Initializes and returns the SimpleCNN model."""
    model = SimpleCNN()
    print(
        f"Model has {sum(p.numel() for p in model.parameters()):,} trainable parameters"
    )
    return model


# ── 4. Optimizer & loss ─────────────────────────────────────────────────────────
def train_model(model, train_loader, epochs, learning_rate):
    """Trains the model for a given number of epochs."""
    criterion = nn.CrossEntropyLoss()  # Measures gap between logits & true labels
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()  # switch to training mode (enables dropout, etc.)
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()  # reset accumulated gradients
            outputs = model(images)  # forward pass
            loss = criterion(outputs, labels)  # compute objective
            loss.backward()  # back-propagation
            optimizer.step()  # weight update
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} • Training loss: {avg_loss:.4f}")


# ── 5. Training loop ────────────────────────────────────────────────────────────
# This section is now part of train_model


# ── 6. Evaluation ───────────────────────────────────────────────────────────────
def evaluate_model(model, test_loader):
    """Evaluates the model on the test set and prints accuracy."""
    model.eval()  # set layers like dropout to inference mode
    correct = total = 0
    with torch.no_grad():  # disable gradient tracking for speed
        for images, labels in test_loader:
            outputs = model(images)
            predictions = outputs.argmax(dim=1)  # highest-logit class is the prediction
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    print(f"Test accuracy: {100 * correct / total:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Train a simple CNN on MNIST.")
    parser.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Input batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of epochs to train (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate (default: %(default)s)",
        dest="lr",
    )
    args = parser.parse_args()

    print("Starting training with the following parameters:")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")

    train_loader, test_loader = get_data_loaders(args.batch_size)
    model = create_model()
    train_model(model, train_loader, args.epochs, args.lr)
    evaluate_model(model, test_loader)


if __name__ == "__main__":
    main()
