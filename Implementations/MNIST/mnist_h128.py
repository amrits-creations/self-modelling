"""
MNIST self-modeling experiment, hidden layer size = 128.

Same as mnist_h64.py, only the hidden layer is wider. Reproduces another
slice of Fig. 2 in Premakumar et al. 2024
("Unexpected Benefits of Self-Modeling in Neural Systems"):
a baseline MLP vs. the same MLP with a self-modeling auxiliary head,
trained at five auxiliary weights (AW).
"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

HIDDEN = 128
AW_VALUES = [1, 5, 10, 20, 50]
EPOCHS = 50
BATCH_SIZE = 64
LR = 0.005
MOMENTUM = 0.9
SEED = 0

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


class MLP(nn.Module):
    """784 -> hidden (ReLU) -> 10, with an optional self-modeling head
    that regresses the hidden activations from themselves."""

    def __init__(self, hidden, self_model=False):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)
        self.self_model_head = nn.Linear(hidden, hidden) if self_model else None

    def forward(self, x):
        a = F.relu(self.fc1(x)) # 'a' here is the hidden layer's activations. passed x through first layer, then RELU the output to get 'a'.
        logits = self.fc2(a)
        a_hat = self.self_model_head(a) if self.self_model_head is not None else None
        return logits, a_hat, a

    def prune(self):
        """Return a baseline-shaped copy with the self-modeling head removed."""
        pruned = MLP(HIDDEN, self_model=False)
        pruned.fc1.load_state_dict(self.fc1.state_dict())
        pruned.fc2.load_state_dict(self.fc2.state_dict())
        return pruned


def load_mnist():
    # Mean-normalize using the training set's own mean (paper: "mean-normalized").
    raw = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
    pixel_mean = raw.data.float().mean() / 255.0

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x - pixel_mean),
        transforms.Lambda(lambda x: torch.flatten(x)),
    ])
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tfm)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tfm)
    return train_set, test_set


def self_model_loss(logits, y, a_hat, a, aw):
    ce = F.cross_entropy(logits, y)
    if a_hat is None:
        return ce
    n = a.shape[1]
    mse_term = ((a_hat - a) ** 2).sum(dim=1).mean() / n
    return ce + aw * mse_term


@torch.no_grad()
def test_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        logits, _, _ = model(x)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.shape[0]
    return correct / total


def train_one_condition(aw, train_set, test_loader):
    label = "baseline" if aw is None else f"aw{aw}"
    torch.manual_seed(SEED)

    model = MLP(HIDDEN, self_model=(aw is not None))
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, nesterov=True)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    weight_sd_history = []
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            logits, a_hat, a = model(x)
            loss = self_model_loss(logits, y, a_hat, a, aw)
            loss.backward()
            optimizer.step()

        weight_sd_history.append(model.fc2.weight.detach().std().item())
        print(f"[{label}] epoch {epoch + 1}/{EPOCHS}  weight_sd={weight_sd_history[-1]:.4f}")

    pruned = model.prune()
    acc = test_accuracy(pruned, test_loader)
    print(f"[{label}] final test accuracy: {acc:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.save(pruned.state_dict(), os.path.join(RESULTS_DIR, f"h{HIDDEN}_{label}.pt"))
    with open(os.path.join(RESULTS_DIR, f"h{HIDDEN}_{label}.json"), "w") as f:
        json.dump({
            "hidden": HIDDEN,
            "aw": aw,
            "seed": SEED,
            "test_accuracy": acc,
            "weight_sd_per_epoch": weight_sd_history,
        }, f, indent=2)


def main():
    train_set, test_set = load_mnist()
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

    train_one_condition(None, train_set, test_loader)
    for aw in AW_VALUES:
        train_one_condition(aw, train_set, test_loader)


if __name__ == "__main__":
    main()
