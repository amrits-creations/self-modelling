"""
Complexity measures for the trained MNIST models: weight-distribution
SD and RLCT (local learning coefficient), estimated via localized, tempered
SGLD following Lau et al. 2023, using the calibration reported for MNIST in
Premakumar et al. 2024 (SGLD lr = 1e-4, localization = 1000).

The paper doesn't state its beta (inverse temperature); we use the standard
SLT default beta = 1 / ln(n).

Usage: python complexity.py [hidden size]   (default 64)
"""

import json
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

HIDDEN = int(sys.argv[1]) if len(sys.argv) > 1 else 64
CONDITIONS = ["baseline", "aw1", "aw5", "aw10", "aw20", "aw50"]

SGLD_LR = 1e-4
LOCALIZATION = 1000.0
N_CHAINS = 4
N_BURNIN = 200
N_DRAWS = 200
SGLD_BATCH_SIZE = 512

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


class MLP(nn.Module):
    """Baseline-shaped 784 -> hidden (ReLU) -> 10 MLP (self-model head pruned)."""

    def __init__(self, hidden):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def load_train_tensors():
    raw = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
    pixel_mean = raw.data.float().mean() / 255.0
    x = raw.data.float() / 255.0 - pixel_mean
    x = x.flatten(start_dim=1)
    y = raw.targets
    return x, y


def full_loss(model, x, y):
    with torch.no_grad():
        return F.cross_entropy(model(x), y).item()


def chain_equilibrated(draws, loss_at_w_star, spread_tolerance=0.5, blowup_tolerance=2.0):
    """A chain is 'equilibrated' if it's sampling tightly around w*, per the
    paper's calibration criterion (loss should plateau, not run away):
    - spread: (max-min)/mean of the post-burn-in trace shouldn't be too wide.
    - blowup: the trace shouldn't wander far above the loss at the trained
      solution, which is what happens when localization is too weak to hold
      the chain near w* (seen at AW=50, where draws reached ~3.7x loss@w*)."""
    mean_d = sum(draws) / len(draws)
    spread = (max(draws) - min(draws)) / mean_d
    blowup = max(draws) / loss_at_w_star
    ok = spread <= spread_tolerance and blowup <= blowup_tolerance
    return ok, spread, blowup


def estimate_rlct(model, x, y):
    """Localized, tempered SGLD estimate of the RLCT (learning coefficient)."""
    n = x.shape[0]
    beta = 1.0 / math.log(n)
    w_star = [p.detach().clone() for p in model.parameters()]
    loss_at_w_star = full_loss(model, x, y)

    draws = []
    chain_traces = []
    for _ in range(N_CHAINS):
        chain = MLP(HIDDEN)
        chain.load_state_dict(model.state_dict())

        trace = []
        for step in range(N_BURNIN + N_DRAWS):
            idx = torch.randint(0, n, (SGLD_BATCH_SIZE,))
            chain.zero_grad()
            loss = F.cross_entropy(chain(x[idx]), y[idx])
            loss.backward()

            with torch.no_grad():
                for p, p_star in zip(chain.parameters(), w_star):
                    grad_term = n * beta * p.grad + LOCALIZATION * (p - p_star)
                    p -= (SGLD_LR / 2) * grad_term
                    p += torch.randn_like(p) * math.sqrt(SGLD_LR)

            trace.append(full_loss(chain, x, y))

        chain_traces.append(trace)
        draws.append(trace[N_BURNIN:])

    checks = [chain_equilibrated(d, loss_at_w_star) for d in draws]
    good_draws = [d for d, (ok, _, _) in zip(draws, checks) if ok]
    used_draws = good_draws if good_draws else draws  # fall back to all chains if none pass

    pooled = [x for d in used_draws for x in d]
    mean_draw_loss = sum(pooled) / len(pooled)
    llc = n * beta * (mean_draw_loss - loss_at_w_star)
    return llc, loss_at_w_star, checks, chain_traces, len(good_draws)


def main():
    x, y = load_train_tensors()
    summary = {}
    traces_by_condition = {}

    for label in CONDITIONS:
        path = os.path.join(RESULTS_DIR, f"h{HIDDEN}_{label}.pt")
        if not os.path.exists(path):
            print(f"skipping {label}: {path} not found (run mnist_h{HIDDEN}.py first)")
            continue

        model = MLP(HIDDEN)
        model.load_state_dict(torch.load(path))
        model.eval()

        weight_sd = model.fc2.weight.detach().std().item()
        rlct, loss_at_w_star, checks, chain_traces, n_used = estimate_rlct(model, x, y)

        spreads = [round(s, 2) for _, s, _ in checks]
        blowups = [round(b, 2) for _, _, b in checks]

        print(f"[{label}] weight_sd={weight_sd:.4f}  rlct={rlct:.2f}  loss@w*={loss_at_w_star:.4f}  "
              f"equilibrated_chains={n_used}/{N_CHAINS}  spread_per_chain={spreads}  "
              f"blowup_per_chain={blowups}")
        if n_used < N_CHAINS:
            print(f"  ^ WARNING: {N_CHAINS - n_used} chain(s) did not equilibrate "
                  f"(wandered far from w*) — RLCT may be unreliable for [{label}]")

        summary[label] = {
            "weight_sd": weight_sd,
            "rlct": rlct,
            "loss_at_w_star": loss_at_w_star,
            "equilibrated_chains": n_used,
            "n_chains": N_CHAINS,
            "spread_per_chain": spreads,
            "blowup_per_chain": blowups,
        }
        traces_by_condition[label] = chain_traces

    with open(os.path.join(RESULTS_DIR, f"h{HIDDEN}_complexity.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(RESULTS_DIR, f"h{HIDDEN}_rlct_traces.json"), "w") as f:
        json.dump(traces_by_condition, f)


if __name__ == "__main__":
    main()
