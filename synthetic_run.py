# sparse_experiment.py
# Extension of reproduce_nonrobust.py to test L1 regularization (sparsity) effect.
# Vardi et al., arXiv:2202.04347 reference (baseline experiment). 

import math, random, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import List, Dict
import argparse
# -----------------------
# Dataset utilities (same as before)
# -----------------------
def sample_uniform_sphere(d: int, m: int, radius: float = None, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
    if radius is None:
        radius = math.sqrt(d)
    x = np.random.normal(size=(m, d)).astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / (norms + 1e-12) * radius
    y = np.random.choice([-1.0, 1.0], size=(m,)).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)

# -----------------------
# Model
# -----------------------

class Depth2ReLU(nn.Module):
    def __init__(self, input_dim: int, width: int,
                 bias_hidden: bool = True, bias_out: bool = False,
                 dropout_p: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, width, bias=bias_hidden)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.fc2 = nn.Linear(width, 1, bias=bias_out)

        # initialization
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.01)
        if bias_hidden:
            nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)

    def forward(self, x):
        h = self.act1(self.fc1(x))           # hidden activations
        h = self.dropout(h)
        out = self.fc2(h).squeeze(-1)        # linear output, no ReLU
        return out, h                        # also return hidden activations



# -----------------------
# Losses
# -----------------------
def logistic_loss_from_logits(logits, labels_pm1):
    return torch.log1p(torch.exp(-labels_pm1 * logits)).mean()

def l1_reg_term(model: nn.Module, layer: str):
    """Return L1 norm of parameters of the chosen layer name."""
    if layer == "fc1.weight":
        return model.fc1.weight.abs().sum()
    if layer == "fc1.bias":
        return model.fc1.bias.abs().sum()
    if layer == "fc2.weight":
        return model.fc2.weight.abs().sum()
    if layer == "fc2.bias":
        return model.fc2.bias.abs().sum()
    raise ValueError(f"Unknown layer: {layer}")

# -----------------------
# Training with optional L1
# -----------------------
def train_until_zero_loss(model, X, y, max_epochs=20000, lr=0.1, tol_loss=1e-7,#20000
                          device='cpu', verbose=False, l1_lambda=0.0, l1_layer=None, mode="l1"):
    model.to(device)
    X, y = X.to(device), y.to(device)
    opt = optim.SGD(model.parameters(), lr=lr)
    loss  = None
    for epoch in range(max_epochs):
        opt.zero_grad()
        logits, h = model(X)
        loss = logistic_loss_from_logits(logits, y)
        # add L1 regularization if specified
        if mode == "l1" and l1_lambda > 0.0:
            loss = loss + l1_lambda * h.abs().sum()
        loss_val = float(loss.item())
        if loss_val < tol_loss:
            break
        loss.backward()
        opt.step()
    return model, loss

# -----------------------
# Attack evaluation (same as before)
# -----------------------
def margin_samples(model, X, y, device='cpu', slack_ratio=1.1):
    model.to(device)
    with torch.no_grad():
        logits, h = model(X.to(device))
        margins = logits * y.to(device)
        min_margin = torch.min(margins).item()
        thr = slack_ratio * min_margin
        idx = torch.where(margins <= thr)[0].cpu().numpy().tolist()
    return idx

def compute_z_direction(X, y, indices):
    if len(indices) == 0:
        return torch.zeros(X.shape[1], dtype=X.dtype)
    return (y[indices].unsqueeze(1) * X[indices]).sum(dim=0)

def minimal_c_to_flip(model, X, y, indices, z, device='cpu', c_tol=1e-4, c_max=None):
    if len(indices) == 0:
        return float("nan")
    d = X.shape[1]
    if c_max is None:
        c_max = 4.0 * math.sqrt(d)
    z_norm = z.norm().item()
    if z_norm == 0:
        return float("inf")
    z_unit = z / (z_norm + 1e-12)
    def flips_all(c: float):
        with torch.no_grad():
            pert = (-c) * y[indices].unsqueeze(1) * z_unit.unsqueeze(0)
            Xp = X[indices] + pert
            logits, h = model(Xp.to(device))
            return bool(torch.all((logits * y[indices].to(device)) < 0).item())
    if not flips_all(c_max):
        return float("inf")
    lo, hi = 0.0, c_max
    while hi - lo > c_tol:
        mid = 0.5*(lo+hi)
        if flips_all(mid):
            hi = mid
        else:
            lo = mid
    return hi

# -----------------------
# Experiment runner
# -----------------------
def run_experiment(d=200, m=None, width=1000, seed=0, l1_lambda=0.0, l1_layer=None, 
                   device='cpu',  dropout_p=0.0, mode="l1"):
    if m is None:
        m = d
    X, y = sample_uniform_sphere(d, m, seed=seed)
    y = y.clone()
    model = Depth2ReLU(d, width,  dropout_p=(dropout_p if mode=="dropout" else 0.0))
    model, loss = train_until_zero_loss(model, X, y, l1_lambda=l1_lambda, l1_layer=l1_layer,  mode=mode, device=device)
    idx = margin_samples(model, X, y, device=device)
    z = compute_z_direction(X, y, idx)
    c = minimal_c_to_flip(model, X, y, idx, z, device=device)
    if math.isinf(c):
        c = 100.0
    print(loss.item())
    return {
        "d": d, "m": m, "width": width, "seed": seed, "dropout_p": dropout_p,
        "l1_lambda": l1_lambda, "l1_layer": l1_layer, "loss": float(loss.item()),
        "min_c": c, "num_margin": len(idx)
    }

# -----------------------
# Sweeps and plotting
# -----------------------
def sweep_l1_layers(d=200, m=None, width=1000, seeds=[0,1,2], l1_lambdas=[0.0,1e-4,1e-3,1e-2,1e-1],
                    device='cpu',
                    out_file="sparsity_results.json"):
    results = []
    for lam in l1_lambdas:
        for seed in seeds:
            r = run_experiment(d=d, m=m, width=width, seed=seed,
                               l1_lambda=lam, device=device)
            results.append(r)
            print(r)
    # save
    with open(out_file,"w") as f:
        json.dump(results, f, indent=2)
    return results


def sweep_dropout(d=200, m=None, width=1000, seeds=[0,1,2],
                  dropout_ps=[0.0,0.1,0.2,0.5],
                  device='cpu', out_file="dropout_results.json"):
    results = []
    for p in dropout_ps:
        for seed in seeds:
            r = run_experiment(d=d, m=m, width=width, seed=seed,
                               dropout_p=p, device=device, mode="dropout")
            results.append(r)
            print(r)
    with open(out_file,"w") as f:
        json.dump(results, f, indent=2)
    return results


def plot_results(results_file="sparsity_results.json",
                 mode="l1",
                 out_file="my_fig2.png"):
    """
    Plot mean minimal perturbation 'min_c' vs regularization strength.
    - In 'l1' mode, aggregates over all l1_layer entries by l1_lambda.
    - In 'dropout' mode, aggregates by dropout_p.
    """

    with open(results_file, "r") as f:
        results = json.load(f)

    if mode == "dropout":
        x_key = "dropout_p"
        label = "Dropout"
        title = "Effect of Dropout on Robustness"
        x_label = "Dropout probability $p$"
    else:
        x_key = "l1_lambda"
        label = r"$\ell_1$ sparsity (all layers)"
        title = "Effect of $\ell_1$ Sparsity on Robustness"
        x_label = r"$\ell_1$ regularization strength $\lambda$"

    x_vals = sorted(set([r[x_key] for r in results]))
    means = []
    counts = []

    for x in x_vals:
        vals = [r["min_c"] for r in results if r[x_key] == x]
        vals = [v for v in vals if np.isfinite(v)]
        if len(vals) == 0:
            means.append(np.nan)
            counts.append(0)
        else:
            means.append(np.mean(vals))
            counts.append(len(vals))

    # Plot
    plt.figure(figsize=(8, 6))
    plt.errorbar(x_vals, means, marker='o', capsize=4, label=label)
    plt.xlabel(x_label)
    plt.ylabel("Mean minimal perturbation $c$")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.show()

    return {"x": x_vals, "mean": means, "count": counts}





# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["l1", "dropout"], default="l1")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    # Run a sweep and plot
    if args.mode == "l1":
        res = sweep_l1_layers(d=200, m=200, width=500, seeds=[0,1, 2, 3, 4], l1_lambdas=[0.0,1e-5, 2e-5, 4e-5, 6e-5,8e-5, 1e-4], device=args.device)
        plot_results("sparsity_results.json", out_file="l1_res.png")
    else:
        res = sweep_dropout(d=200, m=200, width=500,
                            seeds=[0,1, 2, 3, 4],
                            dropout_ps=[0.0,0.1,0.2,0.5],
                            device=args.device,
                            out_file="dropout_results.json")
        plot_results("dropout_results.json", mode = "dropout", out_file="dropout_res.png")