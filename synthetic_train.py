# reproduce_nonrobust.py
# Reproduces the empirical experiment style from:
# "Gradient Methods Provably Converge to Non-Robust Networks" (Vardi et al., arXiv:2202.04347).
# Paper: https://arxiv.org/pdf/2202.04347  (used as reference). :contentReference[oaicite:1]{index=1}

import math
import random
import argparse
from typing import Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# -----------------------
# Utilities: dataset
# -----------------------
def sample_uniform_sphere(d: int, m: int, radius: float = None, seed: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample m points uniformly from the d-dimensional sphere of given radius.
    Return X (m x d) and y (m), with y in {-1, +1} sampled uniformly.
    By default, radius = sqrt(d) (as in the paper).
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
    if radius is None:
        radius = math.sqrt(d)
    # sample gaussian and normalize to sphere
    x = np.random.normal(size=(m, d)).astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / (norms + 1e-12) * radius
    y = np.random.choice([-1.0, 1.0], size=(m,)).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)

# -----------------------
# Model: depth-2 ReLU
# -----------------------
class Depth2ReLU(nn.Module):
    def __init__(self, input_dim: int, width: int, bias_hidden: bool = True, bias_out: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, width, bias=bias_hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(width, 1, bias=bias_out)  # outputs scalar logit
        # initialization: small
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.01)
        if bias_hidden:
            nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)

    def forward(self, x):
        # x: (batch, d)
        h = self.act(self.fc1(x))
        out = self.fc2(h).squeeze(-1)  # shape: (batch,)
        return out

# -----------------------
# Logistic loss with labels in {-1,+1}
# loss = log(1 + exp(-y * f(x)))
# -----------------------
def logistic_loss_from_logits(logits: torch.Tensor, labels_pm1: torch.Tensor):
    return torch.log1p(torch.exp(-labels_pm1 * logits)).mean()

# -----------------------
# Training
# -----------------------
def train_until_zero_loss(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                          max_epochs=20000, lr=0.1, tol_loss=1e-7, device='cpu', verbose=False):
    """
    Train model on (X,y) until training loss is near zero or max_epochs reached.
    Uses SGD (to somewhat mimic gradient flow behavior). Returns trained model.
    """
    model.to(device)
    X = X.to(device)
    y = y.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        logits = model(X)  # shape (m,)
        loss = logistic_loss_from_logits(logits, y)
        loss_val = float(loss.item())
        if epoch % 500 == 0 and verbose:
            # compute fraction correct too
            with torch.no_grad():
                preds = (logits.detach() > 0).float() * 2 - 1  # {-1,+1}
                acc = ((preds == y).float().mean().item())
            print(f"epoch {epoch:5d} loss {loss_val:.3e} acc {acc:.4f}")
        if loss_val < tol_loss:
            if verbose:
                print(f"Converged at epoch {epoch}, loss {loss_val:.3e}")
            break
        loss.backward()
        optimizer.step()
    return model

# -----------------------
# Margin samples and perturbation
# -----------------------
def margin_samples(model: nn.Module, X: torch.Tensor, y: torch.Tensor, device='cpu', slack_ratio=1.1):
    """
    Compute margin = min_i y_i * N(x_i). Return indices of samples on the margin
    defined by y_i * N(x_i) <= slack_ratio * min_margin.
    """
    model.to(device)
    X = X.to(device)
    y = y.to(device)
    with torch.no_grad():
        logits = model(X)  # shape (m,)
        margins = (logits * y)  # shape (m,)
        min_margin = torch.min(margins).item()
        thr = slack_ratio * min_margin
        indices = torch.where(margins <= thr)[0].cpu().numpy().tolist()
    return indices, margins.cpu().numpy()

def compute_z_direction(X: torch.Tensor, y: torch.Tensor, indices: List[int]) -> torch.Tensor:
    """
    z := sum_{i in I} y_i * x_i
    """
    if len(indices) == 0:
        return torch.zeros(X.shape[1], dtype=X.dtype)
    selected = X[indices]  # (k, d)
    ys = y[indices].unsqueeze(1)  # (k,1)
    z = (ys * selected).sum(dim=0)  # (d,)
    return z

def minimal_c_to_flip(model: nn.Module, X: torch.Tensor, y: torch.Tensor, indices: List[int],
                      z: torch.Tensor, device='cpu', c_tol=1e-4, c_max=None) -> float:
    """
    Find minimal c > 0 s.t. for all i in indices,
      sign( N( x_i - y_i * c * (z/||z||) ) ) != sign(y_i)
    i.e., the perturbed point flips the output.
    We search c via binary search between 0 and c_max (if None, set to 4*sqrt(d)).
    Returns c (float). If cannot flip up to c_max, returns np.inf.
    """
    if len(indices) == 0:
        return float('nan')
    d = X.shape[1]
    if c_max is None:
        c_max = 4.0 * math.sqrt(d)
    device = device
    model.to(device)
    X = X.to(device)
    y = y.to(device)
    z = z.to(device)
    z_norm = z.norm().item()
    if z_norm == 0:
        return float('inf')
    z_unit = z / (z_norm + 1e-12)

    # helper: check if c flips all indices
    def flips_all(c: float) -> bool:
        # x' = x_i - y_i * c * z_unit
        with torch.no_grad():
            perturb = (y[indices].unsqueeze(1) * (-c)) * z_unit.unsqueeze(0)  # because -y * c * z_unit
            Xp = X[indices] + perturb
            logits = model(Xp)
            # sign != y means logits * y < 0
            vals = (logits * y[indices])
            return bool(torch.all(vals < 0).item())

    # quick check if c_max suffices
    if not flips_all(c_max):
        return float('inf')

    lo, hi = 0.0, c_max
    while hi - lo > c_tol:
        mid = 0.5 * (lo + hi)
        if flips_all(mid):
            hi = mid
        else:
            lo = mid
    return float(hi)

# -----------------------
# Single experiment runner
# -----------------------
def run_single_experiment(d: int, m: int, width: int, seed: int = 0, verbose=False, device='cpu'):
    X, y = sample_uniform_sphere(d, m, radius=math.sqrt(d), seed=seed)
    # convert labels to {-1,+1} torch
    y_t = y.clone()
    model = Depth2ReLU(input_dim=d, width=width)
    # train until zero loss
    model = train_until_zero_loss(model, X, y_t, max_epochs=20000, lr=0.1, tol_loss=1e-7, device=device, verbose=verbose)
    # find margin samples
    I, margins = margin_samples(model, X, y_t, device=device, slack_ratio=1.1)
    z = compute_z_direction(X, y_t, I)
    c = minimal_c_to_flip(model, X, y_t, I, z, device=device, c_tol=1e-4, c_max=4.0*math.sqrt(d))
    # Also return fraction of margin samples
    frac_on_margin = len(I) / float(m)
    return {
        'd': d,
        'm': m,
        'width': width,
        'seed': seed,
        'min_pert_c': c,
        'z_norm': float(z.norm().item()),
        'frac_on_margin': frac_on_margin,
        'num_margin': len(I),
        'margins': margins
    }

# -----------------------
# Sweep experiments and plot
# -----------------------
def sweep_and_plot(ds: List[int], alphas: List[float], width=1000, seeds=[0,1,2,3,4], device='cpu'):
    """
    Reproduce plots similar to Figure 1(a) and 1(b) in paper:
      - minimal perturbation size vs dimension for different m scaling (m = d^alpha)
      - minimal perturbation size vs width for m = d
    """
    results = []
    for alpha in alphas:
        for d in ds:
            m = max(4, int(round(d ** alpha)))
            cs = []
            for seed in seeds:
                r = run_single_experiment(d=d, m=m, width=width, seed=seed, verbose=False, device=device)
                cs.append(r['min_pert_c'])
                results.append(r)
                print(f"alpha {alpha:.2f} d {d} m {m} seed {seed} c {r['min_pert_c']:.4f} frac_margin {r['frac_on_margin']:.3f}")
            # average over seeds:
            mean_c = np.nanmean([c if np.isfinite(c) else np.nan for c in cs])
            std_c  = np.nanstd([c if np.isfinite(c) else np.nan for c in cs])
            print(f"alpha {alpha:.2f} d {d} mean_c {mean_c:.4f} std {std_c:.4f}")
    # plotting minimal perturbation size normalized with sqrt(d)
    plt.figure(figsize=(7,5))
    for alpha in alphas:
        ds_plot = []
        cs_plot = []
        for d in ds:
            m = max(4, int(round(d ** alpha)))
            # collect seeds results
            cs = [r['min_pert_c'] for r in results if r['d']==d and r['m']==m]
            if len(cs) == 0:
                continue
            mean_c = np.nanmean([c if np.isfinite(c) else np.nan for c in cs])
            ds_plot.append(d)
            cs_plot.append(mean_c)
        plt.plot(ds_plot, cs_plot, marker='o', label=f"m = d^{alpha}")
    # plot sqrt(d)
    ds_float = np.array(ds)
    plt.plot(ds_float, np.sqrt(ds_float), '--', label=r'$\sqrt{d}$ (trivial bound)')
    plt.xlabel("input dimension d")
    plt.ylabel("minimal perturbation c (mean over seeds)")
    plt.legend()
    plt.title("Minimal perturbation size vs input dim (reproduction)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("my_fig.png")

    return results

# -----------------------
# Main CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['single', 'sweep'], default='single')
    parser.add_argument('--d', type=int, default=200)
    parser.add_argument('--m', type=int, default=None)
    parser.add_argument('--width', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    if args.mode == 'single':
        m = args.m if args.m is not None else args.d
        out = run_single_experiment(d=args.d, m=m, width=args.width, seed=args.seed, verbose=True, device=args.device)
        print("Result:", out)
    else:
        # quick sweep similar to fig 1(a) in the paper
        ds = list(range(100, 1001, 100))
        #ds = list(range(100, 1001, 100))

        alphas = [0.5, 0.8, 1.0, 1.2, 1.5]
        sweep_and_plot(ds, alphas, width=args.width, seeds=[0,1,2,3,4], device=args.device)
