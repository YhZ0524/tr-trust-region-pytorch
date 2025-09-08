import os, argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tr_optimizer import TRNewtonCG
from utils import MetricsLogger, set_seed, accuracy_from_logits

def make_synth(task="logistic", n=20000, d=50, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    w_true = rng.normal(size=(d, 1)).astype(np.float32)
    bias = rng.normal(scale=0.3, size=(1,)).astype(np.float32)
    logits = X @ w_true + bias
    if task == "logistic":
        p = 1/(1+np.exp(-logits))
        y = (rng.uniform(size=(n,1)) < p).astype(np.int64)
        C = 1
    else:
        K = 3
        W = rng.normal(size=(d, K)).astype(np.float32)
        b = rng.normal(scale=0.3, size=(K,)).astype(np.float32)
        scores = X @ W + b
        exp = np.exp(scores - scores.max(axis=1, keepdims=True))
        p = exp / exp.sum(axis=1, keepdims=True)
        y = np.array([rng.choice(3, p=p[i]) for i in range(n)], dtype=np.int64).reshape(-1, 1)
        C = 3
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)
    return torch.from_numpy(X), torch.from_numpy(y), C

class LogReg(nn.Module):
    def __init__(self, d, C=1):
        super().__init__()
        self.linear = nn.Linear(d, 1 if C==1 else C)
        self.C = C
    def forward(self, x):
        return self.linear(x)

def criterion_fn(logits, y, C=1):
    return (nn.functional.binary_cross_entropy_with_logits(logits, y.float())
            if C==1 else nn.functional.cross_entropy(logits, y.view(-1)))

def run(task="logistic", optimizer="tr", seed=42, batch_size=1024, steps=300, lr=0.01, outdir="results_logreg"):
    set_seed(seed)
    X, y, C = make_synth(task=task, n=20000, d=50, seed=seed)
    n = X.shape[0]; split = int(0.8 * n)
    Xtr, Ytr = X[:split], y[:split]
    Xva, Yva = X[split:], y[split:]

    train_ds = TensorDataset(Xtr, Ytr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")))
    model = LogReg(X.shape[1], C=C).to(device)

    if optimizer == "tr":
        opt = TRNewtonCG(model.parameters(), delta0=1.0, precond="diag", amp=False)
    elif optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer == "lbfgs":
        opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, line_search_fn=None)
    else:
        raise ValueError("optimizer must be one of: tr, adamw, lbfgs")

    out_dir = os.path.join(outdir, f"{task}_{optimizer}_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    logger = MetricsLogger(out_dir)

    Xva = Xva.to(device); Yva = Yva.to(device)

    step_count = 0
    while step_count < steps:
        for xb, yb in train_loader:
            if step_count >= steps: break
            xb = xb.to(device); yb = yb.to(device)

            if optimizer == "tr":
                def closure():
                    logits = model(xb)
                    loss = criterion_fn(logits, yb, C=C)
                    return loss
                loss = opt.step(closure)
                last = opt.state.get('last', {})
                with torch.no_grad():
                    model.eval()
                    logits_va = model(Xva)
                    val_loss = criterion_fn(logits_va, Yva, C=C).item()
                    val_acc  = accuracy_from_logits(logits_va, Yva)
                    model.train()
                logger.log(step=step_count, train_loss=float(last.get('new_loss', loss.item())),
                           val_loss=val_loss, val_acc=val_acc,
                           accepted=int(last.get('accepted', 0)),
                           rho=float(last.get('rho', 0.0)),
                           delta=float(last.get('delta', 0.0)),
                           cg_iters=int(last.get('cg_iters', 0)))
            elif optimizer == "lbfgs":
                def closure():
                    opt.zero_grad()
                    logits = model(xb)
                    loss = criterion_fn(logits, yb, C=C)
                    loss.backward()
                    return loss
                loss = opt.step(closure)
                with torch.no_grad():
                    model.eval()
                    logits_va = model(Xva)
                    val_loss = criterion_fn(logits_va, Yva, C=C).item()
                    val_acc  = accuracy_from_logits(logits_va, Yva)
                    model.train()
                logger.log(step=step_count, train_loss=float(loss.item()),
                           val_loss=val_loss, val_acc=val_acc)
            else:
                opt.zero_grad()
                logits = model(xb)
                loss = criterion_fn(logits, yb, C=C)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    model.eval()
                    logits_va = model(Xva)
                    val_loss = criterion_fn(logits_va, Yva, C=C).item()
                    val_acc  = accuracy_from_logits(logits_va, Yva)
                    model.train()
                logger.log(step=step_count, train_loss=float(loss.item()),
                           val_loss=val_loss, val_acc=val_acc)

            step_count += 1

    # quick plot
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.read_csv(os.path.join(out_dir, "metrics.csv"))
    plt.figure(figsize=(7,5))
    plt.plot(df["time_s"], df["val_loss"], label="val_loss")
    plt.xlabel("time (s)"); plt.ylabel("val_loss"); plt.title(f"{task} - {optimizer}")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_vs_time.png"), dpi=160)
    plt.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="logistic", choices=["logistic","softmax"])
    ap.add_argument("--optimizer", type=str, default="tr", choices=["tr","adamw","lbfgs"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--outdir", type=str, default="results_logreg")
    args = ap.parse_args()
    run(**vars(args))
