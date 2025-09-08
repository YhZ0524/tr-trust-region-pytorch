import os, argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tr_optimizer import TRNewtonCG
from utils import MetricsLogger, set_seed

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*7*7, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

def run(optimizer="tr", seed=42, batch_size=128, epochs=1, lr=1e-3, outdir="results_mnist"):
    set_seed(seed)
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")))

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=512, shuffle=False)

    model = SmallCNN().to(device)

    if optimizer == "tr":
        opt = TRNewtonCG(model.parameters(), delta0=1.0, precond="diag", amp=False)
        def step_batch(xb, yb):
            def closure():
                logits = model(xb)
                loss = nn.functional.cross_entropy(logits, yb)
                return loss
            return opt.step(closure), opt.state.get('last', {})
    elif optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        def step_batch(xb, yb):
            opt.zero_grad()
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            return loss, {}
    else:
        raise ValueError("optimizer must be one of: tr, adamw")

    out_dir = os.path.join(outdir, f"mnist_{optimizer}_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    logger = MetricsLogger(out_dir)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            loss, info = step_batch(xb, yb)

            # quick validation
            model.eval()
            correct = 0; total = 0; val_loss = 0.0
            with torch.no_grad():
                for xt, yt in test_loader:
                    xt = xt.to(device); yt = yt.to(device)
                    logits = model(xt)
                    val_loss += nn.functional.cross_entropy(logits, yt, reduction='sum').item()
                    pred = logits.argmax(dim=1)
                    correct += (pred == yt).sum().item(); total += yt.size(0)
            val_loss /= total
            val_acc = correct / total
            model.train()

            logger.log(epoch=epoch, train_loss=float(loss.item()),
                       val_loss=val_loss, val_acc=val_acc,
                       accepted=int(info.get('accepted', 0)),
                       rho=float(info.get('rho', 0.0)),
                       delta=float(info.get('delta', 0.0)),
                       cg_iters=int(info.get('cg_iters', 0)))

    # plot
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.read_csv(os.path.join(out_dir, "metrics.csv"))
    plt.figure(figsize=(7,5))
    plt.plot(df["time_s"], df["val_acc"], label="val_acc")
    plt.xlabel("time (s)"); plt.ylabel("val_acc"); plt.title(f"MNIST - {optimizer}")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acc_vs_time.png"), dpi=160)
    plt.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", type=str, default="tr", choices=["tr","adamw"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--outdir", type=str, default="results_mnist")
    args = ap.parse_args()
    run(**vars(args))
