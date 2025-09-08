from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, re

base = Path.home() / "trncg" / "trncg_pytorch_suite"

def load_last_clean(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"missing: {csv_path}")
    df = pd.read_csv(csv_path)
    for col in ["time_s","val_loss","val_acc","accepted","cg_iters","rho","delta","train_loss"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time_s"]).reset_index(drop=True)
    t = df["time_s"].to_numpy()
    cuts = np.where(np.diff(t) < 0)[0]
    if len(cuts):
        df = df.iloc[cuts[-1]+1:].reset_index(drop=True)
        t = df["time_s"].to_numpy()
    df["t"] = t - t[0]
    return df

def save_logistic(tr_csv, aw_csv, out_png, thr=0.17):
    tr = load_last_clean(tr_csv)
    aw = load_last_clean(aw_csv)
    def t_to_thr(df):
        ok = df[df["val_loss"] <= thr]
        return float(ok["t"].iloc[0]) if len(ok) else float("inf")
    t_tr, t_aw = t_to_thr(tr), t_to_thr(aw)
    plt.figure(figsize=(7,5))
    plt.plot(tr["t"], tr["val_loss"], label="TR-Newton-CG")
    plt.plot(aw["t"], aw["val_loss"], label="AdamW")
    plt.axhline(thr, ls=":", alpha=.6)
    for x,name in [(t_tr,"TR"),(t_aw,"AdamW")]:
        if np.isfinite(x):
            plt.axvline(x, ls="--", alpha=.6)
            plt.text(x, thr+0.005, f"{name}@{x:.2f}s")
    plt.xlabel("time (s)"); plt.ylabel("val_loss"); plt.title("Logistic Regression"); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=160); plt.close()
    return t_tr, t_aw

def save_mnist(tr_csv, aw_csv, out_png, thr_acc=0.97):
    tr = load_last_clean(tr_csv)
    aw = load_last_clean(aw_csv)
    def t_to_acc(df):
        ok = df[df["val_acc"] >= thr_acc]
        return float(ok["t"].iloc[0]) if len(ok) else float("inf")
    t_tr, t_aw = t_to_acc(tr), t_to_acc(aw)
    plt.figure(figsize=(7,5))
    plt.plot(tr["t"], tr["val_acc"], label="TR-Newton-CG")
    plt.plot(aw["t"], aw["val_acc"], label="AdamW")
    plt.axhline(thr_acc, ls=":", alpha=.6)
    for x,name in [(t_tr,"TR"),(t_aw,"AdamW")]:
        if np.isfinite(x):
            plt.axvline(x, ls="--", alpha=.6)
            plt.text(x, thr_acc+0.002, f"{name}@{x:.2f}s")
    plt.xlabel("time (s)"); plt.ylabel("val_acc"); plt.title("MNIST (1 epoch)"); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=160); plt.close()
    return t_tr, t_aw

def fmt_time(x): return f"{x:.2f}s" if np.isfinite(x) else "not reached"

log_tr = base/"results_logreg/logistic_tr_seed42/metrics.csv"
log_aw = base/"results_logreg/logistic_adamw_seed42/metrics.csv"
png_log = Path("figures/logreg_tr_vs_adamw.png")
t_tr_loss, t_aw_loss = save_logistic(log_tr, log_aw, png_log)

png_mnist = Path("figures/mnist_tr_vs_adamw.png")
mnist_ok = False
try:
    mn_tr = base/"results_mnist/mnist_tr_seed42/metrics.csv"
    mn_aw = base/"results_mnist/mnist_adamw_seed42/metrics.csv"
    t_tr_acc, t_aw_acc = save_mnist(mn_tr, mn_aw, png_mnist)
    mnist_ok = True
except Exception as e:
    t_tr_acc = t_aw_acc = float("inf")

rd_path = Path("README.md")
rd = rd_path.read_text(encoding="utf-8") if rd_path.exists() else ""

log_block = f"""
## Results (Logistic Regression)
- Time-to-0.17 loss: **TR {fmt_time(t_tr_loss)}** vs **AdamW {fmt_time(t_aw_loss)}**.

![TR vs AdamW](figures/logreg_tr_vs_adamw.png)
"""

if "figures/logreg_tr_vs_adamw.png" not in rd:
    rd += "\n" + log_block

if mnist_ok:
    mnist_block = f"""
## Results (MNIST)
- Time-to-97% accuracy (1 epoch): **TR {fmt_time(t_tr_acc)}** vs **AdamW {fmt_time(t_aw_acc)}**.

![MNIST TR vs AdamW](figures/mnist_tr_vs_adamw.png)
"""
    if "## Results (MNIST)" in rd:
        rd = re.sub(r"## Results \(MNIST\)[\s\S]*?(?=\n## |\Z)", mnist_block.strip(), rd, count=1)
    else:
        rd += "\n" + mnist_block

rd_path.write_text(rd.strip()+"\n", encoding="utf-8")
print("Saved figures and updated README.")
