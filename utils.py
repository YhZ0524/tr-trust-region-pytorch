import os, csv, time, random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)

class MetricsLogger:
    def __init__(self, out_dir, filename="metrics.csv"):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, filename)
        self.start = time.time()
        self.header_written = False
        self.header = None
    def log(self, **kwargs):
        kwargs['time_s'] = time.time() - self.start
        if self.header is None:
            self.header = list(kwargs.keys())
        write_header = (not self.header_written) or (not os.path.exists(self.path))
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.header)
            if write_header:
                w.writeheader(); self.header_written = True
            w.writerow(kwargs)

def accuracy_from_logits(logits, y_true):
    if logits.shape[1] == 1:
        pred = (logits.sigmoid() > 0.5).long().view(-1)
    else:
        pred = logits.argmax(dim=1)
    return (pred == y_true.view(-1)).float().mean().item()
