"""Benchmark: MLP vs CNN-A vs CNN-B inference speed on CPU."""

import time
import torch
import torch.nn as nn

# ── Model definitions ──

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(126, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 7),
        )
    def forward(self, x):
        return self.net(x)

class CNN_A(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 7, 256), nn.ReLU(),
            nn.Linear(256, 7),
        )
    def forward(self, x):
        return self.fc(self.conv(x))

class CNN_B(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 7, 256), nn.ReLU(),
            nn.Linear(256, 7),
        )
    def forward(self, x):
        return self.fc(self.conv(x))

# ── Benchmark ──

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def bench(model, x, n_iter=1000, warmup=100):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            model(x)
        elapsed = time.perf_counter() - t0
    return elapsed / n_iter * 1000  # ms per forward

models = {
    "MLP":   MLP(),
    "CNN-A": CNN_A(),
    "CNN-B": CNN_B(),
}

print("=" * 60)
print("Model Parameter Counts")
print("=" * 60)
for name, m in models.items():
    print(f"  {name:8s}: {count_params(m):>10,} params")

for bs_label, bs in [("batch=128 (training)", 128), ("batch=1 (eval)", 1)]:
    x = torch.randn(bs, 3, 6, 7)
    print()
    print("=" * 60)
    print(f"Inference Speed  --  {bs_label}  (1000 iters, CPU)")
    print("=" * 60)
    for name, m in models.items():
        ms = bench(m, x)
        print(f"  {name:8s}: {ms:.3f} ms/forward")

print()
print("Done.")
