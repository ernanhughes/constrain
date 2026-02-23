from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Any, Dict
import math
import time
from collections import deque

import torch
import torch.nn as nn

from constrain.control.rl_parameter_adapter import RLParameterAdapter, AdapterObservation, AdapterAction


@dataclass
class EvalConfig:
    checkpoint: str
    steps: int = 2000
    batch_size: int = 32
    seeds: List[int] = None
    leakage: float = 0.9
    block_sizes: Tuple[int, ...] = (32, 16)
    mixture_alphas: Tuple[float, ...] = (0.1, 0.5, 5.0)
    static_tau: float = 8.0
    static_temp: float = 1.0
    static_rec: int = 4


@dataclass
class RunMetrics:
    env: str
    model: str  # S/R/G
    seed: int
    collapse_rate: float
    performance_proxy: float
    fallback_rate: float
    avg_jitter: float


class SimpleHRM(nn.Module):
    def __init__(self, vocab_size: int = 1000, hidden_size: int = 128, leak: float = 0.9):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, vocab_size)
        self.leak = leak
        self.hidden = None
        self.unc = deque(maxlen=2000)

    def reset(self):
        self.hidden = None

    def forward(self, x):
        emb = self.embedding(x)
        if self.hidden is None:
            self.hidden = torch.zeros(1, x.size(0), self.rnn.hidden_size, device=x.device)
        self.hidden = self.leak * self.hidden
        out, self.hidden = self.rnn(emb, self.hidden)
        logits = self.head(out[:, -1, :])
        probs = torch.softmax(logits, dim=-1)
        uncertainty = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
        self.unc.append(float(uncertainty.detach().cpu().item()))
        return logits

    def uncertainty(self) -> float:
        return self.unc[-1] if self.unc else 0.0

    def uncertainty_stats(self, window: int = 100):
        if len(self.unc) < 2:
            return 0.0, 0.0
        w = min(window, len(self.unc))
        recent = list(self.unc)[-w:]
        m = sum(recent) / w
        if w < 2:
            return m, 0.0
        v = sum((x - m) ** 2 for x in recent) / (w - 1)
        return m, math.sqrt(v)


def _sample_batch(task: str, batch_size: int, seq_len: int = 32) -> torch.Tensor:
    if task == "grounded":
        return torch.randint(0, 50, (batch_size, seq_len))
    if task == "creative":
        return torch.randint(0, 1000, (batch_size, seq_len))
    # ambiguous
    base = torch.randint(0, 500, (batch_size, 1))
    return (base + torch.randint(0, 50, (batch_size, seq_len))) % 1000


def eval_one(
    cfg: EvalConfig,
    adapter: RLParameterAdapter,
    model_id: str,  # S/R/G
    schedule_type: str,
    block_size: int,
    alpha: float,
    seed: int,
    steps_csv_writer=None,
) -> RunMetrics:
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleHRM(leak=cfg.leakage).to(device)
    crit = nn.CrossEntropyLoss()

    # simple schedule: blocks rotate tasks; mixture samples with alpha-ish weights
    tasks = ["grounded", "creative", "ambiguous"]
    energy_hist: List[float] = []
    fallbacks = 0
    collapses = 0

    # jitter tracking on action space
    last_action: Optional[AdapterAction] = None
    jitter_vals: List[float] = []

    for t in range(cfg.steps):
        if schedule_type == "blocks":
            if t > 0 and (t % block_size == 0):
                model.reset()
            task = tasks[(t // block_size) % len(tasks)]
        else:
            # crude dirichlet-ish: alpha small => peaky weights
            # we just implement a simple temperatured categorical
            if alpha <= 0.2:
                probs = [0.8, 0.1, 0.1]
            elif alpha <= 0.5:
                probs = [0.45, 0.30, 0.25]
            else:
                probs = [1/3, 1/3, 1/3]
            task = tasks[int(torch.multinomial(torch.tensor(probs), 1).item())]

        x = _sample_batch(task, cfg.batch_size).to(device)
        y = x[:, 0]

        logits = model(x)
        loss = crit(logits, y)

        unc = model.uncertainty()
        prev = energy_hist[-1] if energy_hist else unc
        delta = unc - prev
        energy_hist.append(unc)

        # Build observation (map uncertainty -> energy proxy)
        obs = AdapterObservation(
            total_energy=float(unc),
            delta_energy=float(delta),
            entropy=float(unc),
            iteration_progress=float(t / max(1, cfg.steps)),
        )

        # Choose action
        if model_id == "S":
            action = AdapterAction(cfg.static_tau, cfg.static_temp, cfg.static_rec).clamp(adapter.bounds)
            fallback = False
        elif model_id == "R":
            # ungated (clamp only)
            action = adapter.propose_action_ungated(obs)
            fallback = False
        else:
            gate = adapter.get_safe_action(obs, energy_hist)
            action = gate.action
            fallback = gate.fallback_triggered
            fallbacks += 1 if fallback else 0

        if last_action is not None:
            dv = torch.tensor([action.tau_hard, action.temperature, action.recursion_limit], dtype=torch.float32) - \
                 torch.tensor([last_action.tau_hard, last_action.temperature, last_action.recursion_limit], dtype=torch.float32)
            jitter_vals.append(float(torch.norm(dv).item()))
        last_action = action

        # Collapse rule: 3σ + tau_hard as additional cap
        mean, std = model.uncertainty_stats(100)
        thr = mean + 3.0 * std
        tau_thr = action.tau_hard  # interpret tau as an upper bound on uncertainty proxy
        if (len(model.unc) >= 100) and (unc > thr or unc > tau_thr):
            collapses += 1

        # recursion_limit effect: if low, reset more often
        if action.recursion_limit <= 2 and (t % 8 == 0):
            model.reset()

    collapse_rate = collapses / cfg.steps
    perf = -float(sum(energy_hist) / max(1, len(energy_hist)))  # higher is better
    fallback_rate = (fallbacks / cfg.steps) if model_id == "G" else 0.0
    avg_jitter = float(sum(jitter_vals) / max(1, len(jitter_vals)))

    env_name = f"{schedule_type}_{block_size}" if schedule_type == "blocks" else f"mixture_{alpha}"
    return RunMetrics(
        env=env_name,
        model=model_id,
        seed=seed,
        collapse_rate=float(collapse_rate),
        performance_proxy=float(perf),
        fallback_rate=float(fallback_rate),
        avg_jitter=float(avg_jitter),
    )
