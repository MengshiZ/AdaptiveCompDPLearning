from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np


class NaiveDPQuery:
    def __init__(self, dim: int, epsilon: float, delta: float, seed: int | None = None):
        self.dim = dim
        self.epsilon = epsilon
        self.delta = delta
        self.rng = np.random.default_rng(seed)
        self.sigma = math.sqrt(math.log(1.25 / self.delta)) / self.epsilon

        self.value = np.zeros(self.dim, dtype=float)
        self.t = 0

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.dim,):
            raise ValueError(f"Expected shape {(self.dim,)}, got {x.shape}")

        self.t += 1
        self.value = x.copy()

    def single_query(self) -> np.ndarray:
        fresh_noise = self.rng.normal(0.0, self.sigma, size=self.dim)
        return self.value + fresh_noise


class DPPreSumQuery:
    def __init__(self, dim: int, epsilon: float, delta: float, seed: int | None = None):
        self.dim = dim
        self.epsilon = epsilon
        self.delta = delta
        self.rng = np.random.default_rng(seed)
        self.sigma = 2 * math.sqrt(2 * math.log(2.5 / self.delta)) / self.epsilon

        self.t = 0
        self.true_prefix = np.zeros(self.dim, dtype=float)
        self.checkpoint_noise: Dict[int, np.ndarray] = {}
        self.tail_tree_noise: Dict[int, Dict[Tuple[int, int], np.ndarray]] = defaultdict(dict)
        self.current_base: int | None = None
        self.prev_query: np.ndarray | None = None

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.dim,):
            raise ValueError(f"Expected shape {(self.dim,)}, got {x.shape}")

        self.t += 1
        t = self.t

        if t == 1:
            self.true_prefix = x.copy()
            self.prev_query = np.zeros(self.dim, dtype=float)
        else:
            self.prev_query = self.query()
            self.true_prefix += x.copy()

        if (t + 1) & t == 0:
            base = t
            self.checkpoint_noise[base] = self.rng.normal(0.0, self.sigma, size=self.dim)
            self.tail_tree_noise[base] = {}
            self.current_base = base
            return

        if self.current_base is None:
            return

        base = self.current_base
        offset = t - (base + 1)

        o = offset
        while o >= 0:
            lowbit = (o + 1) & (-(o + 1))
            length = lowbit
            block_level = length.bit_length() - 1
            start_offset = o + 1 - length
            index = start_offset // length
            key = (block_level, index)

            if key not in self.tail_tree_noise[base]:
                self.tail_tree_noise[base][key] = self.rng.normal(
                    0.0,
                    self.sigma * (math.log(self.current_base)),
                    size=self.dim,
                )

            o = start_offset - 1

    def query(self) -> np.ndarray:
        tau = self.t
        if tau < 1 or tau > self.t:
            raise ValueError("Invalid query time")

        result = self.true_prefix.copy()

        m = tau + 1
        largest_pow_two = 1 << (m.bit_length() - 1)
        base = largest_pow_two - 1

        checkpoint = self.checkpoint_noise.get(base)
        if checkpoint is not None:
            result += checkpoint

        if tau > base:
            offset = tau - (base + 1)
            o = offset
            while o >= 0:
                lowbit = (o + 1) & (-(o + 1))
                length = lowbit
                block_level = length.bit_length() - 1
                start_offset = o + 1 - length
                index = start_offset // length
                key = (block_level, index)
                noise = self.tail_tree_noise[base].get(key)
                if noise is not None:
                    result += noise
                o = start_offset - 1

        return result

    def single_query(self) -> np.ndarray:
        if self.prev_query is None:
            raise RuntimeError("Call update() before single_query().")
        return self.query() - self.prev_query
