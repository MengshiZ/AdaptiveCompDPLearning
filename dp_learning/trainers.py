from __future__ import annotations

import gc

import torch
from torch.func import functional_call, grad, vmap

from .dp_queries import DPPreSumQuery, NaiveDPQuery


class StandardSGD:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device

    def train_batch(self, x: torch.Tensor, y: torch.Tensor, loss_fn) -> float:
        self.model.train()
        x, y = x.to(self.device), y.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        preds = self.model(x)
        loss = loss_fn(preds, y)
        loss.backward()
        self.optimizer.step()

        return float(loss.item())


class NormalTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        max_grad_norm: float,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.max_grad_norm = max_grad_norm

    def train_batch(self, x: torch.Tensor, y: torch.Tensor, loss_fn) -> float:
        self.model.train()
        x, y = x.to(self.device), y.to(self.device)
        batch_size = x.shape[0]

        params_dict = dict(self.model.named_parameters())
        buffers_dict = dict(self.model.named_buffers())
        param_names = list(params_dict.keys())

        def loss_per_sample(params, buffers, x_s, y_s):
            preds = functional_call(self.model, (params, buffers), (x_s.unsqueeze(0),))
            loss = loss_fn(preds, y_s.unsqueeze(0))
            return loss.mean()

        with torch.no_grad():
            report_loss = loss_fn(self.model(x), y).item()

        grad_fn = grad(loss_per_sample)
        per_sample_grads = vmap(grad_fn, in_dims=(None, None, 0, 0))(
            params_dict,
            buffers_dict,
            x,
            y,
        )

        flat_list = [per_sample_grads[name].reshape(batch_size, -1) for name in param_names]
        flat = torch.cat(flat_list, dim=1)

        norms = flat.norm(2, dim=1)
        factors = torch.clamp(self.max_grad_norm / (norms + 1e-6), max=1.0)
        flat = flat * factors.unsqueeze(1)
        avg = flat.mean(dim=0)

        self.optimizer.zero_grad(set_to_none=True)
        idx = 0
        for name in param_names:
            p = params_dict[name]
            n = p.numel()
            g = avg[idx : idx + n].reshape_as(p)
            p.grad = g.detach().clone()
            idx += n

        self.optimizer.step()
        return report_loss


class DPSGDTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epsilon: float,
        delta: float,
        max_grad_norm: float,
        dp_mechanism: str = "Naive",
        device: str = "cpu",
        seed: int = 0,
        expected_batch_size: int | None = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.epsilon = epsilon
        self.delta = delta
        self.seed = seed
        self.max_grad_norm = max_grad_norm
        self.dp_mechanism = dp_mechanism
        self.expected_batch_size = expected_batch_size

        self.grad_dim = sum(p.numel() for p in self.model.parameters())

        self.mechanism = self._new_mechanism()
        self.step = 0

    def _new_mechanism(self):
        if self.dp_mechanism == "DPPreSum":
            return DPPreSumQuery(
                dim=self.grad_dim,
                epsilon=self.epsilon,
                delta=self.delta,
                seed=self.seed,
            )
        return NaiveDPQuery(
            dim=self.grad_dim,
            epsilon=self.epsilon,
            delta=self.delta,
            seed=self.seed,
        )

    def reset_presum(self) -> None:
        self.mechanism = None
        self.step = 0
        gc.collect()
        self.mechanism = self._new_mechanism()

    def train_batch(self, x: torch.Tensor, y: torch.Tensor, loss_fn) -> float:
        self.model.train()
        x, y = x.to(self.device), y.to(self.device)
        batch_size = x.shape[0]

        params_dict = dict(self.model.named_parameters())
        buffers_dict = dict(self.model.named_buffers())
        param_names = list(params_dict.keys())

        def loss_per_sample(params, buffers, x_s, y_s):
            preds = functional_call(
                self.model,
                (params, buffers),
                (x_s.unsqueeze(0),),
            )
            return loss_fn(preds, y_s.unsqueeze(0))

        grad_fn = grad(loss_per_sample)
        per_sample_grads = vmap(grad_fn, in_dims=(None, None, 0, 0))(
            params_dict,
            buffers_dict,
            x,
            y,
        )

        flat_grads_list = []
        for name in param_names:
            g = per_sample_grads[name]
            flat_grads_list.append(g.reshape(batch_size, -1))
        flat_grads = torch.cat(flat_grads_list, dim=1)

        grad_norms = torch.norm(flat_grads, p=2, dim=1)
        clip_factors = torch.clamp(
            self.max_grad_norm / (grad_norms + 1e-6),
            max=1.0,
        )
        flat_grads = flat_grads * clip_factors.unsqueeze(1)

        summed_grad = flat_grads.sum(dim=0)

        summed_grad_np = summed_grad.detach().cpu().numpy()
        self.mechanism.update(summed_grad_np)
        noisy_sum = self.mechanism.single_query()

        self.step += 1

        denom = self.expected_batch_size or batch_size
        noisy_avg_grad = torch.tensor(
            noisy_sum / denom,
            device=self.device,
            dtype=summed_grad.dtype,
        )

        self.optimizer.zero_grad()

        current_idx = 0
        for name in param_names:
            param = params_dict[name]
            numel = param.numel()

            grad_slice = noisy_avg_grad[current_idx : current_idx + numel]

            if param.grad is None:
                param.grad = grad_slice.reshape(param.shape).detach().clone()
            else:
                param.grad.copy_(grad_slice.reshape(param.shape))

            current_idx += numel

        self.optimizer.step()

        with torch.no_grad():
            loss = loss_fn(self.model(x), y)

        return float(loss.item())
