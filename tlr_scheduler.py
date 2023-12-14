
from math import inf

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class TLRScheduler(LRScheduler):

    def __init__(self, optimizer: Optimizer, eta_1: float = .25,
                 eta_2: float = .75, gamma_1: float = .5,
                 gamma_2: float = 2.0, patience: int = 10) -> None:
        super(TLRScheduler, self).__init__(optimizer)
        self.eta_1, self.eta_2 = eta_1, eta_2
        self.gamma_1, self.gamma_2 = gamma_1, gamma_2
        self.patience = patience
        self._rho = inf
        self._step_count = 0
        self._lr = optimizer.param_groups[0]['lr']
        self._prev_weights = self._gather_flat_param().clone()

    def _gather_flat_param(self):
        params = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.data.is_sparse:
                    p = p.data.to_dense().view(-1)
                else:
                    p = p.data.view(-1)
                params.append(p)
        return torch.cat(params)

    def _update_rho(self, loss: float) -> None:
        actual_reduction = self._prev_loss - loss

        # Compute norm of the steps taken
        current_weights = self._gather_flat_param().clone()
        norm_d = torch.linalg.norm(self._prev_weights - current_weights)
        expected_reduction = self._lr * norm_d

        # Compute rho
        self._rho = actual_reduction / expected_reduction

        # Update cached weights
        self._prev_weights = current_weights

    def _update_lr(self) -> None:
        if self._rho < self.eta_1:
            self._lr = self._lr * self.gamma_1
        elif self._rho > self.eta_2:
            self._lr = self._lr * self.gamma_2

        for group in self.optimizer.param_groups:
            group['lr'] = self._lr

    def get_lr(self) -> float:
        return self._lr

    def step(self, loss: float = None) -> None:
        if loss is None and self._step_count == 0:
            self._step_count += 1
            return
        if loss is None:
            raise ValueError('loss should not be None.')

        if self._step_count == 1:
            self._prev_loss = loss
            self._step_count += 1
            return
        
        self._step_count += 1
        if self._step_count % self.patience == 0:
            self._update_rho(loss)
            self._update_lr()
            self._prev_loss = loss
