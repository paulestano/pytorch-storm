from math import isinf, inf
from numpy import log10
import numpy
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.linalg import norm
from argparse import ArgumentError
from typing import Any, List, Optional, Union
from functools import reduce

__all__ = ["STORM1"]


class STORM1(Optimizer):
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1.0,
        momentum: float = 0.0,
        gamma_1: float = 0.25,
        gamma_2: float = 2,
        eta_1: float = 0.25,
        eta_2: float = 0.75,
        dampening: float = 0,
        nesterov: bool = False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        differentiable: bool = False,
        frequency: int = 10,
        loss: callable = None,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            gamma_1=gamma_1,
            gamma_2=gamma_2,
            eta_1=eta_1,
            eta_2=eta_2,
            dampening=dampening,
            nesterov=nesterov,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
            frequency=frequency,
            iter=0,
        )
        super(STORM1, self).__init__(params, defaults)
        self.updates = None
        self.iter = 0
        self.rho = None
        self.loss = loss

    def storm1(
        self,
        params: List[Tensor],
        d_p_list: List[Tensor],
        *,
        maximize: bool,
        lr: float,
    ) -> None:

        alpha = lr / self._norm_d
        
        for i, param in enumerate(params):
            d_p = d_p_list[i] if not maximize else -d_p_list[i]
            param.add_(d_p, alpha=-alpha)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure : A closure that reevaluates the model
                and returns the loss.
        """
        
        flat_grad = []
        for group in self.param_groups:
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            dampening = group["dampening"]
            maximize = group["maximize"]
            frequency = group["frequency"]
            self.iter += 1
            for p in group["params"]:
                if p.grad is not None:
                    d_p = p.grad if not maximize else -p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        buf = None
                    else:
                        buf = state["momentum_buffer"]
                    if d_p.is_sparse:
                        raise RuntimeError(
                            "STORM1 does not support sparse gradients, please consider SparseAdam instead"
                        )
                    if momentum != 0:

                        if buf is None:
                            buf = torch.clone(d_p).detach()
                        else:
                            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                        if nesterov:
                            d_p = d_p.add(buf, alpha=momentum)
                        else:
                            d_p = buf
                    view = d_p.view(-1)    

                
                else:
                    view = p.new(p.numel()).zero_()
                flat_grad.append(view)
        flat_grad = torch.cat(flat_grad, 0)
        self._norm_d = torch.linalg.norm(flat_grad)

        alpha = group["lr"] / self._norm_d

        if self.updates is None:
            self.updates = alpha * flat_grad.clone()
        else:
            self.updates = self.updates + alpha * flat_grad
                    


        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []

            for p in group["params"]:
                # print(p.shape)
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    
            self.storm1(
                params_with_grad,
                d_p_list,
                maximize=group["maximize"],
                lr=group["lr"],
            )

            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer
        if self.loss is not None and iter == 1:
            self.loss_prev = self.loss()

        if self.loss is not None and iter % frequency == 0:
            loss = self.loss()
            rho = (self.loss_prev - loss) / (lr * torch.linalg.norm(self.updates, 2))
            self.updates = None
            self.loss_prev = loss

            # Update lr
            if rho < 0.25:
                lr = lr / 2
            elif rho > 0.75:
                lr = lr * 2
            
            self.rho = rho
            
            for group in self.param_groups:
                group["lr"] = lr

            return closure()
