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
        )
        super(STORM1, self).__init__(params, defaults)
        self._rho = 0.0
        self._momentum_rho = .9
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
    def step(self, loss, closure):
        """Performs a single optimization step.
        Arguments:
            loss : The loss function evaluated on the current batch.
            closure : A closure that reevaluates the model
                and returns the loss.
        """
        if closure is None:
            raise ValueError("closure is None")
        if loss is None:
            raise ValueError("loss is None")
    
        norm_flat_grad = []
        for group in self.param_groups:
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            dampening = group["dampening"]
            maximize = group["maximize"]
            lr = group["lr"]
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
                            "TR1 does not support sparse gradients, please consider SparseAdam instead"
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
                norm_flat_grad.append(view)
        self._norm_d = torch.linalg.norm(torch.cat(norm_flat_grad, 0))
                    


        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            has_sparse_grad = False
            gamma_1, gamma_2 = group["gamma_1"], group["gamma_2"]
            eta_1, eta_2 = group["eta_1"], group["eta_2"]

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
        
        updated_loss = closure()    
            
        actual_reduction = loss - updated_loss
        
        # Update rho and apply momentum to mitigate the impact of stochasticity
        self._rho = self._momentum_rho * self._rho + (1 - self._momentum_rho) * actual_reduction / \
                (lr * self._norm_d)

        return loss
