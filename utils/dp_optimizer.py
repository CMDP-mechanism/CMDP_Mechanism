
import numpy as np
import torch
from torch.optim import Optimizer
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.distributions.normal import Normal
from torch.optim import SGD, Adam, Adagrad, RMSprop
from algorithm.CMDP_Mechanism import *
import math


def make_optimizer_class(cls):
    class DPOptimizerClass(cls):
        def __init__(self, l2_norm_clip, noise_multiplier, minibatch_size, microbatch_size, *args, **kwargs):

            super(DPOptimizerClass, self).__init__(*args, **kwargs)

            self.l2_norm_clip = l2_norm_clip
            self.noise_multiplier = noise_multiplier
            self.microbatch_size = microbatch_size
            self.minibatch_size = minibatch_size

            for id,group in enumerate(self.param_groups):
                group['accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]

            self.renyi_epsilon = 0.0
            self.alpha = 0
            self.T_tmp = 0
            self.sigma_ = 0

            self.m_tmp = 0
            self.N_tmp = 0
            self.L_tmp = 0
            self.B_tmp = 0
            self.lambd_tmp = 0

        def para_update(self, cmdp, dimen=1):
            self.T_tmp = cmdp.T
            self.N_tmp = cmdp.N
            self.L_tmp = cmdp.L
            self.B_tmp = cmdp.B
            self.lambd_tmp = cmdp.lamda
            self.m_tmp = dimen
            self.sigma_ = cmdp.compute_var_mixed()

        def rdp_to_epsilon_dp(self, epsilon_R, epsilon_con):
            numerator = self.L_tmp * epsilon_R
            denominator = 2 * (self.lambd_tmp ** 2) * self.m_tmp * ((self.N_tmp + 1) * self.L_tmp + 2 * abs(self.B_tmp))
            epsilon = math.sqrt(numerator / denominator)

            if epsilon <= epsilon_con:
                return epsilon_con
            else:
                return epsilon


        def epsilon_R_subsampling(self, q):
            if self.alpha <= 1:
                raise ValueError("alpha must biggger than 1")

            epsilon_R_prime_val = self.alpha*q**2/(2*self.sigma_**2)

            return epsilon_R_prime_val

        def adjust_clipping_and_noise_and_lr(self, noise_multiplier, clipping_threshold, new_lr):
            """
            Adjusts the noise multiplier and clipping threshold for all parameter groups.
            """
            for group in self.param_groups:
                group['noise_multiplier'] = noise_multiplier
                group['l2_norm_clip'] = clipping_threshold
                group['lr'] = new_lr


        def zero_microbatch_grad(self):
            super(DPOptimizerClass, self).zero_grad()


        def microbatch_step(self):
            total_norm = 0.
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        total_norm += param.grad.data.norm(2).item() ** 2.

            total_norm = total_norm ** .5
            clip_coef = min(self.l2_norm_clip / (total_norm+ 1e-6), 1.)

            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        accum_grad.add_(param.grad.data.mul(clip_coef))

            return total_norm


        def zero_accum_grad(self):
            for group in self.param_groups:
                for accum_grad in group['accum_grads']:
                    if accum_grad is not None:
                        accum_grad.zero_()


        def step_dp(self, *args, **kwargs):
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'],
                                             group['accum_grads']):
                    if param.requires_grad:

                        param.grad.data = accum_grad.clone()

                        param.grad.data.add_(self.l2_norm_clip * self.noise_multiplier * torch.randn_like(param.grad.data))

                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)

            super(DPOptimizerClass, self).step(*args, **kwargs)

        def step_dp_cmdp(self, epsilon, lr_used, *args, **kwargs):
            cmdp = CMDPMechanism(epsilon=epsilon, delta2_f=self.l2_norm_clip, batch_size=self.minibatch_size)
            dimen = 0
            for group in self.param_groups:
                group['lr'] = lr_used
                for param, accum_grad in zip(group['params'],
                                             group['accum_grads']):
                    if param.requires_grad:
                        param.grad.data = accum_grad.clone()
                        noise_np = cmdp.generate_random_samples(param.grad.numel())
                        dimen += param.grad.numel()
                        noise_np = noise_np.reshape(param.grad.data.shape)
                        noise = torch.from_numpy(noise_np).to(param.grad.data.device).type_as(param.grad.data)
                        param.grad.data.add_(noise)
                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)

            self.para_update(cmdp, dimen)
            super(DPOptimizerClass, self).step(*args, **kwargs)


        def step_agd_no_update_grad(self, epsilon, closure=None):
            """
            Performs a single optimization step (parameter update).
            """
            loss = None
            if closure is not None:
                loss = closure()

            for group in self.param_groups:
                l2_norm_clip = self.l2_norm_clip
                noise_multiplier = (math.sqrt(2 * math.log(1.25 / 1e-5))) / epsilon

                for p in group['params']:
                    if p.grad is None:
                        continue

                    grad = p.grad.data
                    param_norm = grad.norm(2)
                    clip_coef = l2_norm_clip / (param_norm + 1e-6)
                    clip_coef = min(clip_coef, 1.0)
                    grad = grad * clip_coef

                    noise = torch.randn_like(grad) * l2_norm_clip * noise_multiplier
                    grad = grad + noise / self.minibatch_size
                    p.grad.data = grad

        def step_agd_update_with_new_lr(self, new_lr, closure=None):
            """Performs a single optimization step (parameter update)."""
            loss = None
            if closure is not None:
                loss = closure()

            for group in self.param_groups:
                group['lr'] = new_lr
                for p in group['params']:
                    if p.grad is None:
                        continue
                    p.data.add_(p.grad.data, alpha=-group['lr'])

            return loss

        def step_dp_agd(self, *args, **kwargs):
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'],
                                             group['accum_grads']):
                    if param.requires_grad:

                        param.grad.data = accum_grad.clone()

                        param.grad.data.add_(self.l2_norm_clip * self.noise_multiplier * torch.randn_like(param.grad.data))

                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)

    return DPOptimizerClass

DPAdam_Optimizer = make_optimizer_class(Adam)
DPAdagrad_Optimizer = make_optimizer_class(Adagrad)
DPSGD_Optimizer = make_optimizer_class(SGD)
DPRMSprop_Optimizer = make_optimizer_class(RMSprop)

def get_dp_optimizer(dataset_name,algortithm,lr,momentum,C_t,sigma,batch_size,model,ini_alpha):

    if dataset_name=='IMDB' and algortithm!='DPAGD':
        optimizer = DPAdam_Optimizer(
            l2_norm_clip=C_t,
            noise_multiplier=sigma,
            minibatch_size=batch_size,
            microbatch_size=1,
            params=model.parameters(),
            lr=lr,
        )
        optimizer.alpha = ini_alpha
    else:
        optimizer = DPSGD_Optimizer(
            l2_norm_clip=C_t,
            noise_multiplier=sigma,
            minibatch_size=batch_size,
            microbatch_size=1,
            params=model.parameters(),
            lr=lr,
            momentum=momentum,
        )
        optimizer.alpha = ini_alpha
    return optimizer