import math
from contextlib import nullcontext
from typing import Iterable, Optional, Callable
import torch
from torch.optim import Optimizer

def _flatten_params(params):
    return torch.cat([p.view(-1) for p in params if p.requires_grad])

def _unflatten_to(params, vec):
    idx = 0
    for p in params:
        if not p.requires_grad:
            continue
        num = p.numel()
        p.data.copy_(vec[idx:idx+num].view_as(p))
        idx += num

def hvp_from_loss(loss, params, v):
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat = torch.cat([g.contiguous().view(-1) for g in grads])
    dot = (flat * v).sum()
    Hv = torch.autograd.grad(dot, params, retain_graph=True)
    return torch.cat([h.contiguous().view(-1) for h in Hv])

def steihaug_tr_cg(g, hvp_fn, delta, tol=1e-4, max_iter=250, M_inv: Optional[torch.Tensor]=None):
    device = g.device
    s = torch.zeros_like(g, device=device)
    r = g.clone()
    if M_inv is not None:
        z = M_inv * r
        d = -z.clone()
        rz = (r @ z).item()
        eps = tol * math.sqrt(max(rz, 1e-12))
    else:
        d = -r.clone()
        rr = (r @ r).item()
        eps = tol * math.sqrt(max(rr, 1e-12))
    boundary = False
    it = 0

    def _pred_red(s_vec):
        Hs = hvp_fn(s_vec)
        return float(- (g @ s_vec).item() - 0.5 * (s_vec @ Hs).item())

    while it < max_iter:
        it += 1
        Hd = hvp_fn(d)
        dHd = (d @ Hd).item()

        if dHd <= 0:
            s_norm = s.norm().item()
            d_norm = d.norm().item() + 1e-12
            sd = (s @ d).item()
            a = d_norm**2; b = 2.0 * sd; c = s_norm**2 - delta**2
            disc = max(0.0, b*b - 4*a*c)
            tau = (-b + math.sqrt(disc)) / (2*a)
            s = s + tau * d
            boundary = True
            return s, _pred_red(s), boundary, it

        alpha = (r @ (M_inv * r)).item() / max(dHd, 1e-12) if M_inv is not None else (r @ r).item() / max(dHd, 1e-12)
        s_next = s + alpha * d

        if s_next.norm().item() >= delta:
            s_norm = s.norm().item()
            d_norm = d.norm().item() + 1e-12
            sd = (s @ d).item()
            a = d_norm**2; b = 2.0 * sd; c = s_norm**2 - delta**2
            disc = max(0.0, b*b - 4*a*c)
            tau = (-b + math.sqrt(disc)) / (2*a)
            s = s + tau * d
            boundary = True
            return s, _pred_red(s), boundary, it

        s = s_next
        r_new = r + alpha * Hd

        if M_inv is not None:
            rz_new = (r_new @ (M_inv * r_new)).item()
            if math.sqrt(max(rz_new, 0.0)) <= eps:
                return s, _pred_red(s), boundary, it
            beta = rz_new / max((r @ (M_inv * r)).item(), 1e-16)
            d = -(M_inv * r_new) + beta * d
        else:
            rr_new = (r_new @ r_new).item()
            if math.sqrt(max(rr_new, 0.0)) <= eps:
                return s, _pred_red(s), boundary, it
            beta = rr_new / max((r @ r).item(), 1e-16)
            d = -r_new + beta * d

        r = r_new

    return s, _pred_red(s), boundary, it

class TRNewtonCG(Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter],
                 delta0: float = 1.0,
                 deltamax: float = 100.0,
                 eta: float = 0.1,
                 tol: float = 1e-6,
                 max_cg: int = 250,
                 precond: str = "diag",
                 amp: bool = False,
                 device_type: str = None):
        defaults = dict(delta0=delta0, deltamax=deltamax, eta=eta, tol=tol,
                        max_cg=max_cg, precond=precond, amp=amp, device_type=device_type)
        super().__init__(params, defaults)
        self.delta = delta0

    def step(self, closure: Callable[[], torch.Tensor]):
        assert closure is not None, "TRNewtonCG requires a closure returning the loss (no backward inside)."
        group = self.param_groups[0]
        eta = group['eta']; precond = group['precond']
        amp = group['amp']
        device_type = group['device_type'] or ('cuda' if torch.cuda.is_available() else 'cpu')
        params = [p for p in group['params'] if p.requires_grad]

        # ---- enable grad for computing loss/grad/HVP/CG ----
        with torch.enable_grad():
            ctx = torch.autocast(device_type, dtype=torch.float16) if amp else nullcontext()
            with ctx:
                loss = closure()  # scalar loss

            grads = torch.autograd.grad(loss, params, create_graph=True)
            g = torch.cat([gi.contiguous().view(-1) for gi in grads])

            # diagonal preconditioner
            M_inv = None
            if precond == "diag":
                state = self.state.setdefault('precond', {})
                beta2 = 0.99; eps = 1e-8
                if 'v' not in state:
                    state['v'] = torch.zeros_like(g)
                v = state['v']
                v.mul_(beta2).addcmul_(g, g, value=(1.0 - beta2))
                M_inv = 1.0 / (v.sqrt() + eps)
                state['v'] = v

            def hvp_fn(v):
                return hvp_from_loss(loss, params, v)

            s, pred_red, boundary, iters = steihaug_tr_cg(
                g, hvp_fn, self.delta, tol=1e-4, max_iter=group['max_cg'], M_inv=M_inv
            )

            if pred_red <= 0 or torch.isnan(torch.tensor(pred_red)):
                gnorm = g.norm().item() + 1e-12
                s = - (self.delta / gnorm) * g
                boundary = True

        # ---- try step & evaluate with no_grad ----
        theta = _flatten_params(params).detach()
        _unflatten_to(params, theta + s)

        ctx = torch.autocast(device_type, dtype=torch.float16) if amp else nullcontext()
        with torch.no_grad():
            with ctx:
                new_loss = closure().detach()

        ared = (loss.detach() - new_loss).item()
        rho = ared / (pred_red + 1e-12)

        if rho < 0.25:
            self.delta = max(1e-6, 0.25 * self.delta)
        else:
            if rho > 0.75 and boundary:
                self.delta = min(2.0 * self.delta, group['deltamax'])

        if not (rho > eta and new_loss < loss.detach()):
            _unflatten_to(params, theta)       # rollback
            self.delta = max(1e-6, 0.5 * self.delta)
            accepted = False
            out_loss = loss.detach()
        else:
            accepted = True
            out_loss = new_loss

        self.state['last'] = dict(
            loss=float(loss.detach().item()),
            new_loss=float(new_loss.item()),
            ared=float(ared),
            pred_red=float(pred_red),
            rho=float(rho),
            delta=float(self.delta),
            cg_iters=int(iters),
            accepted=bool(accepted)
        )
        return out_loss
