import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from ssm import SelectiveSSM


def run_with_A(ssm, x, A):
    Bsz, L, D = x.shape
    B, C, dt = ssm.x_proj(x).split([ssm.d_state, ssm.d_state, 1], dim=-1)
    delta = F.softplus(dt + ssm.dt_bias)

    h = torch.zeros(Bsz, D, ssm.d_state)
    norms = []
    for t in range(L):
        dt_t = delta[:, t].unsqueeze(-1)
        h = torch.exp(dt_t * A) * h + dt_t * B[:, t].unsqueeze(1) * x[:, t].unsqueeze(-1)
        norms.append(h.flatten().norm())
    return torch.stack(norms)


def main(L=4000, d_model=16, d_state=8, seed=0):
    torch.manual_seed(seed)
    ssm = SelectiveSSM(d_model, d_state)
    x = torch.randn(1, L, d_model)

    A_neg = -torch.exp(ssm.A_log.detach())         

    A_mixed = A_neg.clone()                       
    idx = torch.randperm(A_mixed.numel())[: int(0.2 * A_mixed.numel())]
    A_mixed.view(-1)[idx] *= -1

    A_pos = -A_neg                                 

    curves = {
        "A = -exp(A_log)  (all negative)": run_with_A(ssm, x, A_neg),
        "20% of entries positive": run_with_A(ssm, x, A_mixed),
        "all entries positive": run_with_A(ssm, x, A_pos),
    }

    plt.figure(figsize=(7, 4))
    for name, c in curves.items():
        plt.plot(c.detach().numpy(), label=name, lw=1.2)
    plt.yscale("log")
    plt.xlabel("timestep")
    plt.ylabel("||h_t||")
    plt.title(f"hidden-state magnitude over {L} steps")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("stability.png", dpi=140)
    print("wrote stability.png\n")

    for name, c in curves.items():
        finite = torch.isfinite(c)
        blew_up = (~finite).nonzero()
        where = f"overflowed at t={blew_up[0].item()}" if len(blew_up) else "stayed finite"
        print(f"{name:34s} start {c[0]:.3g}  end {c[-1]:.3g}  ({where})")


if __name__ == "__main__":
    main()
