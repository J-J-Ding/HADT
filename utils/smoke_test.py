import torch

from models.hadt_nq import HADTNQ
from losses.quantile_loss import multi_quantile_pinball_loss


def run_smoke_test(device=None):
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[SmokeTest] device={device}")

    base_kwargs = dict(
        seq_len=96,
        patch_len=16,
        stride=8,
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_ff=128,
        dropout=0.0,
        n_points="auto",
        max_offset=1.5,
        max_horizon=720,
        horizon_chunk_size=32,
        gradient_checkpointing=True,
    )

    # Shape and monotonicity across different variable counts and horizons.
    for channels, horizon in [(7, 96), (16, 720)]:
        model = HADTNQ(
            pred_len=horizon,
            **base_kwargs,
        ).to(device)
        model.eval()
        x = torch.randn(2, 96, channels, device=device)
        with torch.no_grad():
            q, aux = model(x, return_aux=True)

        expected = (2, horizon, channels, 7)
        assert tuple(q.shape) == expected, (q.shape, expected)
        assert torch.all(q[..., 1:] > q[..., :-1]), (
            "Non-crossing constraint failed."
        )
        positions = aux["positions"]
        assert float(positions.min()) >= 0.0
        assert float(positions.max()) <= model.n_patches - 1 + 1e-6
        print(
            f"[PASS] shape={tuple(q.shape)}, "
            f"N={model.n_patches}, K={model.n_points}, "
            f"positions=[{positions.min().item():.3f},"
            f"{positions.max().item():.3f}]"
        )
        del model, x, q, aux

    # Gradient flow to the deformable offset generator.
    model = HADTNQ(
        pred_len=96,
        **base_kwargs,
    ).to(device)
    model.train()
    x = torch.randn(2, 96, 7, device=device)
    target = torch.randn(2, 96, 7, device=device)

    q = model(x)
    loss = multi_quantile_pinball_loss(q, target)
    loss.backward()

    grad = model.decoder.offset_projection.weight.grad
    assert grad is not None, "Offset generator has no gradient tensor."
    grad_norm = float(grad.detach().abs().sum().item())
    assert grad_norm > 0.0, "Offset generator gradient is zero."
    print(f"[PASS] offset gradient L1 sum={grad_norm:.6e}")

    print("[SmokeTest] All checks passed.")
