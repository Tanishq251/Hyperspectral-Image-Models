"""
Model Complexity Analysis — Parameters and FLOPs

Usage:
  python model_info.py                    # All models
  python model_info.py MambaHSI SSMamba   # Specific models
  python model_info.py --latex            # Also save LaTeX table

How FLOPs are counted:
  1. thop.profile() — standard approach, hooks on every module
  2. Manual hook fallback — counts Conv2d/Conv3d/Linear ops directly
  The two-pass strategy avoids the "attribute 'total_ops' already exists"
  conflict that thop causes when run multiple times in one process.
"""

import sys
import torch
import torch.nn as nn
from thop import profile, clever_format
from models import list_models, create_model, get_model_config, get_model_info
from utils.score_arranger import _tex


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _purge_thop(model: nn.Module):
    """Remove all stale thop hook attributes so profile() can run cleanly."""
    for m in model.modules():
        for attr in ('total_ops', 'total_params'):
            try:
                delattr(m, attr)
            except AttributeError:
                pass


def _manual_flops(model: nn.Module, dummy_input: torch.Tensor) -> int:
    """Count FLOPs via forward hooks — works even when thop fails."""
    total = 0

    def _hook(module, inp, out):
        nonlocal total
        if isinstance(module, nn.Conv1d):
            Lout = out.shape[-1]
            total += module.in_channels * module.out_channels * module.kernel_size[0] * Lout // module.groups
        elif isinstance(module, nn.Conv2d):
            H, W = out.shape[2], out.shape[3]
            kH, kW = module.kernel_size
            total += module.in_channels * module.out_channels * kH * kW * H * W // module.groups
        elif isinstance(module, nn.Conv3d):
            D, H, W = out.shape[2], out.shape[3], out.shape[4]
            kD, kH, kW = module.kernel_size
            total += module.in_channels * module.out_channels * kD * kH * kW * D * H * W // module.groups
        elif isinstance(module, nn.Linear):
            total += module.in_features * module.out_features

    hooks = [m.register_forward_hook(_hook) for m in model.modules()]
    try:
        with torch.no_grad():
            model(dummy_input)
    finally:
        for h in hooks:
            h.remove()
    return total


# ──────────────────────────────────────────────
# Per-model probe overrides
# Some models have strict input requirements
# ──────────────────────────────────────────────
_PROBE_OVERRIDES = {
    'GSCViT':      {'patch_size': 8},   # only supports 8×8
    'SpectralFormer': {},               # works with defaults
}


def get_model_complexity(model_name, num_classes=16, bands=30, patch_size=11, device='cpu'):
    """Return (params, flops) for a model.

    Returns:
        params (int | None): parameter count
        flops  (int | None): FLOPs count (None if could not compute)
        error  (str | None): error message if model creation failed
    """
    # ── Build model ──────────────────────────────────────────────────────────
    try:
        # Apply per-model probe overrides
        probe_patch = _PROBE_OVERRIDES.get(model_name, {}).get('patch_size', patch_size)
        cfg, model = create_model(
            model_name,
            return_config=True,
            num_classes=num_classes,
            bands=bands,
            patch_size=probe_patch,
        )
        effective_patch = probe_patch
    except Exception as e:
        return None, None, str(e)

    expects_4d = cfg.get('expects_4d', False)
    model = model.to(device).eval()

    # ── Dummy input ──────────────────────────────────────────────────────────
    # Feed the RAW (unwrapped) model directly, matching whatever shape it
    # natively expects, instead of wrapping it in InputShapeWrapper. Two
    # independent bugs otherwise corrupt every expects_4d model's FLOPs:
    #  1. InputShapeWrapper.__getattr__ forwards any missing-attribute lookup
    #     to the wrapped model. thop's register_buffer() does `hasattr(self,
    #     name)` before adding "total_ops" -- once the inner model has that
    #     buffer, hasattr(wrapper, ...) spuriously returns True via the
    #     delegation, so register_buffer on the wrapper itself raises
    #     KeyError("attribute already exists"). The wrapper adds zero
    #     parameters/FLOPs of its own, so profiling the raw model directly is
    #     equivalent and sidesteps this entirely.
    #  2. Some models (e.g. WaveMamba's self.ssm_proj, HSIC_FM's dynamic
    #     layers) lazily create submodules on the *first* forward call. thop
    #     installs its counting hooks via model.apply() *before* running
    #     forward, so a lazily-created layer never gets hooked and dfs_count
    #     later crashes with AttributeError. A warmup forward pass (below)
    #     materialises these layers first so both the parameter count and the
    #     hook installation see the model's true, final structure.
    if expects_4d:
        dummy = torch.randn(1, bands, effective_patch, effective_patch, device=device)
    else:
        dummy = torch.randn(1, 1, bands, effective_patch, effective_patch, device=device)

    try:
        with torch.no_grad():
            model(dummy)   # warmup: materialise any lazily-created layers
    except Exception:
        pass

    # ── Parameter count (after warmup, so lazy layers are included) ─────────
    params = sum(p.numel() for p in model.parameters())

    # ── FLOPs: try thop first, fallback to manual ────────────────────────────
    flops = None
    _purge_thop(model)
    try:
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
    except Exception:
        pass

    if flops is None:
        # thop failed (unsupported custom ops, etc.) — use manual
        _purge_thop(model)
        try:
            flops = _manual_flops(model, dummy)
        except Exception:
            flops = None   # give up — FAHM (CUDA-only), Mamba custom ops, etc.

    return params, flops, None


# ──────────────────────────────────────────────
# Table printing
# ──────────────────────────────────────────────

def print_complexity_table(model_names=None, num_classes=16, bands=30, patch_size=11, device=None):
    """Print a formatted complexity table for the given models.

    device defaults to CUDA when available. This matters beyond speed: the
    official `mamba_ssm` package's selective-scan kernel is a CUDA-only
    compiled extension with no CPU fallback at all, so every model built on
    it (MambaHSI, S2Mamba, MambaHSI_Plus, IGroupSS-Mamba, GraphMamba,
    MambaMoE, PHDMamba, SSMamba, ...) raises
    `RuntimeError: Expected u.is_cuda() to be true, but got false` on CPU
    regardless of what the FLOP-counting tool is.
    """
    if model_names is None:
        model_names = [m for m in list_models() if not m.startswith('HSSFN_')]
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*90}")
    print(f"  Model Complexity   |  Input: (1, 1, {bands}, {patch_size}, {patch_size})  |  Classes: {num_classes}  |  Device: {device}")
    print(f"{'='*90}")
    print(f"  {'Model':<25} {'Params':>10}  {'FLOPs':>12}  {'Year':>4}  {'Venue':<14}  Status")
    print(f"  {'-'*85}")

    results = []
    for name in sorted(model_names):
        info   = get_model_info(name)
        year   = str(info.get('year', '—'))
        venue  = info.get('venue', '—')

        params, flops, err = get_model_complexity(name, num_classes, bands, patch_size, device=device)

        if err:
            print(f"  {name:<25} {'—':>10}  {'—':>12}  {year:>4}  {venue:<14}  ERROR: {err[:30]}")
            continue

        p_str = f"{params/1e6:.3f} M" if params else '—'
        f_str = f"{flops/1e9:.4f} G" if flops  else 'N/A*'
        status = 'ok' if flops else 'ok (params only)'

        print(f"  {name:<25} {p_str:>10}  {f_str:>12}  {year:>4}  {venue:<14}  {status}")
        results.append({
            'model': name, 'params': params,
            'flops': flops or 0,
            'params_str': p_str, 'flops_str': f_str,
        })

    print(f"{'='*90}")
    print(f"  * FLOPs marked N/A use custom ops (Mamba SSM, graph nets) not supported by thop/hooks")
    print(f"  Total: {len(model_names)} models | Computed: {len(results)}")
    print(f"{'='*90}\n")
    return results


# ──────────────────────────────────────────────
# LaTeX export
# ──────────────────────────────────────────────

def generate_latex_table(results):
    """Return a LaTeX table string from complexity results."""
    if not results:
        return ''
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Model Complexity Comparison}',
        r'\label{tab:model_complexity}',
        r'\begin{tabular}{lrr}',
        r'\toprule',
        r'Model & Params & FLOPs \\',
        r'\midrule',
    ]
    for r in sorted(results, key=lambda x: x['params']):
        lines.append(f"{_tex(r['model'])} & {r['params_str']} & {_tex(r['flops_str'])} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == '__main__':
    args = sys.argv[1:]

    if '--help' in args or '-h' in args:
        print(__doc__)
        print('\nAvailable models:')
        for n in sorted(list_models()):
            print(f'  - {n}')
        sys.exit(0)

    save_latex = '--latex' in args
    names = [a for a in args if not a.startswith('--')] or None

    results = print_complexity_table(model_names=names)

    if save_latex and results:
        latex = generate_latex_table(results)
        with open('model_complexity.tex', 'w') as f:
            f.write(latex)
        print(f'LaTeX saved: model_complexity.tex')
        print('\n' + latex)
