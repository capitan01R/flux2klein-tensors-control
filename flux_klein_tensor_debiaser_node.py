"""
DIT Tensor Debiaser — FLUX.2 Klein (Full 201 Tensor Control)
=============================================================
Every single tensor exposed individually. No grouping. Full control.

201 TENSORS:
  9 Global:
    img_in.weight, txt_in.weight, 
    time_in.in_layer.weight, time_in.out_layer.weight,
    double_stream_modulation_img.lin.weight, double_stream_modulation_txt.lin.weight,
    single_stream_modulation.lin.weight,
    final_layer.adaLN_modulation.1.weight, final_layer.linear.weight

  96 Double Block (8 blocks × 12 tensors):
    Per block: img_attn.qkv, img_attn.proj, img_attn.norm.key_norm, img_attn.norm.query_norm,
               img_mlp.0, img_mlp.2,
               txt_attn.qkv, txt_attn.proj, txt_attn.norm.key_norm, txt_attn.norm.query_norm,
               txt_mlp.0, txt_mlp.2

  96 Single Block (24 blocks × 4 tensors):
    Per block: linear1, linear2, norm.key_norm, norm.query_norm

LoRA-safe via ComfyUI's add_patches system.
"""

import re
import json
import os
import time
import torch
from pathlib import Path
from collections import defaultdict


# ============================================================================
# SAVE PATH CONFIG
# ============================================================================

_SAVE_CONFIG_NAME = ".tensor_debiaser_paths.json"


def _find_comfyui_root():
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "main.py").exists() or (parent / "comfy").is_dir():
            return parent
    return Path(__file__).resolve().parent.parent.parent


def _get_save_config_path():
    return (Path(__file__).resolve().parent / _SAVE_CONFIG_NAME)


def _load_save_config():
    cfg_path = _get_save_config_path()
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_save_config(config):
    cfg_path = _get_save_config_path()
    try:
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg_path, "w") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"[FLUX Klein Tensor Debiaser] WARNING: Could not save config: {e}")


def _get_default_save_dir():
    root = _find_comfyui_root()
    cfg = _load_save_config()
    if "FluxKleinTensorDebiaser" in cfg:
        p = Path(cfg["FluxKleinTensorDebiaser"])
        if p.is_absolute():
            return p
        return root / p
    return root / "models" / "diffusion_models" / "debiased"


def _resolve_save_path(filename, save_dir_override=""):
    if save_dir_override.strip():
        base = Path(save_dir_override.strip())
        if not base.is_absolute():
            base = _find_comfyui_root() / base
    else:
        base = _get_default_save_dir()
    base.mkdir(parents=True, exist_ok=True)
    if not filename.endswith(".safetensors"):
        filename = filename + ".safetensors"
    return base / filename


# ============================================================================
# COMPLETE 201 TENSOR DEFINITIONS
# ============================================================================

N_DOUBLE = 8
N_SINGLE = 24

# --- 9 Global tensors ---
GLOBAL_TENSORS = [
    "img_in.weight",
    "txt_in.weight",
    "time_in.in_layer.weight",
    "time_in.out_layer.weight",
    "double_stream_modulation_img.lin.weight",
    "double_stream_modulation_txt.lin.weight",
    "single_stream_modulation.lin.weight",
    "final_layer.adaLN_modulation.1.weight",
    "final_layer.linear.weight",
]

GLOBAL_INFO = {
    "img_in.weight": {
        "label": "img_in.weight [4096,128]",
        "tip": "Projects patchified image tokens from 128→4096 dims. Entry point for ALL image data.",
        "group": "global", "color": "blue",
    },
    "txt_in.weight": {
        "label": "txt_in.weight [4096,12288]",
        "tip": "Projects Qwen3 text embeddings from 12288→4096 dims. Entry point for ALL text.",
        "group": "global", "color": "purple",
    },
    "time_in.in_layer.weight": {
        "label": "time_in.in [4096,256]",
        "tip": "First layer of timestep embedding. Timestep→intermediate.",
        "group": "global", "color": "green",
    },
    "time_in.out_layer.weight": {
        "label": "time_in.out [4096,4096]",
        "tip": "Second layer of timestep embedding. Intermediate→final timestep embed.",
        "group": "global", "color": "green",
    },
    "double_stream_modulation_img.lin.weight": {
        "label": "db_mod_img [24576,4096]",
        "tip": "adaLN modulation for IMG stream in ALL double blocks. Massive leverage.",
        "group": "global", "color": "blue",
    },
    "double_stream_modulation_txt.lin.weight": {
        "label": "db_mod_txt [24576,4096]",
        "tip": "adaLN modulation for TXT stream in ALL double blocks. Massive leverage.",
        "group": "global", "color": "purple",
    },
    "single_stream_modulation.lin.weight": {
        "label": "sb_mod [12288,4096]",
        "tip": "adaLN modulation for ALL single blocks. Controls joint cross-modal processing.",
        "group": "global", "color": "orange",
    },
    "final_layer.adaLN_modulation.1.weight": {
        "label": "final_adaLN [8192,4096]",
        "tip": "Final layer adaLN modulation. Last conditioning before output.",
        "group": "global", "color": "green",
    },
    "final_layer.linear.weight": {
        "label": "final_linear [128,4096]",
        "tip": "Final output projection. 4096→128 for unpatchify. Last weights before image.",
        "group": "global", "color": "green",
    },
}

# --- 96 Double block tensors (8 blocks × 12 tensors) ---
DB_TENSORS = []
DB_INFO = {}

DB_TENSOR_SUFFIXES = [
    ("img_attn.qkv.weight", "img_attn.qkv", "[12288,4096]", "IMG self-attention QKV packed", "blue"),
    ("img_attn.proj.weight", "img_attn.proj", "[4096,4096]", "IMG self-attention output projection", "blue"),
    ("img_attn.norm.key_norm.scale", "img_attn.key_norm", "[128]", "IMG attention key normalization", "blue"),
    ("img_attn.norm.query_norm.scale", "img_attn.query_norm", "[128]", "IMG attention query normalization", "blue"),
    ("img_mlp.0.weight", "img_mlp.up", "[24576,4096]", "IMG MLP up-projection (SwiGLU)", "blue"),
    ("img_mlp.2.weight", "img_mlp.down", "[4096,12288]", "IMG MLP down-projection", "blue"),
    ("txt_attn.qkv.weight", "txt_attn.qkv", "[12288,4096]", "TXT self-attention QKV packed", "purple"),
    ("txt_attn.proj.weight", "txt_attn.proj", "[4096,4096]", "TXT self-attention output projection", "purple"),
    ("txt_attn.norm.key_norm.scale", "txt_attn.key_norm", "[128]", "TXT attention key normalization", "purple"),
    ("txt_attn.norm.query_norm.scale", "txt_attn.query_norm", "[128]", "TXT attention query normalization", "purple"),
    ("txt_mlp.0.weight", "txt_mlp.up", "[24576,4096]", "TXT MLP up-projection (SwiGLU)", "purple"),
    ("txt_mlp.2.weight", "txt_mlp.down", "[4096,12288]", "TXT MLP down-projection", "purple"),
]

for i in range(N_DOUBLE):
    for suffix, short, shape, desc, color in DB_TENSOR_SUFFIXES:
        key = f"double_blocks.{i}.{suffix}"
        DB_TENSORS.append(key)
        DB_INFO[key] = {
            "label": f"db{i}.{short} {shape}",
            "tip": f"Double block {i}: {desc}. SEPARATE stream — no cross-modal.",
            "group": f"db{i}", "color": color,
        }

# --- 96 Single block tensors (24 blocks × 4 tensors) ---
SB_TENSORS = []
SB_INFO = {}

SB_TENSOR_SUFFIXES = [
    ("linear1.weight", "linear1", "[36864,4096]", "QKV+MLP gate PACKED — main compute", "orange"),
    ("linear2.weight", "linear2", "[4096,16384]", "Output projection — final per-block output", "orange"),
    ("norm.key_norm.scale", "key_norm", "[128]", "Attention key normalization scale", "orange"),
    ("norm.query_norm.scale", "query_norm", "[128]", "Attention query normalization scale", "orange"),
]

for i in range(N_SINGLE):
    phase = "early" if i < 8 else ("mid" if i < 16 else "late")
    for suffix, short, shape, desc, color in SB_TENSOR_SUFFIXES:
        key = f"single_blocks.{i}.{suffix}"
        SB_TENSORS.append(key)
        SB_INFO[key] = {
            "label": f"sb{i}.{short} {shape}",
            "tip": f"Single block {i} ({phase}): {desc}. JOINT cross-modal processing.",
            "group": f"sb{i}", "color": color,
        }

# Complete ordered list
ALL_TENSORS = GLOBAL_TENSORS + DB_TENSORS + SB_TENSORS
ALL_INFO = {**GLOBAL_INFO, **DB_INFO, **SB_INFO}

# Widget-safe names (replace dots with underscores for ComfyUI)
def _key_to_widget(key):
    return key.replace(".", "_")

def _widget_to_key(widget):
    # Reverse mapping - need to handle carefully
    # We'll store a lookup dict
    return WIDGET_TO_KEY.get(widget, widget)

WIDGET_TO_KEY = {_key_to_widget(k): k for k in ALL_TENSORS}
KEY_TO_WIDGET = {k: _key_to_widget(k) for k in ALL_TENSORS}


# ============================================================================
# SAVE IMPLEMENTATION
# ============================================================================

def _build_modified_state_dict(state_dict, tensor_strengths):
    """Build a new state dict with strengths applied directly to weights."""
    modified_sd = {}
    n_modified = 0

    for key, tensor in state_dict.items():
        s = tensor_strengths.get(key, 1.0)
        if s != 1.0:
            modified_sd[key] = (tensor.float() * s).to(tensor.dtype)
            n_modified += 1
        else:
            modified_sd[key] = tensor

    return modified_sd, n_modified, len(state_dict)


def _build_diff_state_dict(state_dict, tensor_strengths):
    """Build a diff-only state dict: only tensors that changed."""
    diff_sd = {}

    for key, tensor in state_dict.items():
        s = tensor_strengths.get(key, 1.0)
        if s != 1.0:
            diff = (tensor.float() * (s - 1.0)).to(tensor.dtype)
            diff_sd[key] = diff

    return diff_sd, len(diff_sd)


def _generate_filename(tensor_strengths):
    """Auto-generate descriptive filename from settings."""
    modified = {k: v for k, v in tensor_strengths.items() if v != 1.0}
    if not modified:
        return "flux_klein_tensor_default"

    parts = ["flux_klein_tensor"]
    
    # Count by category
    sb_mod = sum(1 for k in modified if k.startswith("single_blocks"))
    db_mod = sum(1 for k in modified if k.startswith("double_blocks"))
    global_mod = len(modified) - sb_mod - db_mod
    
    if sb_mod:
        parts.append(f"sb{sb_mod}")
    if db_mod:
        parts.append(f"db{db_mod}")
    if global_mod:
        parts.append(f"g{global_mod}")

    ts = time.strftime("%m%d_%H%M")
    parts.append(ts)

    return "_".join(parts)


# ============================================================================
# PATCHING (LoRA-safe via add_patches)
# ============================================================================

def _apply_patches(model_patcher, tensor_strengths):
    """Apply strength modifications as additive patches. LoRA-safe."""

    inner = model_patcher.model
    diff = inner.diffusion_model if hasattr(inner, "diffusion_model") else inner
    pfx = "diffusion_model." if hasattr(inner, "diffusion_model") else ""
    sd = diff.state_dict()

    cloned = model_patcher.clone()
    patches = {}
    count = 0

    for key, strength in tensor_strengths.items():
        if strength == 1.0:
            continue
        if key not in sd:
            continue
        w = sd[key]
        w_cpu = w.detach().cpu() if w.device.type != "cpu" else w.detach()
        patches[pfx + key] = (w_cpu * (strength - 1.0),)
        count += 1

    if patches:
        cloned.add_patches(patches, strength_patch=1.0)

    return cloned, count


# ============================================================================
# INFO / ANALYSIS
# ============================================================================

def _compute_stats(sd, tensor_strengths):
    """Compute per-tensor statistics from state dict."""
    stats = {}
    max_norm = 0.0

    for key in ALL_TENSORS:
        if key not in sd:
            continue
        t = sd[key]
        norm = t.float().norm().item()
        s = tensor_strengths.get(key, 1.0)
        stats[key] = {
            "params": t.numel(),
            "shape": list(t.shape),
            "norm": norm,
            "strength": s,
        }
        max_norm = max(max_norm, norm)

    for key in stats:
        stats[key]["score"] = stats[key]["norm"] / max(max_norm, 1e-8) * 100.0

    return stats


def _format_info(stats, tensor_strengths, n_patched):
    """Human-readable summary."""
    total_p = sum(s["params"] for s in stats.values())
    lines = [
        "DIT Tensor Debiaser — FLUX.2 Klein (Full 201 Tensor Control)",
        "=" * 70,
        f"Model: {total_p / 1e9:.2f}B params | 201 tensors individually controllable",
        "",
    ]

    # Group modifications by category
    global_mods = []
    db_mods = defaultdict(list)
    sb_mods = defaultdict(list)
    untouched = 0

    for key in ALL_TENSORS:
        s = stats.get(key)
        if not s:
            continue
        st = s["strength"]
        if st == 1.0:
            untouched += 1
            continue
        
        info = ALL_INFO.get(key, {})
        tag = "OFF" if st == 0.0 else f"{st:.2f}"
        entry = f"  {info.get('label', key):<45} → {tag}"
        
        if key in GLOBAL_TENSORS:
            global_mods.append(entry)
        elif key.startswith("double_blocks"):
            m = re.match(r"double_blocks\.(\d+)\.", key)
            if m:
                db_mods[int(m.group(1))].append(entry)
        elif key.startswith("single_blocks"):
            m = re.match(r"single_blocks\.(\d+)\.", key)
            if m:
                sb_mods[int(m.group(1))].append(entry)

    if global_mods or db_mods or sb_mods:
        lines.append("MODIFIED:")
        
        if global_mods:
            lines.append("\nGLOBAL:")
            lines.extend(global_mods)
        
        if db_mods:
            lines.append("\nDOUBLE BLOCKS (separate streams):")
            for i in sorted(db_mods.keys()):
                lines.append(f"  DB{i}:")
                lines.extend(["  " + l for l in db_mods[i]])
        
        if sb_mods:
            lines.append("\nSINGLE BLOCKS (joint cross-modal):")
            for i in sorted(sb_mods.keys()):
                lines.append(f"  SB{i}:")
                lines.extend(["  " + l for l in sb_mods[i]])
    else:
        lines.append("(All at 1.0 — no modifications)")

    lines.append(f"\n{untouched} tensors unchanged at 1.00")
    lines.append(f"Patched {n_patched} tensors (LoRA-safe)")
    lines.append("=" * 70)
    return "\n".join(lines)


def _make_json(stats):
    return json.dumps({
        "architecture": "FLUX2_KLEIN_FULL_TENSOR",
        "tensors": {
            key: {
                "params": s["params"],
                "shape": s["shape"],
                "score": round(s["score"], 1),
                "strength": s["strength"],
            }
            for key, s in stats.items()
        }
    })


# ============================================================================
# NODE
# ============================================================================

class FluxKleinTensorDebiaser:
    """
    Full tensor-level debiaser for FLUX.2 Klein 9B.
    
    All 201 tensors exposed individually. No grouping. Maximum control.
    LoRA-safe via ComfyUI's add_patches system.
    """

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {"required": {
            "model": ("MODEL", {"tooltip": "FLUX.2 Klein model to debias"}),
            "preset": (["Custom", "Default"], {"default": "Default"}),
            "save_model": ("BOOLEAN", {
                "default": False,
                "tooltip": "Save the modified model to disk as .safetensors",
            }),
            "save_mode": (["full_model", "diff_only"], {
                "default": "full_model",
                "tooltip": "full_model: complete ~18GB safetensors. "
                           "diff_only: small file with only changed tensors.",
            }),
            "save_filename": ("STRING", {
                "default": "auto",
                "multiline": False,
                "tooltip": "'auto' generates a descriptive name.",
            }),
            "save_directory": ("STRING", {
                "default": "",
                "multiline": False,
                "tooltip": "Override save directory. Empty = use default.",
            }),
        }}

        # Add all 201 tensors as individual controls
        for key in ALL_TENSORS:
            info = ALL_INFO.get(key, {})
            widget_name = KEY_TO_WIDGET[key]
            
            inputs["required"][widget_name] = ("BOOLEAN", {
                "default": True, 
                "tooltip": info.get("tip", key),
            })
            inputs["required"][f"{widget_name}_str"] = ("FLOAT", {
                "default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05,
                "tooltip": f"Strength: {info.get('label', key)}",
            })

        return inputs

    RETURN_TYPES = ("MODEL", "STRING", "STRING")
    RETURN_NAMES = ("model", "info", "save_path")
    OUTPUT_TOOLTIPS = (
        "Model with tensor-level modifications (LoRA patches preserved)",
        "Summary of all modifications applied",
        "Path where model was saved (empty if save_model=False)",
    )
    FUNCTION = "debias"
    CATEGORY = "model_patches"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "FLUX.2 Klein full tensor-level debiaser.\n\n"
        "ALL 201 TENSORS exposed individually:\n"
        "  9 Global (img_in, txt_in, time_in, modulations, final)\n"
        "  96 Double block (8 blocks × 12 tensors each)\n"
        "  96 Single block (24 blocks × 4 tensors each)\n\n"
        "No grouping. Maximum granularity. Full control.\n"
        "Strength 1.0 = unchanged. LoRA-safe."
    )

    def debias(self, model, preset, save_model, save_mode, save_filename,
               save_directory, **kwargs):
        print("[FLUX Klein Tensor Debiaser] Starting (201 tensor control)...")

        # Build strength map from widgets
        tensor_strengths = {}
        for key in ALL_TENSORS:
            widget_name = KEY_TO_WIDGET[key]
            enabled = kwargs.get(widget_name, True)
            if enabled:
                tensor_strengths[key] = kwargs.get(f"{widget_name}_str", 1.0)
            else:
                tensor_strengths[key] = 0.0

        # Access model
        inner = model.model
        diff = inner.diffusion_model if hasattr(inner, "diffusion_model") else inner
        sd = diff.state_dict()

        # Verify tensor mapping
        found = sum(1 for k in ALL_TENSORS if k in sd)
        print(f"[FLUX Klein Tensor Debiaser] Found {found}/201 tensors in model")

        # Check if anything needs modification
        needs_mod = any(s != 1.0 for s in tensor_strengths.values())

        if needs_mod:
            model_out, n_patched = _apply_patches(model, tensor_strengths)
            print(f"[FLUX Klein Tensor Debiaser] Patched {n_patched} tensors (LoRA-safe)")
        else:
            model_out = model.clone()
            n_patched = 0
            print("[FLUX Klein Tensor Debiaser] No modifications (all at 1.0)")

        # Stats and info
        stats = _compute_stats(sd, tensor_strengths)
        info = _format_info(stats, tensor_strengths, n_patched)
        analysis_json = _make_json(stats)

        # ---- SAVE ----
        saved_path = ""
        if save_model and needs_mod:
            try:
                import safetensors.torch as sf_torch

                if save_filename.strip().lower() in ("auto", ""):
                    fname = _generate_filename(tensor_strengths)
                else:
                    fname = save_filename.strip()

                if save_mode == "diff_only" and not fname.endswith("_diff"):
                    fname = fname.replace(".safetensors", "") + "_diff"

                full_path = _resolve_save_path(fname, save_directory)

                print(f"[FLUX Klein Tensor Debiaser] Saving ({save_mode}) to: {full_path}")
                t0 = time.time()

                if save_mode == "full_model":
                    modified_sd, n_mod, n_total = _build_modified_state_dict(
                        sd, tensor_strengths
                    )
                    metadata = {
                        "debiaser": "FluxKleinTensorDebiaser",
                        "architecture": "FLUX2_KLEIN_FULL_TENSOR",
                        "save_mode": "full_model",
                        "modified_tensors": str(n_mod),
                        "total_tensors": str(n_total),
                        "strengths": json.dumps(
                            {k: v for k, v in tensor_strengths.items() if v != 1.0}
                        ),
                    }
                    sf_torch.save_file(modified_sd, str(full_path), metadata=metadata)
                    size_gb = full_path.stat().st_size / (1024**3)
                    elapsed = time.time() - t0
                    print(f"[FLUX Klein Tensor Debiaser] Saved full model: "
                          f"{size_gb:.2f} GB, {n_mod}/{n_total} modified, "
                          f"{elapsed:.1f}s")

                elif save_mode == "diff_only":
                    diff_sd, n_diff = _build_diff_state_dict(
                        sd, tensor_strengths
                    )
                    metadata = {
                        "debiaser": "FluxKleinTensorDebiaser",
                        "architecture": "FLUX2_KLEIN_FULL_TENSOR",
                        "save_mode": "diff_only",
                        "diff_tensors": str(n_diff),
                        "strengths": json.dumps(
                            {k: v for k, v in tensor_strengths.items() if v != 1.0}
                        ),
                        "usage": "Apply as additive patch: weight = base_weight + diff_weight",
                    }
                    sf_torch.save_file(diff_sd, str(full_path), metadata=metadata)
                    size_mb = full_path.stat().st_size / (1024**2)
                    elapsed = time.time() - t0
                    print(f"[FLUX Klein Tensor Debiaser] Saved diff: "
                          f"{size_mb:.1f} MB, {n_diff} tensors, {elapsed:.1f}s")

                saved_path = str(full_path)
                info += f"\n\nSaved to: {saved_path}"

                if save_directory.strip():
                    cfg = _load_save_config()
                    cfg["FluxKleinTensorDebiaser"] = save_directory.strip()
                    _save_save_config(cfg)

            except ImportError:
                print("[FLUX Klein Tensor Debiaser] ERROR: safetensors not installed.")
                info += "\n\nSAVE FAILED: safetensors package not installed"
            except Exception as e:
                print(f"[FLUX Klein Tensor Debiaser] SAVE ERROR: {e}")
                info += f"\n\nSAVE FAILED: {e}"

        elif save_model and not needs_mod:
            info += "\n\nSkipped save — no modifications to save (all at 1.0)"
            print("[FLUX Klein Tensor Debiaser] Skipped save — nothing modified")

        print("[FLUX Klein Tensor Debiaser] Done.")
        return {
            "ui": {"analysis_json": [analysis_json]},
            "result": (model_out, info, saved_path),
        }


# ============================================================================
# REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "FluxKleinTensorDebiaser": FluxKleinTensorDebiaser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxKleinTensorDebiaser": "DIT Tensor Debiaser (FLUX.2 Klein — 201 Tensors)",
}
