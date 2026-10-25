import os
import time
from pathlib import Path
import torch
import numpy as np
import timm
from contextlib import nullcontext

# Speed options
torch.backends.cudnn.benchmark = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32  # switch to torch.float16 if you want fp16 timings and your GPUs support it

# -------------------------------------------------------------------------------------------------
# Your model paths
# -------------------------------------------------------------------------------------------------
MODEL_DIR = Path(os.environ.get("DMS_MODEL_DIR", Path(__file__).parent / "models"))
MODEL_PATHS = {
    "mobilenetv4": str(MODEL_DIR / "mobilenetv4.pth"),
    "convnext_tiny": str(MODEL_DIR / "convnext.pth"),
    "mobilevitv2_100": str(MODEL_DIR / "mobilevitv2_100.pth"),
    "efficientnet_v2_rw_t": str(MODEL_DIR / "efficientnet_v2_rw_t.pth"),
}

# Timm architecture aliases per family
ARCH_ALIASES = {
    "mobilenetv4": [
        "mobilenetv4_conv_small.e2400",
        "mobilenetv4_conv_small",
        "mobilenetv4_conv_medium.e600",
        "mobilenetv4_conv_medium",
    ],
    "convnext_tiny": ["convnext_tiny"],
    "mobilevitv2_100": ["mobilevitv2_100"],
    # common names seen in the wild
    "efficientnet_v2_rw_t": ["efficientnetv2_rw_t", "efficientnet_v2_rw_t", "tf_efficientnetv2_rw_t"],
}

PREF_SD_KEYS = [
    "state_dict_ema", "model_ema", "ema_state_dict",
    "state_dict", "model", "network", "weights"
]

def safe_torch_load(path):
    # Try new safer API if available, else fall back
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=DEVICE)

def _first_tensor_state_dict(d):
    if not isinstance(d, dict):
        return None
    for k in PREF_SD_KEYS:
        if k in d:
            # Module case
            if isinstance(d[k], torch.nn.Module):
                return d[k].state_dict()
            # Raw dict case
            if isinstance(d[k], dict) and all(isinstance(v, torch.Tensor) for v in d[k].values()):
                return d[k]
    # Could be a raw state dict at top level
    if all(isinstance(v, torch.Tensor) for v in d.values()) and len(d) > 0:
        return d
    return None

def _strip_prefix(sd, prefixes=("module.", "model.")):
    out = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out

def _guess_num_classes_from_sd(sd):
    # Probe some common head keys
    head_candidates = [
        "classifier.3.weight",   # mobilenet_v3 style
        "classifier.1.weight",   # efficientnet
        "fc.weight",             # resnet like
        "head.fc.weight",        # several timm models
        "head.weight",           # convnext timm
        "classifier.2.weight",   # convnext torchvision
        "classifier.weight",
    ]
    for k in head_candidates:
        w = sd.get(k)
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            return int(w.shape[0])
    # fallback: any smallish 2D tensor
    for v in sd.values():
        if isinstance(v, torch.Tensor) and v.ndim == 2 and v.shape[0] <= 8192:
            return int(v.shape[0])
    return None

def _pick_arch_key_from_path(path):
    low = (Path(path).stem + " " + Path(path).parent.name).lower()
    if "mobilenetv4" in low:
        return "mobilenetv4"
    if "convnext" in low and "tiny" in low:
        return "convnext_tiny"
    if "mobilevitv2_100" in low or "mobilevitv2" in low:
        return "mobilevitv2_100"
    if "efficientnet" in low and "v2" in low and "rw" in low and "t" in low:
        return "efficientnet_v2_rw_t"
    return None

def _try_create_timm_model(aliases, num_classes):
    last_err = None
    for name in aliases:
        try:
            m = timm.create_model(name, pretrained=False, num_classes=num_classes)
            return m
        except Exception as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    raise RuntimeError("No valid timm alias to build the model.")

def build_model_for_checkpoint(path, ckpt_dict, default_arch_key=None):
    # Prefer explicit arch keys in checkpoint
    arch = (ckpt_dict.get("arch")
            or ckpt_dict.get("model_name")
            or default_arch_key
            or _pick_arch_key_from_path(path))
    if arch is None:
        raise RuntimeError("Could not infer architecture. Add 'arch' to the checkpoint or adjust ARCH_ALIASES mapping.")

    # Map the arch key to a list of timm aliases
    aliases = ARCH_ALIASES.get(arch, None)
    if aliases is None and arch in ARCH_ALIASES:
        aliases = ARCH_ALIASES[arch]
    if aliases is None:
        # If arch looks like a direct timm name, try it as is
        aliases = [arch]

    sd = _first_tensor_state_dict(ckpt_dict)
    if sd is None:
        raise RuntimeError("No state_dict found in checkpoint. Save a full nn.Module or include a proper state dict.")
    sd = _strip_prefix(sd)

    # Find class count
    num_classes = (ckpt_dict.get("num_classes")
                   or ckpt_dict.get("n_classes")
                   or _guess_num_classes_from_sd(sd))
    if num_classes is None:
        raise RuntimeError("Could not infer num_classes. Add 'num_classes' to checkpoint or save a full nn.Module.")

    model = _try_create_timm_model(aliases, num_classes)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[{aliases[0]}] load_state_dict not strict. Missing={len(missing)} Unexpected={len(unexpected)}")
    return model

def load_model_any(path, default_arch_key=None):
    path = str(path)

    # 1) TorchScript module
    try:
        m = torch.jit.load(path, map_location=DEVICE)
        m.eval().to(DEVICE)
        return m
    except Exception:
        pass

    # 2) torch.save formats
    obj = safe_torch_load(path)

    # Full module saved directly
    if isinstance(obj, torch.nn.Module):
        return obj.to(DEVICE).eval()

    # Dict-like checkpoint
    if isinstance(obj, dict):
        model = build_model_for_checkpoint(path, obj, default_arch_key=default_arch_key)
        return model.to(DEVICE).eval()

    raise RuntimeError(f"Unsupported checkpoint type: {type(obj)}")

def time_inference(model, input_size=(1, 3, 224, 224), warmup=10, iters=100):
    x = torch.randn(*input_size, device=DEVICE, dtype=DTYPE)
    use_amp = (DEVICE.type == "cuda" and DTYPE == torch.float16)
    amp_ctx = torch.cuda.amp.autocast if use_amp else nullcontext

    # Warmup
    with torch.no_grad():
        with amp_ctx():
            for _ in range(warmup):
                y = model(x)
                if isinstance(y, (list, tuple)):
                    y = y[0]
                _ = y.sum().item()
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        with amp_ctx():
            for _ in range(iters):
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                y = model(x)
                if isinstance(y, (list, tuple)):
                    y = y[0]
                _ = y.sum().item()
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append(t1 - t0)

    med_time = float(np.median(np.array(times)))
    med_fps = 1.0 / med_time if med_time > 0 else float("inf")
    return med_time, med_fps

def main():
    print(f"Device: {DEVICE}")

    # Default arch key hints for each entry in MODEL_PATHS dict
    arch_key_hints = {
        "mobilenetv4": "mobilenetv4",
        "convnext_tiny": "convnext_tiny",
        "mobilevitv2_100": "mobilevitv2_100",
        "efficientnet_v2_rw_t": "efficientnet_v2_rw_t",
    }

    # Optional: adjust sizes if your models expect a different resolution
    input_size = (1, 3, 224, 224)
    warmup = int(os.environ.get("WARMUP", 10))
    iters = int(os.environ.get("ITERS", 100))

    results = []
    for name, path in MODEL_PATHS.items():
        print(f"\n=== {name} ===")
        if not os.path.isfile(path):
            print(f"Missing file: {path}")
            results.append((name, None, None, "missing file"))
            continue
        try:
            model = load_model_any(path, default_arch_key=arch_key_hints.get(name))
            mt, mfps = time_inference(model, input_size=input_size, warmup=warmup, iters=iters)
            print(f"Median frame time: {mt:.6f} s")
            print(f"Median FPS: {mfps:.3f}")
            results.append((name, mt, mfps, "ok"))
        except Exception as e:
            print(f"Failed: {e}")
            results.append((name, None, None, str(e)))

    print("\nSummary:")
    for name, mt, mfps, status in results:
        if status == "ok":
            print(f"{name}: median_time={mt:.6f}s  median_fps={mfps:.3f}")
        else:
            print(f"{name}: error={status}")

if __name__ == "__main__":
    main()
