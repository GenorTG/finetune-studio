"""Models tab — list, download, configure models."""

import os
import subprocess

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

from finetune_studio.models.loader import load_model_info

router = APIRouter()


def _identify_process(args_line: str, pid: int) -> str:
    """Human-identifiable name for a GPU consumer. `ps comm` truncates
    everything to 'python3.12' — useless. Prefer a recognizable project
    marker from the full cmdline, then the script basename."""
    low = args_line.lower()
    for marker in ("comfyui", "comfy", "ollama", "vllm", "lmstudio", "lm studio",
                   "stable-diffusion", "sd-webui", "fooocus", "kohaku",
                   "text-generation-webui", "jupyter", "tensorboard"):
        if marker in low:
            return marker
    try:
        import shlex
        parts = shlex.split(args_line)
    except ValueError:
        parts = args_line.split()
    for p in parts:
        if p.endswith((".py", ".js", ".sh", ".ts", ".jar")):
            return os.path.basename(p)
    if parts:
        return os.path.basename(parts[0])
    return f"pid-{pid}"


def _gpu_snapshot():
    """Return (free_mib, top consumers) from nvidia-smi, or (None, [])."""
    try:
        free = int(subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.splitlines()[0].strip())
    except Exception:  # noqa: BLE001
        return None, []
    top = []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        for line in out.splitlines():
            try:
                pid_s, mem_s = line.split(",")
                top.append({"pid": int(pid_s.strip()), "vram_mib": int(mem_s.strip())})
            except ValueError:
                continue
    except Exception:  # noqa: BLE001
        pass
    top.sort(key=lambda t: -t["vram_mib"])
    for t in top:
        try:
            cmd = subprocess.run(
                ["ps", "-o", "args=", "-p", str(t["pid"])],
                capture_output=True, text=True, timeout=3,
            ).stdout.strip()
            t["name"] = _identify_process(cmd, t["pid"]) if cmd else f"pid-{t['pid']}"
        except Exception:  # noqa: BLE001, S110
            t["name"] = f"pid-{t['pid']}"
    return free, top[:5]


def _vram_hint(model_path: str) -> str:
    """Actionable VRAM-capacity note appended to load errors, or '' if fine."""
    free, top = _gpu_snapshot()
    if free is None:
        return ""
    try:
        need_mib = int(os.path.getsize(model_path) / (1024 * 1024) * 1.2) + 1024
    except OSError:
        return ""
    if need_mib <= free:
        return ""
    occ = ", ".join(
        f"{t['name']}(pid {t['pid']}) {t['vram_mib'] / 1024:.1f}GB"
        for t in top if t["vram_mib"] > 1024
    )
    return (
        f" — not enough VRAM: model needs ~{need_mib / 1024:.1f} GB but only "
        f"{free / 1024:.1f} GB is free"
        + (f"; top consumers: {occ}. Close them or load a smaller quant."
           if occ else ".")
    )

@router.get("/list")
async def list_models():
    from finetune_studio.webui.app import discovered_models
    return [{"name": m.name, "path": m.path, "format": m.format,
            "size_gb": m.size_gb, "architecture": m.architecture,
            "vision": getattr(m, "vision", False)} for m in discovered_models]

@router.get("/count")
async def count_models():
    from finetune_studio.webui.app import discovered_models
    return PlainTextResponse(str(len(discovered_models)))

@router.get("/info")
async def model_info(path: str):
    info = load_model_info(path)
    # Enrich with context length and parameter count if available
    if info and not info.get('context_length'):
        info['context_length'] = _guess_context_length(info)
    return info

def _guess_context_length(info: dict) -> int:
    """Guess context length from architecture or model name."""
    arch = (info.get('architecture') or '').lower()
    name = (info.get('name') or info.get('path') or '').lower()
    # Common context lengths by architecture
    if 'llama' in arch or 'llama' in name:
        return 131072 if '3' in name else 8192
    elif 'qwen' in arch or 'qwen' in name:
        return 131072 if '2' in name or '3' in name else 8192
    elif 'gemma' in arch or 'gemma' in name:
        return 131072 if '2' in name else 8192
    elif 'mistral' in arch or 'mistral' in name:
        return 32768
    elif 'phi' in arch or 'phi' in name:
        return 131072 if '3' in name or '4' in name else 4096
    return 0

@router.post("/refresh")
async def refresh_models():
    from finetune_studio.config import settings
    from finetune_studio.models.registry import scan_models
    from finetune_studio.webui.app import discovered_models
    discovered_models.clear()
    discovered_models.extend(scan_models(settings.model_dirs))
    return {"count": len(discovered_models)}


@router.post("/load")
async def load_model_endpoint(request: Request):
    """Load a model into the global inference engine.

    Accepts either {"path": "..."} (inference page) or {"model_path": "..."}
    (legacy chat_v2 form). All inference parameters have sensible defaults
    so the inference-page call Just Works.
    """
    from finetune_studio.webui.app import inference_engine
    body = await request.json()
    model_path = body.get("path") or body.get("model_path") or ""
    if not model_path:
        return {"error": "No model path provided"}
    try:
        inference_engine.load(
            model_path,
            n_ctx=body.get("n_ctx", 16384),
            n_gpu_layers=body.get("n_gpu_layers", 99),
            n_batch=body.get("n_batch", 512),
            mmap=body.get("mmap", True),
            mlock=body.get("mlock", False),
            n_threads=body.get("n_threads"),
            flash_attn=body.get("flash_attn", True),
            seed=body.get("seed"),
            rope_freq_base=body.get("rope_freq_base", 0.0),
            rope_freq_scale=body.get("rope_freq_scale", 0.0),
        )
        vision = getattr(inference_engine, "vision", False)
        return {"status": "loaded", "model": model_path, "vision": vision}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e) + _vram_hint(model_path)}


@router.post("/unload")
async def unload_model_endpoint():
    """Manually unload the currently loaded model."""
    from finetune_studio.webui.app import inference_engine
    try:
        inference_engine.unload()
        return {"status": "unloaded"}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


# ── /api/inference/* aliases (cleaner URL namespace) ─────────────
# These mirror the routes exposed under /api/chat-v2/* so that the
# inference page can use the natural /api/inference/{status,load,...}
# paths. Single source of truth stays in chat_v2.

inference_router = APIRouter()


@inference_router.get("/status")
async def inference_status():
    from finetune_studio.webui.app import inference_engine
    from finetune_studio.testing.inference import IDLE_TIMEOUT
    loaded = inference_engine.model is not None
    return {
        "loaded": loaded,
        "model": inference_engine.model_path if loaded else None,
        "vision": getattr(inference_engine, "vision", False) if loaded else False,
        "is_gguf": inference_engine.is_gguf if loaded else False,
        "idle_seconds": inference_engine.idle_seconds,
        "idle_timeout": IDLE_TIMEOUT,
        "auto_unload_remaining": (
            max(0, IDLE_TIMEOUT - inference_engine.idle_seconds)
            if loaded and IDLE_TIMEOUT > 0
            else None
        ),
    }


@inference_router.post("/load")
async def inference_load(request: Request):
    """Alias for /api/models/load — same handler."""
    return await load_model_endpoint(request)


@inference_router.post("/unload")
async def inference_unload():
    """Alias for /api/models/unload."""
    return await unload_model_endpoint()


@inference_router.post("/chat")
async def inference_chat(request: Request):
    """Generate a chat completion using the global inference engine."""
    from finetune_studio.webui.app import inference_engine
    from finetune_studio.webui.thinking import split_thinking
    from fastapi.responses import JSONResponse

    if inference_engine.model is None:
        return JSONResponse(
            {"error": "No model loaded. Click 'Load model' first."}, status_code=409
        )

    body = await request.json()
    messages = body.get("messages") or []
    # Prepend system prompt if provided
    sp = body.get("system_prompt", "").strip()
    if sp:
        messages = [{"role": "system", "content": sp}] + messages

    try:
        response = inference_engine.generate(
            messages,
            max_tokens=body.get("max_tokens", 512),
            temperature=body.get("temperature", 0.7),
            top_p=body.get("top_p", 0.9),
            top_k=body.get("top_k", 40),
            repeat_penalty=body.get("repeat_penalty", 1.1),
        )
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=500)

    # inference_engine.generate() returns a string — split thinking from response
    if isinstance(response, dict):
        resp_text = response.get("response", "")
        thinking_text = response.get("thinking", "")
    else:
        parts = split_thinking(str(response))
        resp_text = parts["response"]
        thinking_text = parts["thinking"]

    return {
        "response": resp_text,
        "thinking": thinking_text,
        "vision": getattr(inference_engine, "vision", False),
    }


@inference_router.post("/memory-estimate")
async def inference_memory_estimate(request: Request):
    """Estimate VRAM needed for a model with given loader params."""
    from finetune_studio.webui.app import inference_engine
    body = await request.json()
    model_path = body.get("model_path") or body.get("path") or ""
    if not model_path:
        return {"error": "No model_path provided"}
    try:
        est = inference_engine.estimate_memory(
            model_path,
            n_ctx=body.get("n_ctx", 16384),
            n_gpu_layers=body.get("n_gpu_layers", 99),
        )
        return est
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
