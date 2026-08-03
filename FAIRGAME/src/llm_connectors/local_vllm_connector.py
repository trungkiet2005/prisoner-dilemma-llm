"""
Local LLM Connector using vLLM for offline GPU inference.
Designed for Kaggle environments with no internet access.

Usage:
    The model must be pre-downloaded and available as a local path.
    On Kaggle, this means uploading the model weights as a dataset
    and pointing MODEL_PATH to /kaggle/input/<dataset-name>/<model-folder>.
"""

from FAIRGAME.src.llm_connectors.abstract_connector import AbstractConnector

# Global singleton to avoid loading the model multiple times
_GLOBAL_LLM = None
_GLOBAL_TOKENIZER = None
_GLOBAL_SAMPLING_PARAMS = None
_GLOBAL_VLLM_SAMPLING_KWARGS = None  # raw kwargs to rebuild SamplingParams per-seed
_GLOBAL_ENGINE_TYPE = None  # "vllm" or "transformers"


def _init_vllm_engine(model_path: str, max_model_len: int = 4096,
                      temperature: float = 1.0, max_tokens: int = 1024,
                      gpu_memory_utilization: float = 0.90,
                      tensor_parallel_size: int = 1,
                      enforce_eager: bool = True,
                      quantization: str = None,
                      dtype: str = None,
                      kv_cache_dtype: str = None,
                      max_num_seqs: int = None,
                      cpu_offload_gb: float = 0,
                      trust_remote_code: bool = True):
    """Initialize vLLM engine (preferred for high throughput).

    ``enforce_eager=True`` (default) disables CUDA graphs — safest across GPUs but
    slower; set False to recover throughput on big models when the GPU is known-good.

    Quantization (để chạy 70B+ trên 1 GPU 96GB):
      - Checkpoint ĐÃ quantize sẵn (AWQ/GPTQ/bnb-4bit): để ``quantization=None`` —
        vLLM tự đọc ``quantization_config`` trong config.json của checkpoint và chọn
        kernel tốt nhất (awq_marlin/gptq_marlin...). Chỉ đặt tường minh ("awq",
        "gptq"...) khi cần ép kernel khác vì kernel auto lỗi trên GPU lạ.
      - Checkpoint bf16 thường: ``quantization="fp8"`` (quantize on-the-fly, cần GPU
        Ada/Hopper/Blackwell) hoặc ``"bitsandbytes"`` (cần wheel bitsandbytes).
      - ``dtype``: AWQ/GPTQ thường muốn "auto" (→ float16); ép "bfloat16" chỉ với
        checkpoint bf16 thường.
      - ``kv_cache_dtype="fp8"``: nén KV-cache khi VRAM sát nút (đổi chút chính xác).
      - ``max_num_seqs``/``cpu_offload_gb``: van xả cuối nếu vẫn thiếu VRAM.
    """
    from vllm import LLM, SamplingParams

    global _GLOBAL_LLM, _GLOBAL_TOKENIZER, _GLOBAL_SAMPLING_PARAMS, _GLOBAL_ENGINE_TYPE
    global _GLOBAL_VLLM_SAMPLING_KWARGS

    _GLOBAL_VLLM_SAMPLING_KWARGS = dict(
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=5,  # Capture top-5 token logprobs for XAI analysis
    )
    _GLOBAL_SAMPLING_PARAMS = SamplingParams(**_GLOBAL_VLLM_SAMPLING_KWARGS)
    # Knob tuỳ chọn CHỈ truyền khi đặt tường minh — config cũ (không quantize) giữ
    # nguyên kwargs như trước, tương thích cả vLLM cũ chưa có tham số mới.
    engine_kwargs = dict(
        model=model_path,
        trust_remote_code=bool(trust_remote_code),
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=enforce_eager,  # True = no CUDA graphs (safe, slower)
    )
    if quantization:
        engine_kwargs["quantization"] = str(quantization)
    if dtype and str(dtype).lower() != "auto":
        engine_kwargs["dtype"] = str(dtype)
    if kv_cache_dtype and str(kv_cache_dtype).lower() != "auto":
        engine_kwargs["kv_cache_dtype"] = str(kv_cache_dtype)
    if max_num_seqs:
        engine_kwargs["max_num_seqs"] = int(max_num_seqs)
    if cpu_offload_gb:
        engine_kwargs["cpu_offload_gb"] = float(cpu_offload_gb)
    try:
        _GLOBAL_LLM = LLM(**engine_kwargs)
    except (TypeError, ValueError) as e:
        # vLLM đời cũ (<0.8) đòi load_format="bitsandbytes" đi kèm
        # quantization="bitsandbytes"; bản mới tự suy ra và có thể từ chối kwarg.
        if quantization == "bitsandbytes" and "load_format" not in engine_kwargs:
            print(f"[LocalVLLM] init lại với load_format=bitsandbytes ({type(e).__name__}: {e})")
            engine_kwargs["load_format"] = "bitsandbytes"
            _GLOBAL_LLM = LLM(**engine_kwargs)
        else:
            raise
    # Lấy tokenizer để áp chat template TRƯỚC khi generate. BẮT BUỘC: gemma2 (và mọi
    # model instruct nhạy template) phát EOS ngay nếu nhận prompt thô không có marker
    # <start_of_turn>/<|im_start|> -> trả về RỖNG. Tái dùng tokenizer của chính vLLM
    # (đã nạp sẵn) để khỏi tải lại; fallback sang AutoTokenizer nếu API đổi.
    try:
        _GLOBAL_TOKENIZER = _GLOBAL_LLM.get_tokenizer()
    except Exception:  # noqa: BLE001
        from transformers import AutoTokenizer
        _GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=bool(trust_remote_code))
    _GLOBAL_ENGINE_TYPE = "vllm"
    has_tmpl = bool(getattr(_GLOBAL_TOKENIZER, "chat_template", None))
    quant_note = engine_kwargs.get("quantization", "auto-detect")
    print(f"[LocalVLLM] Loaded model from {model_path} with vLLM engine "
          f"(chat_template={'yes' if has_tmpl else 'NONE'}, quantization={quant_note})")


def _init_transformers_engine(model_path: str, temperature: float = 1.0,
                               max_tokens: int = 1024, quantization: str = None,
                               trust_remote_code: bool = True):
    """Fallback: Initialize HuggingFace Transformers pipeline.

    Optimizations applied:
      - Flash-Attention 2 (fallback to SDPA if package missing)
      - eval() + cudnn benchmark for fixed-shape kernels
      - Left-padding tokenizer (cần khi muốn batch sau này)

    ``quantization`` (để chạy 70B+ trên 1 GPU): "bnb-4bit" (NF4, ~0.55 byte/param)
    hoặc "bnb-8bit" — quantize on-the-fly checkpoint bf16 thường, cần wheel
    ``bitsandbytes``. Checkpoint ĐÃ quantize sẵn (AWQ/GPTQ/bnb-4bit serialized)
    thì để None: transformers tự đọc quantization_config (cần lib tương ứng:
    autoawq/gptqmodel/bitsandbytes). Các giá trị kiểu vLLM ("awq", "fp8"...) không
    hợp lệ ở backend này -> báo lỗi sớm cho rõ.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    global _GLOBAL_LLM, _GLOBAL_TOKENIZER, _GLOBAL_SAMPLING_PARAMS, _GLOBAL_ENGINE_TYPE

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    quant_cfg = None
    if quantization:
        q = str(quantization).lower()
        if q in ("bnb-4bit", "bnb", "bitsandbytes"):
            from transformers import BitsAndBytesConfig
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif q == "bnb-8bit":
            from transformers import BitsAndBytesConfig
            quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(
                f"Backend transformers chỉ hỗ trợ quantization 'bnb-4bit'/'bnb-8bit', "
                f"nhận {quantization!r}. AWQ/GPTQ/fp8 hãy dùng backend 'vllm' "
                f"(checkpoint quantize sẵn thì để quantization=None)."
            )

    _GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=bool(trust_remote_code), padding_side="left"
    )
    if _GLOBAL_TOKENIZER.pad_token_id is None:
        _GLOBAL_TOKENIZER.pad_token_id = _GLOBAL_TOKENIZER.eos_token_id

    common_kwargs = dict(
        trust_remote_code=bool(trust_remote_code),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if quant_cfg is not None:
        common_kwargs["quantization_config"] = quant_cfg
    try:
        _GLOBAL_LLM = AutoModelForCausalLM.from_pretrained(
            model_path, attn_implementation="flash_attention_2", **common_kwargs
        )
        print("[LocalVLLM] attn=flash_attention_2")
    except (ImportError, ValueError, RuntimeError) as e:
        _GLOBAL_LLM = AutoModelForCausalLM.from_pretrained(
            model_path, attn_implementation="sdpa", **common_kwargs
        )
        print(f"[LocalVLLM] flash-attn unavailable ({type(e).__name__}); attn=sdpa")

    _GLOBAL_LLM.eval()

    _GLOBAL_SAMPLING_PARAMS = {
        "temperature": temperature,
        "max_new_tokens": max_tokens,
    }
    _GLOBAL_ENGINE_TYPE = "transformers"
    print(f"[LocalVLLM] Loaded model from {model_path} with Transformers engine "
          f"(quantization={quantization or 'none/auto'})")


def init_local_llm(model_path: str, engine: str = "vllm", force: bool = False, **kwargs):
    """
    Initialize the local LLM engine. Call this ONCE at startup.

    Args:
        model_path: Path to the local model directory.
        engine: "vllm" (recommended) or "transformers" (fallback).
        force: if True and an engine is already loaded, free it first and reload
            (needed to run several models in one process, e.g. on Kaggle).
        **kwargs: Additional arguments passed to the engine init function.
    """
    global _GLOBAL_LLM
    if _GLOBAL_LLM is not None:
        if not force:
            print("[LocalVLLM] Engine already initialized, skipping.")
            return
        free_local_llm()

    if engine == "vllm":
        _init_vllm_engine(model_path, **kwargs)
    elif engine == "transformers":
        _init_transformers_engine(model_path, **kwargs)
    else:
        raise ValueError(f"Unknown engine type: {engine}. Use 'vllm' or 'transformers'.")


def free_local_llm() -> None:
    """Release the loaded model and ACTUALLY free GPU memory (best effort).

    Lets one process load several models in sequence (load → run → free → next),
    the cleanest way to sweep multiple models on a single Kaggle GPU.

    vLLM note: nulling the global / relying on ``LLM.__del__`` does NOT reliably
    release the GPU weights + KV-cache. We must (1) tear down the parallel/dist
    state, (2) explicitly drop the engine's ``model_executor``/``engine_core`` and
    the ``LLM`` handle BEFORE gc + empty_cache, (3) best-effort destroy the process
    group and shut down Ray (TP>1). Without (2) a re-init at gpu_memory_utilization
    ~0.9 OOMs on the 2nd/3rd model.
    """
    global _GLOBAL_LLM, _GLOBAL_TOKENIZER, _GLOBAL_SAMPLING_PARAMS
    global _GLOBAL_VLLM_SAMPLING_KWARGS, _GLOBAL_ENGINE_TYPE

    engine = _GLOBAL_ENGINE_TYPE
    llm = _GLOBAL_LLM
    # Drop the module-level strong refs first (so only our local `llm` holds it).
    _GLOBAL_LLM = None
    _GLOBAL_TOKENIZER = None
    _GLOBAL_SAMPLING_PARAMS = None
    _GLOBAL_VLLM_SAMPLING_KWARGS = None
    _GLOBAL_ENGINE_TYPE = None

    if engine == "vllm" and llm is not None:
        # 1) Tear down NCCL / model-parallel / distributed state.
        try:
            from vllm.distributed.parallel_state import (
                destroy_model_parallel,
                destroy_distributed_environment,
            )
            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception as e:  # noqa: BLE001
            print(f"[LocalVLLM] vLLM parallel teardown warning: {type(e).__name__}: {e}")
        # 2) Explicitly drop the executor/engine_core that pins VRAM.
        try:
            eng = getattr(llm, "llm_engine", None) or getattr(llm, "engine", None)
            for attr in ("model_executor", "engine_core"):
                if eng is not None and hasattr(eng, attr):
                    try:
                        delattr(eng, attr)
                    except Exception:  # noqa: BLE001
                        pass
        except Exception:  # noqa: BLE001
            pass

    del llm  # drop the last strong reference

    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass

    # 3) Best-effort: destroy lingering process group + Ray (needed for TP>1).
    try:
        import contextlib
        import torch
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            with contextlib.suppress(Exception):
                torch.distributed.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass
    try:
        import ray  # type: ignore
        if ray.is_initialized():
            ray.shutdown()
    except Exception:  # noqa: BLE001
        pass

    try:
        import torch
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            print(f"[LocalVLLM] Freed local LLM. GPU free: "
                  f"{free_b / 1024**3:.1f}/{total_b / 1024**3:.1f} GiB.")
        else:
            print("[LocalVLLM] Freed local LLM (no CUDA).")
    except Exception:  # noqa: BLE001
        print("[LocalVLLM] Freed local LLM and GPU cache.")


def _generate_vllm(prompt: str) -> str:
    """Generate text using vLLM engine."""
    outputs = _GLOBAL_LLM.generate([_format_prompt_for_model(prompt)], _GLOBAL_SAMPLING_PARAMS)
    return outputs[0].outputs[0].text.strip()


def _generate_vllm_with_details(prompt: str) -> dict:
    """Generate text using vLLM with XAI details (logprobs, top alternatives).

    Returns:
        dict with keys: text, logprobs, top_alternatives, cumulative_logprob
    """
    outputs = _GLOBAL_LLM.generate([_format_prompt_for_model(prompt)], _GLOBAL_SAMPLING_PARAMS)
    output = outputs[0].outputs[0]

    result = {
        "text": output.text.strip(),
        "cumulative_logprob": output.cumulative_logprob,
        "token_logprobs": [],
        "top_alternatives": [],
    }

    if output.logprobs:
        for token_logprob_dict in output.logprobs:
            sorted_entries = sorted(
                token_logprob_dict.values(),
                key=lambda x: x.logprob,
                reverse=True
            )
            if sorted_entries:
                top = sorted_entries[0]
                result["token_logprobs"].append({
                    "token": top.decoded_token,
                    "logprob": top.logprob,
                })
                result["top_alternatives"].append([
                    {"token": e.decoded_token, "logprob": e.logprob}
                    for e in sorted_entries[:5]
                ])

    return result


def _format_prompt_for_model(prompt: str) -> str:
    """Apply chat template if model has one, else return prompt as-is.

    ``enable_thinking=False`` chỉ có ý nghĩa với vài model (Qwen3); các tokenizer
    khác (gemma2, llama3.1, qwen2.5) bỏ qua nó vô hại, nhưng nếu một template nào đó
    báo lỗi với kwarg lạ thì thử lại không kèm kwarg để vẫn áp được template.
    """
    if _GLOBAL_TOKENIZER is not None and getattr(_GLOBAL_TOKENIZER, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        try:
            return _GLOBAL_TOKENIZER.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return _GLOBAL_TOKENIZER.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
    return prompt


def _generate_transformers(prompt: str) -> str:
    """Generate text using Transformers engine (single prompt)."""
    import torch

    text = _format_prompt_for_model(prompt)

    inputs = _GLOBAL_TOKENIZER(text, return_tensors="pt").to(_GLOBAL_LLM.device)
    with torch.inference_mode():
        output_ids = _GLOBAL_LLM.generate(
            **inputs,
            do_sample=_GLOBAL_SAMPLING_PARAMS["temperature"] > 0,
            temperature=_GLOBAL_SAMPLING_PARAMS["temperature"] if _GLOBAL_SAMPLING_PARAMS["temperature"] > 0 else None,
            max_new_tokens=_GLOBAL_SAMPLING_PARAMS["max_new_tokens"],
            use_cache=True,
            pad_token_id=_GLOBAL_TOKENIZER.pad_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return _GLOBAL_TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()


def _sampling_params_with_seed(seed):
    """Build a per-request vLLM SamplingParams with its own seed.

    Rebuilt from the stored kwargs (rather than mutating/cloning the global) so it
    works across vLLM versions even when SamplingParams is immutable. Gives
    reproducible-yet-independent sampling: each (rep, agent, round) request gets its
    own deterministic seed, so reruns are bit-reproducible while distinct requests
    stay statistically independent.
    """
    from vllm import SamplingParams

    return SamplingParams(seed=int(seed), **_GLOBAL_VLLM_SAMPLING_KWARGS)


def _seeded_generate_transformers(prompt: str, seed) -> str:
    """Single-prompt transformers generation with a fixed per-request torch seed."""
    import torch

    torch.manual_seed(int(seed))
    return _generate_transformers(prompt)


def _generate_transformers_batch(prompts: list, seeds=None) -> list:
    """Generate text for many prompts.

    If ``seeds`` is given (one per prompt), generation falls back to PER-PROMPT
    (each with its own ``torch.manual_seed``), because a single batched forward
    pass shares one torch RNG and cannot honor per-request seeds. This keeps the
    transformers backend reproducible + independent + invariant to batchSize (vLLM
    is the primary, fully-batched backend; transformers is a fallback, so trading
    throughput for correct per-(rep,round,agent) seeding is the right call).

    When ``seeds`` is None, the fast left-padded batched path is used as before.
    """
    import torch

    if seeds is not None:
        return [_seeded_generate_transformers(p, s) for p, s in zip(prompts, seeds)]

    texts = [_format_prompt_for_model(p) for p in prompts]
    inputs = _GLOBAL_TOKENIZER(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(_GLOBAL_LLM.device)

    with torch.inference_mode():
        output_ids = _GLOBAL_LLM.generate(
            **inputs,
            do_sample=_GLOBAL_SAMPLING_PARAMS["temperature"] > 0,
            temperature=_GLOBAL_SAMPLING_PARAMS["temperature"] if _GLOBAL_SAMPLING_PARAMS["temperature"] > 0 else None,
            max_new_tokens=_GLOBAL_SAMPLING_PARAMS["max_new_tokens"],
            use_cache=True,
            pad_token_id=_GLOBAL_TOKENIZER.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    results = []
    for i in range(len(texts)):
        new_tokens = output_ids[i][input_len:]
        results.append(
            _GLOBAL_TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()
        )
    return results


def _generate_vllm_batch(prompts: list, seeds=None) -> list:
    """Batched generation via vLLM (single .generate() call handles batching).

    If ``seeds`` is given (one per prompt), each prompt uses its own seeded
    SamplingParams for reproducible-yet-independent sampling.
    """
    if seeds is not None:
        sampling = [_sampling_params_with_seed(s) for s in seeds]
    else:
        sampling = _GLOBAL_SAMPLING_PARAMS
    templated = [_format_prompt_for_model(p) for p in prompts]
    outputs = _GLOBAL_LLM.generate(templated, sampling)
    return [o.outputs[0].text.strip() for o in outputs]


def send_prompts_global(prompts: list, batch_size: int = 0, seeds=None) -> list:
    """Module-level batched send. Used by batch_runner without instantiating
    a connector. ``batch_size=0`` runs the whole list as one batch.

    ``seeds`` (optional): one integer seed per prompt for reproducible sampling.
    """
    if _GLOBAL_LLM is None:
        raise RuntimeError("Local LLM not initialized. Call init_local_llm() first.")
    if not prompts:
        return []
    if seeds is not None and len(seeds) != len(prompts):
        raise ValueError(
            f"seeds length {len(seeds)} != prompts length {len(prompts)}"
        )
    if batch_size and batch_size > 0:
        results = []
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            chunk_seeds = seeds[i:i + batch_size] if seeds is not None else None
            if _GLOBAL_ENGINE_TYPE == "vllm":
                results.extend(_generate_vllm_batch(chunk, chunk_seeds))
            else:
                results.extend(_generate_transformers_batch(chunk, chunk_seeds))
        return results
    if _GLOBAL_ENGINE_TYPE == "vllm":
        return _generate_vllm_batch(prompts, seeds)
    return _generate_transformers_batch(prompts, seeds)


class LocalVLLMConnector(AbstractConnector):
    """
    Chat model connector for locally hosted LLMs via vLLM or Transformers.
    Must call init_local_llm() before using this connector.
    """

    def __init__(self, provider_model: str, temperature: float = 1.0):
        """
        Args:
            provider_model: The model identifier (used for logging/tracking).
            temperature: Sampling temperature (applied at init_local_llm level).
        """
        self.provider_model = provider_model
        self.temperature = temperature

        if _GLOBAL_LLM is None:
            raise RuntimeError(
                "Local LLM not initialized. Call init_local_llm(model_path) first."
            )

    def send_prompt(self, prompt: str) -> str:
        """
        Send a prompt to the locally loaded LLM and return the response.

        Args:
            prompt: The prompt text.

        Returns:
            The generated text response.
        """
        if _GLOBAL_ENGINE_TYPE == "vllm":
            return _generate_vllm(prompt)
        elif _GLOBAL_ENGINE_TYPE == "transformers":
            return _generate_transformers(prompt)
        else:
            raise RuntimeError(f"Unknown engine type: {_GLOBAL_ENGINE_TYPE}")

    def send_prompt_with_details(self, prompt: str) -> dict:
        """
        Send a prompt and return detailed output including logprobs for XAI analysis.
        Only available with vLLM engine.

        Args:
            prompt: The prompt text.

        Returns:
            dict: Contains 'text', 'cumulative_logprob', 'token_logprobs', 'top_alternatives'.
                  Falls back to {'text': response} for transformers engine.
        """
        if _GLOBAL_ENGINE_TYPE == "vllm":
            return _generate_vllm_with_details(prompt)
        else:
            return {"text": self.send_prompt(prompt)}
