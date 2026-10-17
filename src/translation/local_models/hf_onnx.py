"""HuggingFace Opus-MT ONNX 模型"""

import gc
import os
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import List, Tuple
from logger import get_logger
from dml_gate import dml_guard, is_dml_provider
from dml_device import dml_provider_options, describe_dml_device
from .base import BaseTranslationModel

log = get_logger('local_models.hf_onnx')

_preferred_device = None
_device_lock = threading.Lock()

def set_preferred_device(device):
    global _preferred_device
    with _device_lock:
        _preferred_device = str(device).strip().lower() if device else None

def get_preferred_device() -> str:
    with _device_lock:
        if _preferred_device is not None: return _preferred_device
    return (os.environ.get('LST_LOCAL_DEVICE', 'auto') or 'auto').lower()

def _resolve_provider():
    force = get_preferred_device()
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except Exception:
        return "CPUExecutionProvider", None

    if force == 'cpu': return "CPUExecutionProvider", None
    if force == 'cuda' and 'CUDAExecutionProvider' in available:
        return "CUDAExecutionProvider", [{"device_id": 0}]
    if force == 'directml' and 'DmlExecutionProvider' in available:
        return "DmlExecutionProvider", dml_provider_options()

    # auto
    if 'CUDAExecutionProvider' in available: return "CUDAExecutionProvider", [{"device_id": 0}]
    if 'DmlExecutionProvider' in available:
        return "DmlExecutionProvider", dml_provider_options()
    return "CPUExecutionProvider", None

def _build_session_options(use_gpu: bool):
    try:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        return so
    except Exception: return None

def detect_variants(model_path: Path) -> dict:
    onnx_dir = Path(model_path) / 'onnx'
    return {
        'cpu': (onnx_dir / 'encoder_model_quantized.onnx').exists(),
        'gpu': (onnx_dir / 'encoder_model.onnx').exists(),
    }

class HFONNXModel(BaseTranslationModel):
    def __init__(self, model_id: str, meta: dict):
        self._model_id = model_id
        self._meta = meta
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._provider_name = 'CPUExecutionProvider'
        self._variant_label = ''
        self.last_error = ''

    @property
    def name(self) -> str: return self._meta.get('name', self._model_id)
    @property
    def size_mb(self) -> float: return self._meta.get('size_mb', 0.0)
    @property
    def supported_languages(self) -> List[Tuple[str, str]]: return self._meta.get('languages', [])
    @property
    def is_loaded(self) -> bool: return self._model is not None
    @property
    def device(self) -> str:
        return 'CUDA' if 'CUDA' in self._provider_name else ('DirectML' if 'Dml' in self._provider_name else 'CPU')
    @property
    def variant_tag(self) -> str: return 'GPU' if self._variant_label == 'FP32' else 'CPU'

    def load(self, model_path: Path) -> bool:
        with self._lock:
            if self._model is not None: return True
            self.last_error = ''
            
            try:
                from transformers import AutoTokenizer
                from optimum.onnxruntime import ORTModelForSeq2SeqLM
                self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            except Exception as e:
                self.last_error = f"初始化失败：{e}"
                return False

            pref = get_preferred_device()
            prefer_gpu = pref in ('auto', 'cuda', 'directml')
            
            onnx_dir = model_path / 'onnx'
            has_fp32 = (onnx_dir / 'encoder_model.onnx').exists()
            has_int8 = (onnx_dir / 'encoder_model_quantized.onnx').exists()

            # ⭐ 智能选择：GPU 必须用 FP32，CPU 优先 int8
            if prefer_gpu:
                if not has_fp32:
                    self.last_error = "当前设备偏好 GPU，但未下载 FP32 模型。请在管理面板下载 GPU 版。"
                    log.warning(f"[ONNX] {self.last_error}")
                    return False
                enc, dec, dec_past, self._variant_label = (
                    'encoder_model.onnx', 'decoder_model.onnx', 'decoder_with_past_model.onnx', 'FP32'
                )
                pname, popts = _resolve_provider()
                attempts = [(pname, popts), ('CPUExecutionProvider', None)]
            else:
                if has_int8:
                    enc, dec, dec_past, self._variant_label = (
                        'encoder_model_quantized.onnx', 'decoder_model_quantized.onnx', 'decoder_with_past_model_quantized.onnx', 'int8'
                    )
                elif has_fp32:
                    enc, dec, dec_past, self._variant_label = (
                        'encoder_model.onnx', 'decoder_model.onnx', 'decoder_with_past_model.onnx', 'FP32'
                    )
                else:
                    self.last_error = "未找到模型文件"
                    return False
                attempts = [('CPUExecutionProvider', None)]

            for pname, popts in attempts:
                if self._try_load(model_path, ORTModelForSeq2SeqLM, pname, popts, enc, dec, dec_past):
                    return True
            return False

    def _try_load(self, model_path, ORTModelCls, pname, popts, enc, dec, dec_past) -> bool:
        # DirectML：会话创建 + 预热必须与其他 DML 推理互斥
        guard = dml_guard() if is_dml_provider(pname) else nullcontext()
        model = None
        try:
            with guard:
                try:
                    kwargs = dict(
                        encoder_file_name=enc, decoder_file_name=dec, decoder_with_past_file_name=dec_past,
                        provider=pname,
                    )
                    if popts: kwargs['provider_options'] = popts
                    so = _build_session_options(pname != 'CPUExecutionProvider')
                    if so: kwargs['session_options'] = so

                    model = ORTModelCls.from_pretrained(str(model_path), **kwargs)

                    # Warmup 探测（DirectML 设备真正跑一次）
                    inputs = self._tokenizer("hello", return_tensors='pt', max_length=64)
                    _ = model.generate(**inputs, max_length=10)

                    self._model = model
                    self._provider_name = pname
                except Exception:
                    # 半初始化对象若在锁外析构，会与其它 DML 会话并发释放 -> 原生崩溃
                    model = None
                    try:
                        gc.collect()
                    except Exception:
                        pass
                    raise
            if is_dml_provider(pname):
                log.info(f"[ONNX] 加载成功：{self._model_id} @ {pname}（{self._variant_label}）"
                         f" | {describe_dml_device()}")
            else:
                log.info(f"[ONNX] 加载成功：{self._model_id} @ {pname}（{self._variant_label}）")
            return True
        except Exception as e:
            log.warning(f"[ONNX] {pname} 失败：{e}")
            return False

    def translate(self, text: str, src_lang: str = 'auto', tgt_lang: str = 'zh') -> str:
        if not self._model: return '[未加载]'
        text = (text or '').strip()
        if not text: return ''
        with self._lock:
            # DirectML：与 OCR 等其它 DML 会话串行执行，避免驱动崩溃
            guard = dml_guard() if is_dml_provider(self._provider_name) else nullcontext()
            with guard:
                try:
                    inputs = self._tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                    # 按输入长度限制输出，防止短词（如 Settings）重复生成到 512 token
                    max_new = max(24, min(192, len(text) + 24))
                    # 短标签严格禁止 token 重复，避免 "Settings" -> "设置设置设置"
                    if len(text) <= 16:
                        no_repeat = 1
                    elif len(text) <= 40:
                        no_repeat = 2
                    else:
                        no_repeat = 3
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=max_new,
                        num_beams=1,
                        no_repeat_ngram_size=no_repeat,
                        repetition_penalty=1.15,
                    )
                    return self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                except Exception as e:
                    snippet = text[:80].replace('\n', ' ')
                    log.warning(f"[ONNX] 翻译失败 @ {self._provider_name}: {e} | 文本={snippet!r}")
                    return f'[翻译错误: {e}]'

    def unload(self) -> None:
        with self._lock:
            # DirectML：释放会话同样会触碰 DML 资源，需与推理互斥
            guard = dml_guard() if is_dml_provider(self._provider_name) else nullcontext()
            with guard:
                self._model = None
                self._tokenizer = None
                # 有些 ORT/optimum 对象存在引用环，不手动 GC 会延迟到下一次
                # DML Run 期间才析构，进而触发驱动层 0xC0000005。
                try:
                    gc.collect()
                except Exception:
                    pass