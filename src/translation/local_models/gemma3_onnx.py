"""Gemma3 270M 翻译模型（ONNX 版）"""

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

log = get_logger('local_models.gemma3')

_preferred_device = None
_device_lock = threading.Lock()

def set_preferred_device(device):
    global _preferred_device
    with _device_lock: _preferred_device = str(device).strip().lower() if device else None

def get_preferred_device() -> str:
    with _device_lock:
        if _preferred_device is not None: return _preferred_device
    return (os.environ.get('LST_GEMMA3_DEVICE', 'auto') or 'auto').lower()

def _resolve_provider():
    force = get_preferred_device()
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except Exception: return "CPUExecutionProvider", None

    if force == 'cpu': return "CPUExecutionProvider", None
    if force == 'cuda' and 'CUDAExecutionProvider' in available: return "CUDAExecutionProvider", [{"device_id": 0}]
    if force == 'directml' and 'DmlExecutionProvider' in available:
        return "DmlExecutionProvider", dml_provider_options()
    if 'CUDAExecutionProvider' in available: return "CUDAExecutionProvider", [{"device_id": 0}]
    if 'DmlExecutionProvider' in available:
        return "DmlExecutionProvider", dml_provider_options()
    return "CPUExecutionProvider", None

def _inference_mode():
    """生成阶段关闭 autograd，降低显存/内存占用。"""
    try:
        import torch
        return torch.inference_mode()
    except Exception:
        return nullcontext()


_LANG_MAP = {'ko': 'ko', 'en': 'en', 'ja': 'ja'}
_SUPPORTED_LANGS = [('ko', 'en'), ('en', 'ko'), ('ja', 'en'), ('en', 'ja'), ('ko', 'ja'), ('ja', 'ko')]

class Gemma3ONNXModel(BaseTranslationModel):
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
    def supported_languages(self) -> List[Tuple[str, str]]: return _SUPPORTED_LANGS
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
                # 直接导入具体模块，避免 optimum 的 _LazyModule 在
                # PyInstaller 冻结环境中解析不到模块。
                from optimum.onnxruntime.modeling_decoder import (
                    ORTModelForCausalLM,
                )
                self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                if self._tokenizer.pad_token is None: self._tokenizer.pad_token = self._tokenizer.eos_token
            except Exception as e:
                self.last_error = f"初始化失败：{e}"
                return False

            pref = get_preferred_device()
            onnx_dir = model_path / 'onnx'
            has_fp32 = (onnx_dir / 'model.onnx').exists()
            has_q4 = (onnx_dir / 'model_q4.onnx').exists()
            # 自动模式只有下载了 FP32 才走 GPU；否则使用量化 CPU 版。
            prefer_gpu = (pref in ('cuda', 'directml')
                          or (pref == 'auto' and has_fp32))

            # ⭐ 智能选择
            if prefer_gpu:
                if not has_fp32:
                    self.last_error = "当前设备偏好 GPU，但未下载 FP32 模型。请在管理面板下载 GPU 版。"
                    return False
                onnx_file, self._variant_label = 'model.onnx', 'FP32'
                pname, popts = _resolve_provider()
                attempts = [(pname, popts), ('CPUExecutionProvider', None)]
            else:
                if has_q4:
                    onnx_file, self._variant_label = 'model_q4.onnx', 'q4'
                elif has_fp32:
                    onnx_file, self._variant_label = 'model.onnx', 'FP32'
                else:
                    self.last_error = "未找到模型文件"
                    return False
                attempts = [('CPUExecutionProvider', None)]

            for pname, popts in attempts:
                if self._try_load(model_path, ORTModelForCausalLM, pname, popts, onnx_file):
                    return True
            return False

    def _try_load(self, model_path, ORTModelCls, pname, popts, onnx_file) -> bool:
        # DirectML：会话创建 + 预热必须与其他 DML 推理互斥
        guard = dml_guard() if is_dml_provider(pname) else nullcontext()
        model = None
        try:
            with guard:
                try:
                    kwargs = dict(file_name=onnx_file, provider=pname)
                    if popts: kwargs['provider_options'] = popts

                    model = ORTModelCls.from_pretrained(str(model_path), **kwargs)
                    model.config.use_cache = False

                    # Warmup
                    prompt = "<src:en><tgt:ko>\nhello\n###\n"
                    inputs = self._tokenizer(prompt, return_tensors='pt')
                    with _inference_mode():
                        _ = model.generate(**inputs, max_new_tokens=5,
                                           use_cache=False)

                    self._model = model
                    self._provider_name = pname
                except Exception:
                    model = None
                    try:
                        gc.collect()
                    except Exception:
                        pass
                    raise
            if is_dml_provider(pname):
                log.info(f"[Gemma3] 加载成功：{self._model_id} @ {pname}（{self._variant_label}）"
                         f" | {describe_dml_device()}")
            else:
                log.info(f"[Gemma3] 加载成功：{self._model_id} @ {pname}（{self._variant_label}）")
            return True
        except Exception as e:
            log.exception(f"[Gemma3] {pname} 失败：{e}")
            return False

    def translate(self, text: str, src_lang: str = 'auto', tgt_lang: str = 'zh') -> str:
        if not self._model: return '[未加载]'
        text = (text or '').strip()
        if not text: return ''
        src, tgt = _LANG_MAP.get(src_lang), _LANG_MAP.get(tgt_lang)
        if not src or not tgt: return f'[不支持 {src_lang}→{tgt_lang}]'
        if src == tgt: return text

        prompt = f"<src:{src}><tgt:{tgt}>\n{text}\n###\n"
        with self._lock:
            # DirectML：与 OCR 等其它 DML 会话串行执行，避免驱动崩溃
            guard = dml_guard() if is_dml_provider(self._provider_name) else nullcontext()
            with guard:
                try:
                    inputs = self._tokenizer(prompt, return_tensors='pt')
                    with _inference_mode():
                        outputs = self._model.generate(
                            **inputs,
                            max_new_tokens=max(24, min(256, len(text) + 24)),
                            do_sample=False,
                            pad_token_id=self._tokenizer.eos_token_id,
                            use_cache=False,
                            no_repeat_ngram_size=4,
                            repetition_penalty=1.1,
                        )
                    full = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
                    res = full.split('###')[-1] if '###' in full else full.replace(prompt, '')
                    for t in ('<end_of_turn>', '<eos>', '</s>', '<pad>', '<start_of_turn>'):
                        res = res.replace(t, '')
                    return res.strip() or text
                except Exception as e:
                    snippet = text[:80].replace('\n', ' ')
                    log.warning(f"[Gemma3] 翻译失败 @ {self._provider_name}: {e} | 文本={snippet!r}")
                    return f'[翻译错误: {e}]'

    def unload(self) -> None:
        with self._lock:
            # DirectML：释放会话同样会触碰 DML 资源，需与推理互斥
            guard = dml_guard() if is_dml_provider(self._provider_name) else nullcontext()
            with guard:
                self._model = None
                self._tokenizer = None
                try:
                    gc.collect()
                except Exception:
                    pass