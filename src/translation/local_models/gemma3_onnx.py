"""Gemma3 270M 多语言翻译模型（ONNX 版）

基于 Gemma3-270M 微调，支持韩/英/日 6 向翻译。
使用 prompt 格式：<src:ko><tgt:en>\n文本\n###\n

⭐ 与 Opus-MT 的区别：
    - 架构：CausalLM（非 Seq2Seq）
    - 输入：需构造 prompt 模板
    - 输出：需从生成文本中提取 ### 之后的内容
    - GPU：支持 DirectML / CUDA
"""

import os
import threading
from pathlib import Path
from typing import List, Tuple

from logger import get_logger
from .base import BaseTranslationModel


log = get_logger('local_models.gemma3')


# ==================== 全局设备偏好 ====================

_preferred_device = None
_device_lock = threading.Lock()


def set_preferred_device(device):
    global _preferred_device
    with _device_lock:
        if device:
            device = str(device).strip().lower()
            if device not in ('auto', 'cpu', 'cuda', 'directml'):
                device = 'auto'
            _preferred_device = device
        else:
            _preferred_device = None
    log.info(f"[Gemma3] 设备偏好：{_preferred_device or 'auto'}")


def get_preferred_device() -> str:
    with _device_lock:
        if _preferred_device is not None:
            return _preferred_device
    return (os.environ.get('LST_GEMMA3_DEVICE', 'auto') or 'auto').lower()


# ==================== provider 解析 ====================

def _resolve_provider():
    force = get_preferred_device()

    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except Exception as e:
        log.warning(f"[Gemma3] 无法读取 providers：{e}")
        return "CPUExecutionProvider", None

    log.info(f"[Gemma3] 可用 providers：{available} | 偏好：{force}")

    if force == 'cpu':
        return "CPUExecutionProvider", None

    if force == 'cuda':
        if 'CUDAExecutionProvider' in available:
            try:
                from cuda_setup import load_cuda_dlls
                load_cuda_dlls()
            except Exception as e:
                log.warning(f"[Gemma3] CUDA DLL 预加载失败：{e}")
            return "CUDAExecutionProvider", [{"device_id": 0}]
        log.warning("[Gemma3] 请求 cuda 但不可用，回退")

    if force == 'directml':
        if 'DmlExecutionProvider' in available:
            return "DmlExecutionProvider", None
        log.warning("[Gemma3] 请求 directml 但不可用，回退")

    if 'CUDAExecutionProvider' in available:
        try:
            from cuda_setup import load_cuda_dlls
            load_cuda_dlls()
        except Exception as e:
            log.warning(f"[Gemma3] CUDA DLL 预加载失败：{e}")
        return "CUDAExecutionProvider", [{"device_id": 0}]

    if 'DmlExecutionProvider' in available:
        return "DmlExecutionProvider", None

    return "CPUExecutionProvider", None


# ==================== 语言代码映射 ====================

_LANG_MAP = {
    'ko': 'ko',
    'en': 'en',
    'ja': 'ja',
}

_SUPPORTED_LANGS = [
    ('ko', 'en'), ('en', 'ko'),
    ('ja', 'en'), ('en', 'ja'),
    ('ko', 'ja'), ('ja', 'ko'),
]


class Gemma3ONNXModel(BaseTranslationModel):
    """Gemma3 270M 翻译模型（ONNX 版）"""

    def __init__(self, model_id: str, meta: dict):
        self._model_id = model_id
        self._meta = meta
        self._model = None
        self._tokenizer = None
        self._generation_config = None
        self._lock = threading.Lock()
        self._provider_name = 'CPUExecutionProvider'
        self._variant_label = ''
        self.last_error = ''

    # ---------- 元信息 ----------

    @property
    def name(self) -> str:
        return self._meta.get('name', self._model_id)

    @property
    def size_mb(self) -> float:
        return self._meta.get('size_mb', 0.0)

    @property
    def supported_languages(self) -> List[Tuple[str, str]]:
        return _SUPPORTED_LANGS

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        p = self._provider_name
        if 'CUDA' in p:
            return 'CUDA'
        if 'Dml' in p or 'DML' in p:
            return 'DirectML'
        return 'CPU'

    @property
    def variant(self) -> str:
        return self._variant_label

    @property
    def variant_tag(self) -> str:
        if self._variant_label in ('q4', 'int8', 'quantized'):
            return 'CPU'
        if self._variant_label in ('fp32', 'fp16'):
            return 'GPU'
        return '?'

    # ---------- 变体探测 ----------

    @staticmethod
    def _detect_onnx_file(onnx_dir: Path, prefer_gpu: bool):
        """
        返回 (onnx_file, variant_label) 或 (None, None)
        - GPU 优先：model.onnx > model_fp16.onnx > model_q4.onnx
        - CPU 优先：model_q4.onnx > model_quantized.onnx > model.onnx
        """
        candidates_gpu = [
            ('model.onnx', 'fp32'),
            ('model_fp16.onnx', 'fp16'),
            ('model_q4.onnx', 'q4'),
            ('model_quantized.onnx', 'quantized'),
        ]
        candidates_cpu = [
            ('model_q4.onnx', 'q4'),
            ('model_quantized.onnx', 'quantized'),
            ('model.onnx', 'fp32'),
            ('model_fp16.onnx', 'fp16'),
        ]

        candidates = candidates_gpu if prefer_gpu else candidates_cpu

        for fname, label in candidates:
            if (onnx_dir / fname).exists():
                return fname, label

        return None, None

    # ---------- 加载 ----------

    def load(self, model_path: Path) -> bool:
        with self._lock:
            if self._model is not None:
                return True

            self.last_error = ''

            try:
                from transformers import AutoTokenizer
                from optimum.onnxruntime import ORTModelForCausalLM
            except ImportError as e:
                self.last_error = f"缺少依赖：{e}"
                log.error(
                    f"[Gemma3] {self.last_error}\n"
                    f"请运行：pip install optimum[onnxruntime] "
                    f"sentencepiece"
                )
                return False

            model_path = Path(model_path)
            if not model_path.exists():
                self.last_error = f"路径不存在：{model_path}"
                log.error(f"[Gemma3] {self.last_error}")
                return False

            # tokenizer
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path))
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = (
                        self._tokenizer.eos_token)
            except Exception as e:
                self.last_error = f"tokenizer 加载失败：{e}"
                log.exception(f"[Gemma3] {self.last_error}")
                return False

            onnx_dir = model_path / 'onnx'
            pref = get_preferred_device()
            prefer_gpu = pref in ('auto', 'cuda', 'directml')

            onnx_file, variant_label = self._detect_onnx_file(
                onnx_dir, prefer_gpu)

            if onnx_file is None:
                self.last_error = f"未找到 ONNX 文件于 {onnx_dir}"
                log.error(f"[Gemma3] {self.last_error}")
                return False

            self._variant_label = variant_label

            log.info(f"[Gemma3] 设备偏好={pref}，"
                     f"变体={variant_label}，文件={onnx_file}")

            # 量化模型在 GPU 上不稳定 → 强制 CPU
            if variant_label in ('q4', 'quantized'):
                attempts = [('CPUExecutionProvider', None)]
                if prefer_gpu:
                    log.warning(
                        "[Gemma3] 量化模型在 GPU 上可能不稳定，"
                        "强制 CPU")
            else:
                provider_name, provider_options = _resolve_provider()
                attempts = [(provider_name, provider_options)]
                if provider_name != 'CPUExecutionProvider':
                    attempts.append(('CPUExecutionProvider', None))

            for pname, popts in attempts:
                ok = self._try_load_with_provider(
                    model_path, ORTModelForCausalLM,
                    pname, popts, onnx_file)
                if ok:
                    return True

            self.last_error = (
                f"所有 provider 加载失败："
                f"{[a[0] for a in attempts]}")
            log.error(f"[Gemma3] {self._model_id} {self.last_error}")
            self._model = None
            self._tokenizer = None
            return False

    def _try_load_with_provider(self, model_path, ORTModelCls,
                                 pname, popts, onnx_file) -> bool:
        try:
            log.info(f"[Gemma3] 尝试加载：{model_path} "
                     f"(provider={pname}, file={onnx_file})")

            kwargs = dict(
                file_name=onnx_file,
                provider=pname,
            )
            if popts is not None:
                kwargs['provider_options'] = popts

            model = ORTModelCls.from_pretrained(
                str(model_path), **kwargs)

            try:
                model.config.use_cache = False
            except Exception:
                pass

        except Exception as e:
            log.warning(f"[Gemma3] {pname} 加载失败：{e}")
            return False

        # warmup
        try:
            prompt = "<src:en><tgt:ko>\nhello\n###\n"
            inputs = self._tokenizer(
                prompt, return_tensors='pt')
            _ = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
                use_cache=False,
            )
            log.info(f"[Gemma3] {pname} warmup 通过")
        except Exception as e:
            log.warning(f"[Gemma3] {pname} warmup 失败：{e}")
            try:
                del model
            except Exception:
                pass
            return False

        self._model = model
        self._provider_name = pname
        log.info(f"[Gemma3] 加载成功：{self._model_id} "
                 f"@ {pname}（{self._variant_label}）")
        return True

    # ---------- 翻译 ----------

    def translate(self, text: str,
                  src_lang: str = 'auto',
                  tgt_lang: str = 'zh') -> str:
        if not self._model or not self._tokenizer:
            return '[Gemma3 未加载]'

        text = (text or '').strip()
        if not text:
            return ''

        src = _LANG_MAP.get(src_lang)
        tgt = _LANG_MAP.get(tgt_lang)

        if src is None or tgt is None:
            return (f'[Gemma3 不支持 {src_lang}→{tgt_lang}，'
                    f'仅支持 ko/en/ja]')

        if src == tgt:
            return text

        prompt = f"<src:{src}><tgt:{tgt}>\n{text}\n###\n"

        with self._lock:
            try:
                inputs = self._tokenizer(
                    prompt, return_tensors='pt')

                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    no_repeat_ngram_size=3,
                    pad_token_id=self._tokenizer.eos_token_id,
                    use_cache=False,
                )

                full = self._tokenizer.decode(
                    outputs[0], skip_special_tokens=True)

                if '###' in full:
                    translation = full.split('###')[-1]
                else:
                    translation = full.replace(prompt, '')

                translation = translation.strip()
                for token in ('<end_of_turn>', '<eos>',
                              '</s>', '<pad>', '<start_of_turn>'):
                    translation = translation.replace(token, '')

                return translation.strip() or text

            except Exception as e:
                log.error(f"[Gemma3] 翻译失败：{e}")
                return f'[Gemma3 翻译错误: {e}]'

    # ---------- 卸载 ----------

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._tokenizer = None
            self._generation_config = None
        log.info(f"[Gemma3] 已卸载：{self._model_id}")