"""HuggingFace Opus-MT ONNX 模型

用 optimum + onnxruntime 加载，无需 torch

⚠️ Opus-MT 模型已禁用 GPU 加速：
   DirectML 对量化模型的 Mul 算子存在 E_INVALIDARG bug，
   而 CPU 模式性能已足够。如需 GPU，请使用 Gemma3 模型。

int8 模型强制 CPU；FP32 模型也用 CPU（性能损失有限）。
"""

import os
import threading
from pathlib import Path
from typing import List, Tuple

from logger import get_logger
from .base import BaseTranslationModel


log = get_logger('local_models.hf_onnx')


# ==================== 全局设备偏好（保留接口兼容） ====================

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


def get_preferred_device() -> str:
    with _device_lock:
        if _preferred_device is not None:
            return _preferred_device
    return 'cpu'   # Opus-MT 强制返回 cpu


def _build_session_options(use_gpu: bool):
    """SessionOptions：限制 CPU 线程数"""
    try:
        import onnxruntime as ort
    except Exception:
        return None

    try:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        return so
    except Exception as e:
        log.warning(f"[ONNX] SessionOptions 创建失败：{e}")
        return None


def _pick_onnx_files(model_path: Path):
    """选择 Opus-MT 的模型文件（优先 int8）"""
    onnx_dir = Path(model_path) / 'onnx'
    if (onnx_dir / 'encoder_model_quantized.onnx').exists():
        return (
            'encoder_model_quantized.onnx',
            'decoder_model_quantized.onnx',
            'decoder_with_past_model_quantized.onnx',
            'int8',
        )
    if (onnx_dir / 'encoder_model.onnx').exists():
        return (
            'encoder_model.onnx',
            'decoder_model.onnx',
            'decoder_with_past_model.onnx',
            'FP32',
        )
    return None


def detect_variants(model_path: Path) -> dict:
    onnx_dir = Path(model_path) / 'onnx'
    return {
        'int8': (onnx_dir / 'encoder_model_quantized.onnx').exists(),
        'fp32': (onnx_dir / 'encoder_model.onnx').exists(),
    }


class HFONNXModel(BaseTranslationModel):
    """ONNX 版 Opus-MT（仅 CPU）"""

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
    def name(self) -> str:
        return self._meta.get('name', self._model_id)

    @property
    def size_mb(self) -> float:
        return self._meta.get('size_mb', 0.0)

    @property
    def supported_languages(self) -> List[Tuple[str, str]]:
        return self._meta.get('languages', [])

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return 'CPU'

    @property
    def variant(self) -> str:
        return self._variant_label

    @property
    def variant_tag(self) -> str:
        return 'CPU'

    # ---------- 加载 ----------

    def load(self, model_path: Path) -> bool:
        with self._lock:
            if self._model is not None:
                return True

            self.last_error = ''

            try:
                from transformers import AutoTokenizer
                from optimum.onnxruntime import ORTModelForSeq2SeqLM
            except ImportError as e:
                self.last_error = f"缺少依赖：{e}"
                log.error(
                    f"[ONNX] {self.last_error}\n"
                    f"请运行：pip install optimum[onnxruntime] "
                    f" sentencepiece"
                )
                return False

            model_path = Path(model_path)
            if not model_path.exists():
                self.last_error = f"路径不存在：{model_path}"
                log.error(f"[ONNX] {self.last_error}")
                return False

            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path))
            except Exception as e:
                self.last_error = f"tokenizer 加载失败：{e}"
                log.exception(f"[ONNX] {self.last_error}")
                return False

            files = _pick_onnx_files(model_path)
            if files is None:
                self.last_error = (
                    f"未找到 ONNX 文件于 {model_path}/onnx")
                log.error(f"[ONNX] {self.last_error}")
                return False

            enc, dec, dec_past, variant_label = files
            self._variant_label = variant_label

            # ⭐ 屏蔽 GPU，强制 CPU
            log.info(f"[ONNX] Opus-MT 强制 CPU（variant={variant_label}）")
            attempts = [('CPUExecutionProvider', None)]

            for pname, popts in attempts:
                ok = self._try_load_with_provider(
                    model_path, ORTModelForSeq2SeqLM,
                    pname, popts, enc, dec, dec_past)
                if ok:
                    return True

            self.last_error = "CPU 加载失败"
            log.error(f"[ONNX] {self._model_id} {self.last_error}")
            self._model = None
            self._tokenizer = None
            return False

    def _try_load_with_provider(self, model_path, ORTModelCls,
                                 pname, popts, enc, dec, dec_past) -> bool:
        session_options = _build_session_options(False)

        try:
            log.info(f"[ONNX] 尝试加载：{model_path} "
                     f"(provider={pname}, variant={self._variant_label})")

            kwargs = dict(
                encoder_file_name=enc,
                decoder_file_name=dec,
                decoder_with_past_file_name=dec_past,
                provider=pname,
            )
            if popts is not None:
                kwargs['provider_options'] = popts
            if session_options is not None:
                kwargs['session_options'] = session_options

            model = ORTModelCls.from_pretrained(
                str(model_path), **kwargs)
        except Exception as e:
            log.warning(f"[ONNX] {pname} session 创建失败：{e}")
            return False

        try:
            inputs = self._tokenizer(
                "hello world",
                return_tensors='pt',
                padding=True, truncation=True, max_length=64,
            )
            _ = model.generate(
                **inputs, max_length=16, num_beams=1)
            log.info(f"[ONNX] {pname} warmup 通过")
        except Exception as e:
            log.warning(f"[ONNX] {pname} warmup 失败：{e}")
            try:
                del model
            except Exception:
                pass
            return False

        self._model = model
        self._provider_name = pname
        log.info(f"[ONNX] 加载成功：{self._model_id} "
                 f"@ {pname}（{self._variant_label}）")
        return True

    # ---------- 翻译 ----------

    def translate(self, text: str,
                  src_lang: str = 'auto',
                  tgt_lang: str = 'zh') -> str:
        if not self._model or not self._tokenizer:
            return '[本地模型未加载]'

        text = (text or '').strip()
        if not text:
            return ''

        with self._lock:
            try:
                inputs = self._tokenizer(
                    text, return_tensors='pt',
                    padding=True, truncation=True, max_length=512,
                )
                # ⭐ 必须用 max_new_tokens（max_length 是「总长度」），
                #    并加 no_repeat_ngram_size：否则部分输入会陷入
                #    「早上好 早上好 早上好…」的重复循环，
                #    既输出垃圾又白白耗时十几秒。
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=512,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
                result = self._tokenizer.decode(
                    outputs[0], skip_special_tokens=True)
                return result.strip()
            except Exception as e:
                log.error(f"[ONNX] 翻译失败：{e}")
                return f'[本地翻译错误: {e}]'

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._tokenizer = None
        log.info(f"[ONNX] 已卸载：{self._model_id}")