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

# torch 默认把线程数开到 CPU 核数，会与 ORT 的 intra_op 线程互相抢占。
# 实测（本机 8 线程）：
#   短句 54ms / CPU 时间 500ms（9.3 核）
#   → 限到 1 线程后：43ms / 141ms（3.3 核）—— 更快，且省 72% CPU
TORCH_THREADS = max(1, int(os.environ.get('LST_TORCH_THREADS', '1') or 1))


def _limit_torch_threads():
    """限制 torch 线程数。

    本地翻译只把 torch 用于每步解码的 logits 处理（no_repeat_ngram /
    repetition_penalty / 张量搬运），单线程足够，多线程纯属内耗。
    """
    try:
        import torch
        torch.set_num_threads(TORCH_THREADS)
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass          # 已初始化过会抛异常，忽略
    except Exception as e:
        log.debug(f"[ONNX] 设置 torch 线程数失败：{e}")


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

    # ⭐ auto → CPU（实测结论，勿改回“GPU 优先”）
    # Opus-MT 是小模型 + 逐 token 自回归，一次翻译要跑十几次小 Run；
    # DirectML 每次 Run 都有固定提交/同步开销，实测慢 6~7 倍：
    #   短句 335ms → CPU 43ms；长段落 3039ms → CPU 363ms
    # 而 OCR 是卷积大图、一次 Run，DML 反而快 1.6 倍，所以 OCR 继续走 DML。
    # 显式选 directml / cuda 时仍按上面的分支执行，只有 auto 收敛到 CPU。
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


def _inference_mode():
    """生成阶段关闭 autograd，降低显存/内存占用。"""
    try:
        import torch
        return torch.inference_mode()
    except Exception:
        return nullcontext()


try:
    from config import (LOCAL_BATCH_MAX as _BATCH_MAX,
                        LOCAL_BATCH_PADDING_RATIO as _BATCH_RATIO)
except Exception:
    _BATCH_MAX, _BATCH_RATIO = 8, 1.6


def _bucket_by_length(texts, max_size=_BATCH_MAX, ratio=_BATCH_RATIO):
    """按长度分桶：长短混批会让 padding 吃掉全部收益。"""
    ordered = sorted(texts, key=len)
    bucket = []
    for t in ordered:
        if bucket:
            shortest = len(bucket[0])          # ordered 升序，首元素最短
            if len(bucket) >= max_size or len(t) > max(1, shortest) * ratio:
                yield bucket
                bucket = []
        bucket.append(t)
    if bucket:
        yield bucket

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
        _limit_torch_threads()      # 必须在建会话/生成之前限制
        with self._lock:
            if self._model is not None: return True
            self.last_error = ''
            
            try:
                from transformers import AutoTokenizer
                # 直接导入具体模块，避免 optimum 的 _LazyModule 在
                # PyInstaller 冻结环境中解析不到模块。
                from optimum.onnxruntime.modeling_seq2seq import (
                    ORTModelForSeq2SeqLM,
                )
                self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            except Exception as e:
                self.last_error = f"初始化失败：{e}"
                # 打包版常见为缺少 hiddenimport；保留完整堆栈便于定位。
                log.exception(f"[ONNX] 初始化失败：{self._model_id}（{e}）")
                return False

            pref = get_preferred_device()
            onnx_dir = model_path / 'onnx'
            has_fp32 = (onnx_dir / 'encoder_model.onnx').exists()
            has_int8 = (onnx_dir / 'encoder_model_quantized.onnx').exists()
            # 只有显式选 cuda / directml 才走 GPU；auto 一律 CPU
            # （DML 对本地翻译模型是负优化，见 _resolve_provider 的说明）。
            prefer_gpu = pref in ('cuda', 'directml')

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
                    with _inference_mode():
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
            log.exception(f"[ONNX] {pname} 失败：{e}")
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
                    # 固定 3：原 nrn=1（短文本分支）会禁止任何 token 重复，
                    # 中文译文基本无法满足，实测译文更差（"Settings"→"设置布设"）；
                    # 固定值同时保证单条与批量推理的输出一致。
                    no_repeat = 3
                    with _inference_mode():
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

    def translate_batch(self, texts, src_lang='auto', tgt_lang='zh'):
        """批量翻译：一次 generate 处理多条，摊薄固定开销。

        实测 8 条短句：逐条 56.1ms/条 → 批量 18.4ms/条（约 3.1×）。
        去重 + 按长度分桶 + DML 下整批只取一次闸门。
        """
        if not self._model:
            return ['[未加载]'] * len(texts)
        texts = list(texts or [])
        if not texts:
            return []
        if len(texts) == 1:
            return [self.translate(texts[0], src_lang, tgt_lang)]

        # 去重（同一帧常出现重复短语）
        uniq, out_map = [], {}
        for t in texts:
            s = (t or '').strip()
            if s and s not in out_map and s not in uniq:
                uniq.append(s)
        for s in uniq:
            out_map[s] = None

        for group in _bucket_by_length(uniq):
            got = self._generate_group(group)
            if got is None:
                # 批量失败 → 逐条兜底（注意此处已释放实例锁，避免重入死锁）
                for t in group:
                    out_map[t] = self.translate(t, src_lang, tgt_lang)
            else:
                out_map.update(got)

        return [out_map.get((t or '').strip(), '') if (t or '').strip() else ''
                for t in texts]

    def _generate_group(self, group):
        """一个批次跑一次 generate；失败返回 None 交给调用方逐条重试。"""
        if not group:
            return {}
        longest = max(group, key=len)
        max_new = max(24, min(192, len(longest) + 24))
        no_repeat = 3       # 与单条路径保持一致（见 translate 中的说明）

        guard = dml_guard() if is_dml_provider(self._provider_name) else nullcontext()
        with self._lock:
            with guard:
                try:
                    inputs = self._tokenizer(
                        list(group), return_tensors='pt', padding=True,
                        truncation=True, max_length=512)
                    with _inference_mode():
                        outputs = self._model.generate(
                            **inputs,
                            max_new_tokens=max_new,
                            num_beams=1,
                            no_repeat_ngram_size=no_repeat,
                            repetition_penalty=1.15,
                        )
                    decoded = self._tokenizer.batch_decode(
                        outputs, skip_special_tokens=True)
                    if len(decoded) != len(group):
                        return None
                    return {t: (s or '').strip() for t, s in zip(group, decoded)}
                except Exception as e:
                    log.warning(f"[ONNX] 批量翻译失败（{len(group)} 条），"
                                f"回退逐条：{e}")
                    return None

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