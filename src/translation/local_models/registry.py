"""模型注册表（ONNX 版）

- Gemma3：支持 GPU 加速，韩/英/日 6 向互译
- Opus-MT：仅 CPU，覆盖中英日韩法德西俄
"""

MODEL_REGISTRY = {
    # ===== Gemma3 翻译模型（支持 GPU，但仅韩/英/日，不支持中文）=====
    'gemma3-translator-270m': {
        'id': 'gemma3-translator-270m',
        'name': 'Gemma3 翻译 (270M)',
        'size_mb': 802,
        'download_mb': 802,
        'fp32_mb': 1124,
        'languages': [('ko', 'en'), ('en', 'ko'),
                       ('ja', 'en'), ('en', 'ja'),
                       ('ko', 'ja'), ('ja', 'ko')],
        'hf_repo': 'sappho192/gemma3-multilingual-translator-270m',
        'description': '支持显卡加速 · ⚠️ 仅韩/英/日互译，不支持中文',
        'recommended': False,
        'loader': 'gemma3',
    },

    # ===== Opus-MT（仅 CPU）=====
    'opus-mt-ja-zh': {
        'id': 'opus-mt-ja-zh',
        'name': 'Opus-MT 日译中',
        'size_mb': 171.0,
        'download_mb': 171.0,
        'fp32_mb': 647.0,
        'languages': [('ja', 'zh')],
        'hf_repo': 'Xenova/opus-mt-ja-zh',
        'description': '轻量级，ONNX 加速（CPU）',
        'recommended': False,
        'loader': 'hf_onnx',
    },
    'opus-mt-ja-en': {
        'id': 'opus-mt-ja-en',
        'name': 'Opus-MT 日译英',
        'size_mb': 171.0,
        'download_mb': 171.0,
        'fp32_mb': 647.0,
        'languages': [('ja', 'en')],
        'hf_repo': 'Xenova/opus-mt-ja-en',
        'description': '轻量级，ONNX 加速（CPU）',
        'recommended': False,
        'loader': 'hf_onnx',
    },
    'opus-mt-en-zh': {
        'id': 'opus-mt-en-zh',
        'name': 'Opus-MT 英译中',
        'size_mb': 171.0,
        'download_mb': 171.0,
        'fp32_mb': 647.0,
        'languages': [('en', 'zh')],
        'hf_repo': 'Xenova/opus-mt-en-zh',
        'description': '轻量级，ONNX 加速（CPU）· 与默认翻译方向一致',
        'recommended': True,
        'loader': 'hf_onnx',
    },
    'opus-mt-zh-en': {
        'id': 'opus-mt-zh-en',
        'name': 'Opus-MT 中译英',
        'size_mb': 171.0,
        'download_mb': 171.0,
        'fp32_mb': 647.0,
        'languages': [('zh', 'en')],
        'hf_repo': 'Xenova/opus-mt-zh-en',
        'description': '轻量级，ONNX 加速（CPU）',
        'recommended': False,
        'loader': 'hf_onnx',
    },
    'opus-mt-ko-zh': {
        'id': 'opus-mt-ko-zh',
        'name': 'Opus-MT 韩译中',
        'size_mb': 171.0,
        'download_mb': 171.0,
        'fp32_mb': 647.0,
        'languages': [('ko', 'zh')],
        'hf_repo': 'Xenova/opus-mt-ko-zh',
        'description': '轻量级，ONNX 加速（CPU）',
        'recommended': False,
        'loader': 'hf_onnx',
    },
    'opus-mt-ko-en': {
        'id': 'opus-mt-ko-en',
        'name': 'Opus-MT 韩译英',
        'size_mb': 171.0,
        'download_mb': 171.0,
        'fp32_mb': 647.0,
        'languages': [('ko', 'en')],
        'hf_repo': 'Xenova/opus-mt-ko-en',
        'description': '轻量级，ONNX 加速（CPU）',
        'recommended': False,
        'loader': 'hf_onnx',
    },
    'opus-mt-fr-zh': {
        'id': 'opus-mt-fr-zh',
        'name': 'Opus-MT 法译中',
        'size_mb': 171.0,
        'download_mb': 171.0,
        'fp32_mb': 647.0,
        'languages': [('fr', 'zh')],
        'hf_repo': 'Xenova/opus-mt-fr-zh',
        'description': '轻量级，ONNX 加速（CPU）',
        'recommended': False,
        'loader': 'hf_onnx',
    },
    'opus-mt-de-zh': {
        'id': 'opus-mt-de-zh',
        'name': 'Opus-MT 德译中',
        'size_mb': 171.0,
        'download_mb': 171.0,
        'fp32_mb': 647.0,
        'languages': [('de', 'zh')],
        'hf_repo': 'Xenova/opus-mt-de-zh',
        'description': '轻量级，ONNX 加速（CPU）',
        'recommended': False,
        'loader': 'hf_onnx',
    },
    'opus-mt-es-zh': {
        'id': 'opus-mt-es-zh',
        'name': 'Opus-MT 西译中',
        'size_mb': 171.0,
        'download_mb': 171.0,
        'fp32_mb': 647.0,
        'languages': [('es', 'zh')],
        'hf_repo': 'Xenova/opus-mt-es-zh',
        'description': '轻量级，ONNX 加速（CPU）',
        'recommended': False,
        'loader': 'hf_onnx',
    },
    'opus-mt-ru-zh': {
        'id': 'opus-mt-ru-zh',
        'name': 'Opus-MT 俄译中',
        'size_mb': 171.0,
        'download_mb': 171.0,
        'fp32_mb': 647.0,
        'languages': [('ru', 'zh')],
        'hf_repo': 'Xenova/opus-mt-ru-zh',
        'description': '轻量级，ONNX 加速（CPU）',
        'recommended': False,
        'loader': 'hf_onnx',
    },
}


def get_registry_entry(model_id: str):
    return MODEL_REGISTRY.get(model_id)


def list_registry():
    return list(MODEL_REGISTRY.values())