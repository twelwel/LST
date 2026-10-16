"""探测 HF 上现成的 Opus-MT CT2 仓库"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import model_info

LANGS = [
    'ja-zh', 'ja-en',
    'en-zh', 'zh-en',
    'ko-zh', 'ko-en',
    'fr-zh', 'de-zh', 'es-zh', 'ru-zh',
]

PREFIXES = [
    'entai2965',
    'santhosh',
    'JustFrederik',
    'kekmodel',
    'wordcab',
]

SUFFIXES = ['-ct2-int8', '-ct2', '-int8', '--ct2-int8', '_ct2_int8']

print("扫描 HF 上的 Opus-MT CT2 仓库...\n")

found = {}

for lang in LANGS:
    mid = f'opus-mt-{lang}'
    print(f"[{mid}]")

    hits = []
    for prefix in PREFIXES:
        for suffix in SUFFIXES:
            candidate = f'{prefix}/{mid}{suffix}'
            try:
                info = model_info(candidate)
                if info:
                    hits.append(candidate)
                    print(f"  OK {candidate}")
            except Exception:
                pass

    if not hits:
        print(f"  -- 无")

    found[mid] = hits

print(f"\n{'='*70}")
print(f"汇总：{sum(1 for v in found.values() if v)}/{len(LANGS)} 个有现成 CT2")
print()

for mid, hits in found.items():
    if hits:
        print(f"{mid}:")
        for h in hits:
            print(f"  {h}")