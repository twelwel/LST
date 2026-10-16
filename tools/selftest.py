"""模块导入自检

把 src/ 下的运行时模块逐个 import 一遍，用于在改动代码结构后
快速确认没有导入错误、缺依赖或循环导入。

用法：
    python tools/selftest.py
退出码 0 = 全部通过，1 = 有模块导入失败。
"""

import importlib
import os
import sys
import traceback

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


MODULES = [
    # 基础设施
    'version', 'logger', 'config', 'config_manager',
    # 文本处理
    'layout', 'cache', 'phrase_cache', 'local_dict', 'emit_debouncer',
    # 图像 / OCR / 硬件
    'inpaint', 'ocr_engine', 'ocr_worker', 'cuda_setup', 'scroll_detector',
    # 翻译后端
    'translation',
    'translation.base', 'translation.api_config', 'translation.throttle',
    'translation.providers', 'translation.strategies', 'translation.manager',
    'translation.local_models',
    'translation.local_models.base',
    'translation.local_models.registry',
    'translation.local_models.downloader',
    'translation.local_models.cache',
    'translation.local_models.custom',
    'translation.local_models.factory',
    'translation.local_models.hf_onnx',
    'translation.local_models.gemma3_onnx',
    # 门面 / 流程
    'translator', 'pipeline', 'latency_tester',
    # 界面
    'ui', 'ui.constants', 'ui.widgets', 'ui.panels', 'ui.main_window',
    'ui.latency_dialog', 'ui.model_manager_dialog', 'ui.model_picker_dialog',
    'region_manager', 'overlay', 'hotkeys',
    'api_setup_dialog', 'about_dialog',
]


def main() -> int:
    print('=' * 62)
    print('模块导入自检')
    print('  src = %s' % _SRC)
    print('=' * 62)

    failed = []
    for name in MODULES:
        try:
            importlib.import_module(name)
            print('[OK]   %s' % name)
        except Exception as e:
            failed.append((name, e))
            print('[FAIL] %s -> %s: %s' % (name, type(e).__name__, e))

    print()
    if failed:
        print('%d / %d 个模块导入失败：' % (len(failed), len(MODULES)))
        for name, e in failed:
            print('-' * 62)
            print('  %s' % name)
            traceback.print_exception(type(e), e, e.__traceback__)
        return 1

    print('全部 %d 个模块导入成功' % len(MODULES))
    return 0


if __name__ == '__main__':
    sys.exit(main())
