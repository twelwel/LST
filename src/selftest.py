import importlib
import traceback

modules = [
    'config',
    'ocr_engine',
    'cache',
    'layout',
    'translator',
    'overlay',
    'region_manager',
    'pipeline',
]

print("======= 模块导入自检 =======")
ok = True
for m in modules:
    try:
        importlib.import_module(m)
        print(f"[OK]   {m}")
    except Exception as e:
        ok = False
        print(f"[FAIL] {m} → {e}")
        traceback.print_exc()

if ok:
    print("\n所有模块导入成功，可以运行 main.py")
else:
    print("\n请先修复上面标红的部分")