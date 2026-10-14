"""AI Provider 诊断"""
import sys
sys.path.insert(0, 'src')

from translation.api_config import load_providers, load_strategy
from translation.providers import create_provider


def main():
    print("=" * 60)
    print("1. Providers")
    print("=" * 60)
    providers = load_providers()
    for p in providers:
        print(f"\n[{p.get('id')}] {p.get('name')}")
        print(f"  type={p.get('type')}  adapter={p.get('provider')}")
        print(f"  enabled={p.get('enabled')}  primary={p.get('is_primary')}")
        cfg = p.get('config', {})
        for k, v in cfg.items():
            if k in ('api_key', 'secret_key'):
                v = '***' + str(v)[-4:] if v else '(empty)'
            print(f"    {k} = {v}")

    print("\n" + "=" * 60)
    print("2. Strategy")
    print("=" * 60)
    print(load_strategy())

    print("\n" + "=" * 60)
    print("3. Test AI translate")
    print("=" * 60)
    for cfg in providers:
        try:
            p = create_provider(cfg)
        except Exception as e:
            print(f"[FAIL] {cfg.get('name')}: {e}")
            continue
        if p.provider_type != 'ai' or not p.enabled:
            continue
        print(f"\n>>> {p.name}")
        print(f"    endpoint: {p._endpoint()}")
        print(f"    model:    {p.params.get('model')}")
        r = p.translate("Hello world")
        if r.success:
            print(f"    ✅ {r.text}  ({r.elapsed_ms}ms)")
        else:
            print(f"    ❌ {r.error}")

    print("\n" + "=" * 60)
    print("4. Full strategy test")
    print("=" * 60)
    from translation.manager import get_manager
    mgr = get_manager()
    print(f"策略: {mgr.get_strategy_name()}")
    print(f"配置: {mgr.is_configured()}")
    print(f"结果: {mgr.translate('Hello world')}")


if __name__ == '__main__':
    main()