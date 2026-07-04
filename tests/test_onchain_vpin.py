import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.actor.onchain_profiler import OnChainProfiler
from src.features.vpin import VPIN


# ---- VPIN ----
def test_vpin_one_sided_high():
    v = VPIN(bucket_volume_usd=100, n_buckets=10)
    for _ in range(20):
        v.add(100, is_buy=True)   # tamamen tek yönlü → VPIN ~1
    assert v.value() > 0.9


def test_vpin_balanced_low():
    v = VPIN(bucket_volume_usd=100, n_buckets=10)
    for _ in range(20):
        v.add(50, is_buy=True)
        v.add(50, is_buy=False)
    assert v.value() < 0.2


# ---- OnChainProfiler (sahte w3 enjekte) ----
class _FakeEth:
    async def get_balance(self, a): return 200 * 10**18
    async def get_transaction_count(self, a): return 1234

class _FakeW3:
    eth = _FakeEth()


def test_profiler_uses_rpc():
    p = OnChainProfiler(w3=_FakeW3())
    prof = asyncio.run(p.get_profile("0xABC"))
    assert prof.balance_eth == 200.0
    assert prof.tx_count == 1234
    assert prof.age_days >= 0


def test_profiler_etherscan_age():
    captured = {}
    def fake_http(url, timeout=10):
        captured["url"] = url
        return {"result": [{"timeStamp": "1600000000"}]}
    p = OnChainProfiler(w3=_FakeW3(), etherscan_key="KEY", http_get=fake_http)
    prof = asyncio.run(p.get_profile("0xABC"))
    assert "etherscan.io" in captured["url"]
    assert prof.age_days > 1000   # 2020'den bu yana yıllar


if __name__ == "__main__":
    test_vpin_one_sided_high(); test_vpin_balanced_low()
    test_profiler_uses_rpc(); test_profiler_etherscan_age()
    print("onchain+vpin testleri GEÇTİ")
