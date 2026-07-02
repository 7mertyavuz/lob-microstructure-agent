"""Faz 3 -- enjekte edilebilir cevre testleri (SimEnvironment / MempoolListener)."""
import sys, os, asyncio
from dataclasses import asdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.sim_env import SimEnvironment
from src.models import AgentOrder, FlowState
from src.ingest.mempool_listener import MempoolListener, order_to_pending_tx


def _whale_order(size_usd=500_000, ts=None):
    return AgentOrder(
        token="UniswapV2", side="BUY", size_usd=size_usd,
        actor_label="WHALE", ts=ts or datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_injected_whale_buy_moves_flow_imbalance_and_whale_net():
    env = SimEnvironment(seed=1)
    before = env.step([], "UniswapV2")
    after = env.step([_whale_order()], "UniswapV2")
    assert after.flow_imbalance > before.flow_imbalance
    assert after.whale_net_usd > before.whale_net_usd
    assert after.whale_net_usd > 0


def test_step_returns_flowstate():
    env = SimEnvironment(seed=2)
    fs = env.step([_whale_order()], "UniswapV2")
    assert isinstance(fs, FlowState)


def test_no_autonomous_noise_without_orders():
    # SimEnvironment autonomous sentetik uretim tetiklemez; emirsiz step
    # her zaman noetr/bos akis (sample_count etkisiyle imbalance=0) vermeli.
    env = SimEnvironment(seed=3)
    fs = env.step([], "UniswapV2")
    assert fs.flow_imbalance == 0.0
    assert fs.whale_net_usd == 0


def test_determinism_same_seed_same_order_sequence():
    orders = [_whale_order(size_usd=200_000), _whale_order(size_usd=50_000)]

    env1 = SimEnvironment(seed=42)
    env2 = SimEnvironment(seed=42)
    seq1 = [asdict(env1.step([o], "UniswapV2")) for o in orders]
    seq2 = [asdict(env2.step([o], "UniswapV2")) for o in orders]
    for d1, d2 in zip(seq1, seq2):
        d1.pop("ts"); d2.pop("ts")
    assert seq1 == seq2


def test_order_to_pending_tx_pure_deterministic():
    o = _whale_order()
    tx1 = order_to_pending_tx(o)
    tx2 = order_to_pending_tx(o)
    d1, d2 = asdict(tx1), asdict(tx2)
    d1.pop("ts_ns"); d2.pop("ts_ns")
    assert d1 == d2


def test_mempool_listener_autonomous_default_preserved():
    # Varsayilan davranis (driven=False) korunmus olmali.
    q = asyncio.Queue()
    listener = MempoolListener(q)
    assert listener.driven is False
    assert hasattr(listener, "inject")
    assert hasattr(listener, "_run_simulation")


def test_mempool_listener_driven_mode_idles_until_injected():
    async def _run():
        q = asyncio.Queue()
        listener = MempoolListener(q, driven=True)
        task = asyncio.ensure_future(listener.run())
        # driven modda kendiliğinden tx uretmemeli
        try:
            await asyncio.wait_for(asyncio.sleep(0.05), timeout=0.2)
        finally:
            pass
        assert q.qsize() == 0

        order = AgentOrder(token="UniswapV2", side="BUY", size_usd=100_000,
                            actor_label="WHALE",
                            ts=datetime(2026, 1, 1, tzinfo=timezone.utc))
        await listener.inject(order)
        assert q.qsize() == 1

        listener.stop()
        await asyncio.wait_for(task, timeout=1.0)

    asyncio.run(_run())


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
