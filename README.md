# lob-microstructure-agent

Real-time **DEX market-microstructure analyzer**: it watches the Ethereum
mempool, decodes pending swaps (Uniswap V2 / V3 / V4 + Universal Router),
labels the actor behind each trade (**Whale / MEV bot / Retail**), builds
order-flow features, and produces a **short-term directional probability** —
streamed to a live dashboard.

> ⚠️ **Research / PoC, not financial advice.** Short-horizon prediction is
> noisy; outputs are *probabilities*, never certainties. The public mempool
> shows a shrinking minority of flow (~80% of Ethereum order flow is now
> private / OFA), so visibility is partial by design. Actor labels are
> probabilistic inference, not identity.

---

## Architecture

```
Mempool (WSS)            Layer 1 — Ingest          src/ingest/mempool_listener.py
   │  pending tx             normalize → asyncio.Queue   (+ simulation mode)
   ▼
Decode                   Layer 2 — Decoding        src/decode/tx_decoder.py
   │  V2 / V3 / V4 / Universal Router → side via WETH
   ▼
Actor Layer              Layer 3 — Identity        src/actor/*
   │  wallet age/volume/balance + gas + coinbase.transfer → Whale/MEV/Retail
   ├───────────────► Pipeline (memory/redis/kafka)  src/pipeline/bus.py
   ▼
Features                 Layer 4 — Microstructure  src/features/*
   │  OFI imbalance · VPIN toxicity · multi-horizon · CEX-DEX lead-lag · fracdiff
   ▼
Prediction               Layer 5 — Alpha           src/predict/*
   │  regime router (VPIN/HMM) → linear | MLP · meta-labeling (sizing)
   ▼
Dashboard + Online learning   dashboard.html · src/train/online.py (drift-aware)
```

## What's inside

**Decoding** — Uniswap V2 (`swapExact*`), V3 (`exactInputSingle` / `exactInput`,
both SwapRouter & SwapRouter02), and **Universal Router** `execute()` command
parsing (V2/V3 swap commands + V4_SWAP recognition). Buy/Sell inferred from WETH
position.

**Actor / MEV detection**
- Rule-based, transparent Whale / MEV / Retail scorer with reasons.
- **`coinbase.transfer()` detection** → near-certain searcher signature (builder bribe).
- **Sandwich** detection (front/victim/back legs, same attacker).
- **JIT liquidity** detection (Mint → victim swap → Burn, same provider/pool).
- **Atomic arbitrage** detection (single-tx token-transfer graph, closed loop).
- **zeromev** client for real MEV ground-truth labels.

**Features** — actor-weighted order-flow imbalance, **VPIN** flow toxicity,
multi-horizon (fast/slow) windows, **CEX-DEX lead-lag spread** (with lag
estimation), **fractional differencing** (stationarity + memory for deep models).

**Prediction / models**
- Calibrated logistic direction model (coefficients learned via `train.py`).
- **Regime router**: VPIN threshold or 2-state Gaussian HMM → routes between the
  linear model (normal regime) and a **NumPy MLP** (toxic/high-vol regime).
- **Meta-labeling** (López de Prado): primary picks direction, meta-model sizes
  the position (0..1).
- **Economic / cost-aware labeling** + latency-arb feasibility filter
  (profitable after gas + fees?).

**Infra** — pipeline bus (memory/Redis/Kafka), **online learning** (always-on
SGD with exponential forgetting) and a **feature-decay monitor** for concept drift.

Everything is **pure Python + NumPy** (no sklearn/torch) and runs in a
**simulation mode** with zero external dependencies.

## Project structure

```
src/
  ingest/    mempool_listener.py     # Layer 1 (+ inject()/driven mode, Faz 3)
  decode/    tx_decoder.py           # Layer 2 (V2/V3/V4/UR)
  actor/     classifier.py · wallet_profiler.py · onchain_profiler.py · agent_profiles.py
  mev/       sandwich.py · jit_liquidity.py · arbitrage.py · builder_tip.py · zeromev_client.py · decision.py
  features/  window.py (+ actor_mix) · vpin.py · lead_lag.py · fracdiff.py
  predict/   direction.py · regime.py · mlp.py · meta.py · economic.py
  train/     dataset.py · logreg.py · backtest.py · online.py
  pipeline/  bus.py
  api/       flow_feed.py (FlowFeed) · sim_env.py (SimEnvironment)   # CAS köprüsü
main.py · dashboard.py · dashboard.html · train.py · config.py
docs/        00-ORTAK-SOZLESME.md   # FlowState/FlowFeed sözleşmesi
tests/       15 test files
```

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env          # leave WSS_URL empty → SIMULATION mode (no node needed)

python main.py                # run the pipeline (prints labeled signals + predictions)
python dashboard.py           # + WebSocket server; then open dashboard.html in a browser
python train.py               # calibrate the direction model → models/direction_coeffs.json
```

Simulation mode generates synthetic whale/MEV/retail traffic so the whole
stack runs end-to-end without an RPC endpoint. To go live, set `WSS_URL`
(Alchemy/QuickNode/local node) in `.env`.

## Tests

```bash
for t in tests/test_*.py; do python "$t"; done      # 15 files, no external deps
# or, with pytest:  pytest -q                        # 85 passed
```

## Going live (real data)

1. Set a node `WSS_URL` in `.env`.
2. Fill `MempoolListener._fetch_tx` (already wired to AsyncWeb3) or use a
   provider with `newPendingTransactionsWithBody`.
3. Replace `wallet_profiler` with `OnChainProfiler` (RPC balance/nonce, optional
   Etherscan for wallet age).
4. Collect signals joined with next-block price moves → train `train.py` with a
   real CSV (`flow_imbalance, whale_net_usd, label`).
5. Use `zeromev_client` for MEV ground-truth and `builder_tip` (coinbase
   transfers) for high-precision MEV labels.

**Data sources (no scraping needed):** zeromev REST API (MEV labels, keyless),
direct RPC (wallet profiles), Dune/Etherscan (bulk labels & history).

## cas-market-simulator entegrasyonu

Bu repo, `cas-market-simulator` icin iki ince, test edilebilir kopru arayuzu
sunar (mevcut 5 katmanli analist cekirdegini degistirmeden). Sozlesme:
`docs/00-ORTAK-SOZLESME.md`.

### Katman 1 -- `FlowFeed` (okuma arayuzu)

```python
from src.api import FlowFeed

feed = FlowFeed(mode="simulation", seed=42)   # WSS_URL yoksa otomatik simulation
state = feed.latest("UniswapV2")              # -> FlowState

state.flow_imbalance     # -1..+1
state.vpin_toxicity      # 0..1
state.whale_net_usd
state.actor_mix          # {"WHALE": .., "MEV_BOT": .., "RETAIL": ..} toplam ~= 1.0
state.direction_prob_up  # 0..1
state.lead_lag_spread
state.regime             # "normal" | "toxic" | "highvol"
state.ts                 # UTC, tz-aware
```

`FlowFeed`, mevcut `RollingFlow`/`predict.predict()` hesaplamalarini
degistirmez -- onlari tek bir disa-donuk `FlowState` struct'inda paketler.
Sim modu birinci sinif: harici API/anahtar olmadan gecerli `FlowState`
uretir. **`FlowState` ham/temiz metrik saglar, agirlik karari vermez** --
`cas-market-simulator`'in kendi `order_flow`/`onchain_flow` faktoruyle
kavramsal ortusme (cift sayim) riski oldugu icin motor tarafi agirligi
kendisi belirler.

### Katman 2 -- ajan sablonu + enjekte edilebilir cevre

```python
from src.actor.agent_profiles import PROFILES   # WHALE / MEV_BOT / RETAIL (veri)
from src.mev.decision import decide_sandwich, decide_jit, decide_arbitrage, decide_builder_tip
from src.api import SimEnvironment
from src.models import AgentOrder
from datetime import datetime, timezone

env = SimEnvironment(seed=1)
order = AgentOrder(token="UniswapV2", side="BUY", size_usd=500_000,
                    actor_label="WHALE", ts=datetime.now(timezone.utc))
state = env.step([order], "UniswapV2")   # enjekte edilen emre gore guncellenmis FlowState
```

`src/actor/agent_profiles.py` uc aktor icin sabit, parametrik profil verir
(kod degil, veri). `src/mev/*.py`'deki MEV tespit fonksiyonlari zaten yan
etkisiz ve deterministiktir; `src/mev/decision.py` bunlari simulator-dostu
tek bir cephede toplar. `MempoolListener` iki modda calisir: **autonomous**
(varsayilan, mevcut davranis) ve **driven** (`MempoolListener(q, driven=True)`
+ `await listener.inject(order)` ile disaridan beslenir).

### Sim / canli mod farki

`WSS_URL` bos ise (`.env`'de varsayilan) her iki arayuz de otomatik olarak
`simulation` moduna duser: deterministik (seed'e bagli) sentetik akisla,
harici bagimlilik olmadan calisir. `WSS_URL` set edildiginde `FlowFeed`/
`MempoolListener` gercek mempool/CEX-DEX besemesini kullanir.

## Disclaimer

For research and educational use only. Not financial advice. Trading crypto
involves substantial risk. The authors are not liable for any losses.

## License

MIT — see [LICENSE](LICENSE).
