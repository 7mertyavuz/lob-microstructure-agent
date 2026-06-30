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
  ingest/    mempool_listener.py     # Layer 1
  decode/    tx_decoder.py           # Layer 2 (V2/V3/V4/UR)
  actor/     classifier.py · wallet_profiler.py · onchain_profiler.py
  mev/       sandwich.py · jit_liquidity.py · arbitrage.py · builder_tip.py · zeromev_client.py
  features/  window.py · vpin.py · lead_lag.py · fracdiff.py
  predict/   direction.py · regime.py · mlp.py · meta.py · economic.py
  train/     dataset.py · logreg.py · backtest.py · online.py
  pipeline/  bus.py
main.py · dashboard.py · dashboard.html · train.py · config.py
tests/       11 test files
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
for t in tests/test_*.py; do python "$t"; done      # 11 files, no external deps
# or, with pytest:  pytest -q
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

## Disclaimer

For research and educational use only. Not financial advice. Trading crypto
involves substantial risk. The authors are not liable for any losses.

## License

MIT — see [LICENSE](LICENSE).
