<div align="center">

# 🔬 lob-microstructure-agent

**Real-time DEX market microstructure analysis engine**
_From mempool to actor labeling, from order-flow reads to a virtual order book._

<br/>

![tests](https://img.shields.io/badge/tests-140%20passing-2ea44f?style=flat-square)
![python](https://img.shields.io/badge/python-3.11%2B-3776ab?style=flat-square&logo=python&logoColor=white)
![deps](https://img.shields.io/badge/deps-pure%20Python%20%2B%20NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![simulation](https://img.shields.io/badge/simulation-first-6f42c1?style=flat-square)
![license](https://img.shields.io/badge/license-MIT-black?style=flat-square)

**🇬🇧 English** · [🇹🇷 Türkçe](README.md)

</div>

---

> ⚠️ **Research / PoC — not investment advice.** The probabilities and labels produced are *observational and predictive inferences*, not certainties.

---

## 🎯 What does it do?

It watches **pending swaps** in the Ethereum mempool, decodes Uniswap V2/V3/V4 + Universal Router calls, and classifies the actor behind each transaction:

```text
⚑ WHALE    — large-volume, direction-setting trade
⚑ MEV_BOT  — sandwich / JIT / arbitrage hunter
⚑ RETAIL   — small, momentum follower
```

From this flow it produces **order-flow imbalance**, **VPIN toxicity**, **actor mix**, and a short-term direction probability. In addition, thanks to a V3-style **virtual order book**, you can read an AMM like a classic L2 book and compute the expected price impact.

> The goal is not a "buy/sell now" machine; it is to **measure the microstructure of the market transparently** and expose it to upstream systems as raw metrics.

---

## 🏗️ Architecture

```mermaid
flowchart TD
    subgraph INGEST["1 · Ingest"]
        M["Mempool WSS"]
        SIM["Deterministic simulation"]
    end

    subgraph DECODE["2 · Decode"]
        D["Uniswap V2/V3/V4<br/>Universal Router"]
    end

    subgraph ACTOR["3 · Actor Layer"]
        A["WHALE · MEV_BOT · RETAIL"]
    end

    subgraph FEATURES["4 · Features"]
        F1["OFI imbalance"]
        F2["VPIN toxicity"]
        F3["Lead-lag spread"]
        F4["Actor mix"]
    end

    subgraph BOOK["5 · Virtual Book"]
        B1["V3 tick liquidity"]
        B2["Depth / microprice / OFI"]
        B3["Iceberg / spoof / absorption"]
    end

    subgraph PREDICT["6 · Predict"]
        P["Regime router → linear | MLP"]
    end

    INGEST --> DECODE --> ACTOR --> FEATURES --> PREDICT
    ACTOR --> BOOK
    BOOK --> PREDICT
```

---

## 📦 Key Features

| Layer | Feature | Output |
|---|---|---|
| **Decode** | V2 `swapExact*`, V3 `exactInputSingle`, Universal Router `execute()` | Direction, token, volume |
| **Actor** | Wallet profile + gas + coinbase.transfer detection | WHALE / MEV_BOT / RETAIL label |
| **MEV** | Sandwich, JIT liquidity, atomic arbitrage, builder tip | Text-annotated report |
| **Features** | OFI, VPIN, lead-lag, fracdiff | `FlowState` contract |
| **Book** | V3 virtual book, L2/L3 reads | `BookState` contract |
| **Predict** | Regime router + LogReg / NumPy MLP + meta-labeling | `prob_up`, `regime` |

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
# or editable
pip install -e ".[dev]"

# Simulation mode — no RPC/key required
python main.py

# Model calibration
python train.py

# Tests
pytest -q          # 140 tests, all green
```

When `WSS_URL` is left empty, the system automatically falls back to **simulation mode** and runs end-to-end without a real node connection.

---

## 🔗 Place in the CAS Ecosystem

In the hybrid CAS plan, this repo acts as the **microstructure sensory organ**:

```mermaid
flowchart LR
    A["lob-microstructure-agent"] -->|FlowState| B["cas-market-simulator"]
    A -->|BookState| B
    C["macro-sentiment-agent"] -->|SentimentState + ShockEvent| B
    B -->|Card| D["Decision support card"]
```

The contracts are loosely coupled; `cas-market-simulator` depends only on the `FlowState` / `BookState` / `SentimentState` types.

---

## ⚖️ Disclaimer

The system is for **research and educational purposes only**. It does not submit automated orders and does not provide investment advice. Crypto trading carries significant risk; the authors are not responsible for any losses.

## 📄 License

MIT — see [LICENSE](LICENSE).
