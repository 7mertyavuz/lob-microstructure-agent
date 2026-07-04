# Ortak Sözleşme — `microstructure-analyzer` ↔ `cas-market-simulator`

> Bu doküman iki repo arasındaki **tek doğruluk kaynağıdır**. `FlowState` ve
> `FlowFeed` burada tanımlanan alan adları/tipleriyle birebir eşleşmelidir.
> Değişiklik gerekirse önce burada güncellenir, sonra kod buna uydurulur.

## Kapsam ve rol ayrımı

`microstructure-analyzer` (bu repo) bir **analist**tir: mempool/on-chain akışını
okur, temiz/ham mikroyapı metrikleri üretir. **Ağırlık kararı vermez, pozisyon
büyüklüğü önermez, alım/satım tetiklemez.** `cas-market-simulator` bu metrikleri
kendi ajan/motor mantığında nasıl ağırlıklandıracağına kendisi karar verir.

> Karar destek / araştırma — yatırım tavsiyesi değildir.

## Katman 1 — `FlowState` / `FlowFeed`

### `FlowState` (çıktı şeması)

| Alan | Tip | Aralık/Değerler | Kaynak |
|---|---|---|---|
| `flow_imbalance` | `float` | `[-1, 1]` | `predict.predict()` harmanlı imbalance (fast+slow) |
| `vpin_toxicity` | `float` | `[0, 1]` | `FlowFeatures.vpin` |
| `whale_net_usd` | `float` | `(-inf, inf)` | `FlowFeatures.whale_net_usd` |
| `actor_mix` | `dict[str, float]` | `{"WHALE","MEV_BOT","RETAIL"}` toplamı ≈ 1.0 | `RollingFlow.actor_mix()` |
| `direction_prob_up` | `float` | `[0, 1]` | `predict.predict(feat).prob_up` |
| `lead_lag_spread` | `float` | serbest (tipik `±0.01`) | canlı: `features/lead_lag.py`; sim: deterministik sentetik |
| `regime` | `str` | `"normal" \| "toxic" \| "highvol"` | `predict/regime.py` eşlemesi |
| `ts` | `datetime` | UTC, tz-aware | `datetime.now(timezone.utc)` |
| `token` | `str` | serbest metin | çağıranın verdiği token/bucket anahtarı |

`FlowFeed` bu yedi (+token/ts) alanı dolduran okuma arayüzüdür:

```python
class FlowFeed:
    def __init__(self, mode: str = "simulation", seed: int | None = None): ...
    def latest(self, token: str) -> FlowState: ...
```

- `mode`: `"simulation" | "live"`. `WSS_URL` boşsa otomatik `"simulation"`.
- **Struct döndürür, print etmez.**
- Simülasyon modu **birinci sınıf**: harici API/anahtar olmadan geçerli
  `FlowState` üretebilmeli.

### Çift sayım uyarısı

`FlowState`, motorun kendi `order_flow` / `onchain_flow` faktörleriyle
kavramsal olarak örtüşebilir. Bu yüzden `FlowFeed` **ham/temiz metrik verir,
ağırlık kararı vermez** — motor tarafı bu metriklere ne kadar ağırlık
vereceğine kendisi karar verir (çift sayımdan kaçınmak motorun sorumluluğu).

## Katman 2 — Ajan şablonu + enjekte edilebilir çevre

### `AgentProfile` (veri, davranış değil)

```python
@dataclass(frozen=True)
class AgentProfile:
    size_usd: tuple[float, float]   # tipik emir boyutu aralığı (usd)
    freq: str                       # "low" | "mid" | "high"
    aggression: float               # 0..1
    trigger: str                    # tetikleyici koşul ifadesi (serbest metin)
```

Üç sabit profil: `WHALE`, `MEV_BOT`, `RETAIL` — `src/actor/classifier.py`
eşikleriyle tutarlı.

### MEV karar fonksiyonları

`src/mev/{sandwich,jit_liquidity,arbitrage,builder_tip}.py` — **yan etkisiz,
deterministik**, aynı girdi → aynı çıktı. `src/mev/decision.py` bunları
simülatör-dostu isimlendirilmiş bir cephede sunar: `decide_sandwich`,
`decide_jit`, `decide_arbitrage`, `decide_builder_tip` — hesaplama mantığını
değiştirmez, sadece çağrı arayüzünü sadeleştirir.

### `AgentOrder` (giriş sözleşmesi)

```python
@dataclass
class AgentOrder:
    token: str
    side: str            # "BUY" | "SELL"
    size_usd: float
    actor_label: str      # "WHALE" | "MEV_BOT" | "RETAIL"
    ts: datetime           # UTC tz-aware
```

`src/ingest/mempool_listener.py`'nin ürettiği sentetik `PendingTx` ile eşdeğer
bilgiyi taşır; `inject(order)` ile decode→classify→feature yoluna sokulur.

### `SimEnvironment`

```python
class SimEnvironment:
    def step(self, orders: list[AgentOrder], token: str) -> FlowState: ...
```

Enjekte edilen emirler kendi bağımsız `RollingFlow`'unu besler (autonomous
`FlowFeed`/`MempoolListener` akışını etkilemez); metrikler gözlemlenebilir
şekilde değişir. `MempoolListener(driven=True)` + `await listener.inject(order)`
ile aynı emirler gerçek asyncio kuyruğuna da beslenebilir. Autonomous
(kendi kendine üreten, `driven=False`) mod korunur — bu, varsayılan davranıştır.

## Katman 1b — `BookState` / `BookFeed` (emir defteri okuması)

> Kaynak plan: `DEFTER-L3-OKUMA-PLANI.md`. `FlowState`'ten **ayrı** bir
> struct'tır (akış ≠ defter); çift sayım sorumluluğu yine motordadır.

`BookState`, `microstructure-analyzer`'ın defter okuma çıktısıdır. `FlowFeed`
akışı (flow) okurken, `BookFeed` defteri (book) okur. Analist rolü korunur:
ham/temiz metrik verir, ağırlık kararı vermez.

### `BookState` (çıktı şeması)

| Alan | Tip | Aralık | Açıklama |
|---|---|---|---|
| `symbol` | `str` | — | sembol anahtarı |
| `spread_bps` | `float` | `≥0` | en iyi alış-satış farkı (bps) |
| `microprice` | `float` | `>0` | derinlik-ağırlıklı adil fiyat (Stoikov) |
| `depth_imbalance` | `float` | `[-1,1]` | çok-seviyeli, mesafe-ağırlıklı derinlik dengesizliği |
| `ofi` | `float` | serbest | event-bazlı OFI (Cont-Kukanov-Stoikov) |
| `queue_imbalance` | `float` | `[-1,1]` | en iyi seviye kuyruk dengesizliği |
| `book_slope` | `float` | `≥0` | defter eğimi / esneklik |
| `kyle_lambda` | `float` | `≥0` | hacim başına fiyat etkisi |
| `iceberg_score` | `float` | `[0,1]` | gizli likidite **şüphesi** |
| `spoof_score` | `float` | `[0,1]` | yanıltıcı katmanlama **şüphesi** |
| `absorption` | `float` | `[-1,1]` | `+` = satış baskısı emiliyor (bid güçlü) |
| `liq_map_skew` | `float` | `[-1,1]` | likidasyon yoğunluğu üstte(`+`)/altta(`−`) |
| `ts` | `datetime` | UTC tz-aware | zaman damgası |

```python
class BookFeed:
    def __init__(self, mode: str = "simulation", seed: int | None = None): ...
    def latest(self, symbol: str) -> BookState: ...
```

- `mode`: `"simulation" | "live"`; `WSS_URL` boşsa otomatik `"simulation"`.
- **Struct döndürür, print etmez.** Sim modu birinci sınıf, deterministik
  (aynı seed → aynı `BookState` dizisi).
- `iceberg_score` / `spoof_score` **kanıt değil şüphedir**: tek başına yön oyu
  vermez, motor tarafında güven çarpanı olarak kullanılır. Yüksek `spoof_score`
  iken `depth_imbalance`'ın güveni kısılmalıdır.

### Çift sayım uyarısı (defter)

`depth_imbalance` ile mevcut `flow_imbalance` korelasyonlu çıkabilir; `ofi`
(defter-event) ile işlem-bazlı OFI ayrı faktörlerdir. İkisi de motora **düşük
ağırlıkla** girer; `factor_tracker` pozitif katkı gösterene kadar ağırlık
artmaz ("defter konuşur").

## Belirlilik (determinism) kuralı

Simülatör replay/test edebilsin diye: `time.time()` yerine enjekte edilebilir
saat/seed kullanılır. Aynı seed + aynı emir dizisi → aynı `FlowState` dizisi.

## Sürüm

Bu doküman `feat/cas-integration` dalıyla birlikte oluşturuldu (Faz 0).
Kaynak plan: `CAS-ENTEGRASYON-PLANI.md` (repo kökü).
