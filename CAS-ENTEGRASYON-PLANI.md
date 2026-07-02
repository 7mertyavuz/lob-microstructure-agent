# microstructure-analyzer — CAS Entegrasyon Planı

> Amaç: repo'yu hibrit CAS planına uyumlu hale getirmek; mevcut 5 katmanlı analist
> çekirdeğini **bozmadan** `cas-market-simulator`'a iki yeni arayüz açmak
> (Katman 1: `FlowFeed`, Katman 2: ajan şablonu + enjekte edilebilir çevre).
>
> Kurallar: saf Python+NumPy · sim modu birinci sınıf · geriye uyum (11 test yeşil kalır)
> · UTC tz-aware zaman · ağırlık kararı verme, sadece ham/temiz metrik ver.
>
> Konum notu: "Karar destek / araştırma — yatırım tavsiyesi değildir." korunur.

---

## Mevcut durumun repo haritası (planın dayandığı gerçekler)

| İhtiyaç (FlowState alanı) | Repo'da kaynağı | Durum |
|---|---|---|
| `flow_imbalance` | `predict.predict()` içindeki harmanlanmış imbalance / `FlowFeatures.flow_imbalance` | ✅ var |
| `vpin_toxicity` | `FlowFeatures.vpin` (`features/vpin.py`) | ✅ var |
| `whale_net_usd` | `FlowFeatures.whale_net_usd` | ✅ var |
| `direction_prob_up` | `predict.predict(feat).prob_up` | ✅ var |
| `regime` | `predict/regime.py` `RegimeRouter` / VPIN eşiği | ✅ var (eşleme gerek) |
| `lead_lag_spread` | `features/lead_lag.py` | ⚠️ var ama boru hattına bağlı değil; sim'de sentezlenecek |
| `actor_mix` (WHALE/MEV/RETAIL oranı) | `RollingFlow` tamponunda `label` var ama dışa verilmiyor | ❌ yeni metot gerek |

Diğer gerçekler: `config.simulation_mode` = `WSS_URL` boşsa `True`. `main.py`
şu an `predict()` sonucunu `print` ediyor (struct döndürmüyor). MEV modülleri
(`sandwich/jit_liquidity/arbitrage/builder_tip`) çoğunlukla zaten saf fonksiyon
ama giriş sözleşmeleri belgelenmemiş. `src/api/` klasörü yok.

---

## FAZ 0 — Hazırlık ve emniyet ağı (yarım gün)

**Amaç:** değişikliğe başlamadan önce zemini sabitlemek; regresyonu imkânsıza yakın kılmak.

1. Yeni dal aç: `feat/cas-integration`.
2. Baseline: `pytest -q` çalıştır, 11 dosyanın **hepsinin yeşil** olduğunu kaydet
   (çıktıyı `baseline_tests.txt`'e al). Bu, "geriye uyum" kanıtı.
3. Ortak sözleşmeyi repoya taşı: `docs/00-ORTAK-SOZLESME.md` olarak kopyala ki
   `FlowState`/`FlowFeed` tanımı repo içinden referanslanabilsin (tek doğruluk kaynağı).
4. `src/models.py`'ye (veya yeni `src/api/contracts.py`) `FlowState` dataclass'ını
   **birebir sözleşmedeki alanlarla** ekle. Bu tip, adaptörün çıktı şeması.
5. `src/api/__init__.py` iskeleti oluştur.

**Bitti tanımı:** dal açık, baseline testler yeşil kaydedildi, `FlowState` tipi
import edilebilir, hiçbir mevcut davranış değişmedi.

**Risk/dikkat:** `datetime` alanları UTC tz-aware olmalı (`datetime.now(timezone.utc)`).

---

## FAZ 1 — `FlowFeed` adaptörü (Görev 1 · Katman 1 köprüsü) — çekirdek

**Amaç:** mevcut feature/predict çıktısını `FlowState` struct'ına paketleyen temiz,
test edilebilir bir okuma arayüzü. Hesaplama **yeniden yazılmaz**, sarılır.

### 1a. Eksik metrik: `actor_mix`
- `src/features/window.py` → `RollingFlow`'a `actor_mix(token) -> dict` ekle:
  pencere içindeki `label` başına ağırlıklı hacim payı `{"WHALE","MEV_BOT","RETAIL"}`,
  toplamı 1.0'a normalize. Boş pencerede eşit/nötr dağılım döndür.
- **Additif** değişiklik; mevcut `features()` imzası aynı kalır.

### 1b. `regime` eşlemesi
- `predict/regime.py`'nin ürettiği durumu sözleşme etiketine çevir:
  `normal | toxic | highvol`. Basit eşik kuralı: yüksek VPIN → `toxic`,
  yüksek varyans/HMM toksik-state → `highvol`, aksi halde `normal`.
- Eşleme mantığını `flow_feed.py` içinde küçük saf yardımcıya koy (test edilebilir).

### 1c. `lead_lag_spread`
- Canlı modda `features/lead_lag.py` çıktısını kullan (varsa).
- Sim modunda deterministik sentetik değer üret (seed'e bağlı), çünkü CEX beslemesi yok.

### 1d. `FlowFeed` sınıfı
`src/api/flow_feed.py`:
```python
class FlowFeed:
    def __init__(self, mode: str = "simulation", seed: int | None = None): ...
        # mode: "simulation" | "live"; WSS_URL yoksa otomatik simulation
    def latest(self, token: str) -> FlowState: ...
```
- Bir iç `RollingFlow` örneğini besleyip `FlowState`'in yedi alanını doldurur:
  `flow_imbalance`←harmanlı imbalance, `vpin_toxicity`←vpin,
  `whale_net_usd`←whale_net, `actor_mix`←(1a), `direction_prob_up`←`predict().prob_up`,
  `lead_lag_spread`←(1c), `regime`←(1b), `ts`←UTC now.
- **`print` yok, struct döndür.** Simülasyon modunda deterministik sentetik akışla
  geçerli `FlowState` üretebilmeli (harici API/anahtar olmadan).

### 1e. `main.py` refaktörü (davranış korunur)
- `main.py` döngüsü `FlowFeed`'i kullanacak şekilde düzenlenebilir; **ekrana basılan
  çıktı ve akış aynı kalmalı.** Yani print katmanı `FlowFeed.latest()` üstünde ince
  bir sunum katmanına dönüşür. Refaktör opsiyonel ama önerilir (tek yol tutar).

### 1f. Testler (`tests/test_flow_feed.py`)
- `latest()` sim modunda yedi alanı da dolu, tip-doğru `FlowState` döndürüyor.
- `actor_mix` toplamı ≈ 1.0.
- `regime` sadece izinli üç değerden biri.
- Determinizm: aynı seed → aynı çıktı.
- `vpin_toxicity ∈ [0,1]`, `direction_prob_up ∈ [0,1]`, `flow_imbalance ∈ [-1,1]`.

**Bitti tanımı:** `FlowFeed.latest(token)` sim **ve** canlı modda geçerli `FlowState`
döndürüyor; yeni testler + 11 mevcut test yeşil.

**Risk/dikkat:** Çift sayım — `FlowState`, yeni motorun `order_flow`/`onchain_flow`
faktörüyle örtüşür. Burada **ham/temiz metrik** üret, ağırlık verme; bunu
`docs/00-ORTAK-SOZLESME.md`'de ve `flow_feed.py` docstring'inde açıkça yaz.

---

## FAZ 2 — Ajan davranış şablonu + saf MEV fonksiyonları (Görev 2 · Katman 2)

**Amaç:** simülatörün aktör tiplerini sentetik ajan olarak kullanabilmesi için
*veri olarak* profil + *yan etkisiz* MEV karar fonksiyonları.

### 2a. `src/actor/agent_profiles.py` (yeni — kod değil veri)
```python
@dataclass(frozen=True)
class AgentProfile:
    size_usd: tuple[float, float]   # tipik emir boyutu aralığı
    freq: str                       # "low" | "mid" | "high"
    aggression: float               # 0..1 gas/agresiflik
    trigger: str                    # tetikleyici koşul ifadesi

WHALE   = AgentProfile((50_000, 5_000_000), "low",  0.3, "onchain_flow")
MEV_BOT = AgentProfile((1_000,   200_000),  "high", 1.0, "victim_in_mempool")
RETAIL  = AgentProfile((50,       5_000),   "mid",  0.5, "sentiment|momentum")
PROFILES = {"WHALE": WHALE, "MEV_BOT": MEV_BOT, "RETAIL": RETAIL}
```
- Parametreler `src/actor/classifier.py`'deki mevcut aktör ağırlıkları/eşikleriyle
  tutarlı seçilir (çelişki olmasın).

### 2b. MEV mantığını saf fonksiyona sağlamlaştır
- `src/mev/{sandwich,jit_liquidity,arbitrage,builder_tip}.py` fonksiyonlarını
  gözden geçir: **yan etkisiz, deterministik, dışarıdan çağrılabilir** olduklarını
  garanti et (global durum/log yazımı yoksa dokunma; varsa ayıkla).
- Her birine net giriş/çıkış sözleşmesi (docstring) ekle: simülatörde MEV ajanının
  "karar fonksiyonu" olarak nasıl çağrılacağı.
- İsteğe bağlı ince cephe: `src/mev/decision.py` — profil + gözlem alıp
  `detect_*` fonksiyonlarını çağıran tek giriş noktası (simülatör dostu).

### 2c. Testler (`tests/test_agent_profiles.py`)
- Üç profil mevcut, alanlar sözleşmeye uygun.
- MEV karar fonksiyonları saf: aynı girdi → aynı çıktı, dış durum değişmiyor.
- **Mevcut `tests/test_mev.py` değişmeden yeşil** (saflaştırma regresyon yaratmadı).

**Bitti tanımı:** `agent_profiles.py` üç aktör için parametrik profil veriyor;
MEV mantığı saf fonksiyon olarak çağrılabiliyor; eski + yeni testler yeşil.

**Risk/dikkat:** MEV refaktörü gizli davranış değiştirebilir — `test_mev.py` bekçidir;
önce testi anla, sonra dokun.

---

## FAZ 3 — Enjekte edilebilir çevre (Görev 3 · sim = Environment)

**Amaç:** sim akış üretecini, simülatörün `Environment` katmanından **dışarıdan ajan
emri kabul eden** ve akış metriklerinin buna tepki verdiği bir arayüze dönüştürmek.

### 3a. Emir giriş sözleşmesi
- Basit bir `AgentOrder` yapısı tanımla (token, side, size_usd, actor_label, ts).
  `src/ingest/mempool_listener.py`'nin sentetik ürettiği `PendingTx` ile eşdeğer olsun
  ki aynı decode→classify→feature yolundan geçebilsin.

### 3b. `mempool_listener` sim modunu enjekte edilebilir yap
- Mevcut `_run_simulation()` kendi kendine sentetik tx üretiyor. Yanına
  **push arayüzü** ekle: `inject(order: AgentOrder)` → kuyruğa/akışa besle.
- İki mod: (a) *autonomous* (mevcut, kendi üretir — davranış korunur),
  (b) *driven* (dışarıdan enjekte edilen emirlerle çalışır). Varsayılan autonomous.

### 3c. Geri-bildirim döngüsü
- Simülatör "şu ajan şu emri verdi" dediğinde: emir → decode/classify → `RollingFlow` →
  `FlowFeed.latest()` metrikleri **güncellenmiş** dönmeli. Yani enjeksiyon sonrası
  `flow_imbalance`/`vpin`/`actor_mix` ölçülebilir şekilde değişir.
- İnce bir `SimEnvironment` adaptörü (`src/api/sim_env.py`) bu döngüyü kapsayabilir:
  `step(orders) -> FlowState`.

### 3d. Testler (`tests/test_sim_env.py`)
- Enjekte edilen büyük WHALE alışı → `flow_imbalance` yukarı, `whale_net_usd` artar.
- Autonomous mod hâlâ eskisi gibi çalışıyor (geriye uyum).
- Determinizm: aynı emir dizisi + seed → aynı `FlowState` dizisi (replay).

**Bitti tanımı:** sim modu dışarıdan ajan emri kabul ediyor; metrikler tepki veriyor;
autonomous mod korunmuş; yeni testler yeşil.

**Risk/dikkat:** Determinizm kritik (simülatör replay/test için). `time.time()` yerine
enjekte edilebilir saat kullan ki testler stabil olsun.

---

## FAZ 4 — Entegrasyon, doğrulama, dokümantasyon (kapanış)

1. **Uçtan uca tam test:** `pytest -q` → 11 mevcut + Faz 1-3 yeni testler hepsi yeşil.
2. **Sözleşme uyum kontrolü:** `FlowState` alan adları/tipleri `00-ORTAK-SOZLESME.md`
   ile birebir aynı mı? (küçük bir `test_contract_shape.py` ile alan seti doğrula).
3. **Çift sayım notu:** `FlowState` ↔ motor `order_flow` örtüşmesini README + docstring'de
   belgele (motor düşük ağırlık verecek; biz sadece ham metrik veriyoruz).
4. **README:** "cas-market-simulator entegrasyonu" bölümü ekle — `FlowFeed` kullanımı,
   ajan profilleri, enjekte edilebilir çevre örneği, sim/live mod farkı.
5. **Konum ibaresi** ("yatırım tavsiyesi değildir") korunmuş mu kontrol et.
6. (İsteğe bağlı, yüksek değer) Doğrulama için `engineering:code-review` skill'iyle
   diff'i gözden geçir: saflık, determinizm, geriye uyum.

**Bitti tanımı (tüm proje):** dört "bitti" ölçütü sağlanır —
`FlowFeed.latest()` sim+canlı geçerli · `agent_profiles.py` + saf MEV · enjekte edilebilir
çevre · yeni+mevcut testler yeşil · README güncel.

---

## FAZ 5 — İleri / kapsam dışı (şimdi yapılmaz, kayıt altında)

- **SUI desteği:** object-centric tx modeli ayrı epik. Decode katmanı EVM'e kurulu;
  şimdi üstlenilmez. Ayrı dal/plan.
- **Canlı `lead_lag` beslemesi:** gerçek CEX-DEX akışıyla `lead_lag_spread`'i
  tam bağlama (şimdilik sim'de sentetik).
- **Simülatör tarafı ajanlar** (momentum/market_maker/panic/liquidation_engine):
  bunlar `cas-market-simulator/agents/` altına yazılır — **bu repoya değil.**

---

## Sıra ve bağımlılıklar (özet)

```
Faz 0 (zemin) ─► Faz 1 (FlowFeed) ─► Faz 4 (kapanış)
                      │
Faz 2 (profiller+MEV) ┤   ← Faz 1'den bağımsız, paralel gidebilir
                      │
Faz 3 (enjekte çevre) ┘   ← Faz 1'in RollingFlow beslemesine dayanır
```

Kritik yol: **Faz 0 → Faz 1 → Faz 3 → Faz 4.** Faz 2 araya paralel sığar.
Her faz kendi testleriyle "yeşil" bırakılmadan sonrakine geçilmez.
