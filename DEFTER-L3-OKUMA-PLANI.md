# L3 Emir Defteri Okuma Stratejisi — Geliştirme Planı

**Tarih:** 2026-07-04
**Kapsam:** `microstructure-analyzer` çekirdeği + `cas-market-simulator` (signalcore Faz 4 #2 `indicators/orderbook.py`) + çevre (Environment) tarafı.
**İlke:** Mevcut proje kurallarına birebir uyum — sim modu birinci sınıf, saf Python + NumPy, deterministik (seed/enjekte saat), analist ağırlık kararı vermez, factor_tracker onaylamadan ağırlık artmaz ("defter konuşur").

---

## Mevcut durumun dürüst fotoğrafı

**Elimizde ne var:**
- DEX tarafında *yarı-L3* görüş: mempool'daki her pending swap tek tek görülüyor, üstelik **aktör kimliğiyle** (Whale/MEV/Retail) — bu, klasik CEX L3 verisinde bile olmayan bir avantaj.
- Akış özellikleri: aktör-ağırlıklı OFI (`window.py`), VPIN (`vpin.py`), CEX-DEX lead-lag (`lead_lag.py`), fracdiff.
- signalcore'da `indicators/orderbook.py` var ama **tamamen sentetik stub** (`SimOrderbookFeed` rastgele yürüyüş) — gerçek defter okuması yok.
- CAS motorunda **emir defteri yok**: `engine/loop.py` "hepsi aynı tick-sonu fiyattan doluyor" varsayımıyla çalışıyor; `PaperExecutor` slipajı sabit modelle tahmin ediyor.

**Eksik olan:**
1. Gerçek (veya gerçekçi) **L2 defter yeniden inşası** — derinlik, spread, seviye bazlı likidite.
2. **L3-türevi okumalar** — iceberg, spoofing, absorption, sweep, emir ömrü.
3. DEX tarafında **sanal defter** — Uniswap V3 tick likiditesi aslında bir derinlik merdivenidir, hiç kullanılmıyor.
4. Defter özelliklerinin tahmin katmanına (direction/regime/meta) ve signalcore kartına bağlanması.
5. Simülatörde ajanların etkileşeceği gerçek bir eşleşme motoru (emergence gerçekçiliği + slipaj kalibrasyonu).

---

## Faz D0 — Sözleşme ve iskelet *(0.5–1 gün)*

**Amaç:** `BookState` şeması tanımlı, boş boru hattı uçtan uca bağlı.

- `docs/00-ORTAK-SOZLESME.md`'ye **`BookState`** eklenir (FlowState'e dokunulmaz — ayrı struct, çift sayım sorumluluğu yine motorda):

| Alan | Tip | Aralık | Açıklama |
|---|---|---|---|
| `spread_bps` | float | ≥0 | en iyi alış-satış farkı |
| `microprice` | float | >0 | derinlik-ağırlıklı adil fiyat (Stoikov) |
| `depth_imbalance` | float | [-1,1] | çok-seviyeli, mesafe-ağırlıklı derinlik dengesizliği |
| `ofi` | float | serbest | Cont-Kukanov-Stoikov event-bazlı order flow imbalance |
| `queue_imbalance` | float | [-1,1] | en iyi seviyedeki kuyruk dengesizliği |
| `book_slope` | float | ≥0 | defter eğimi / esneklik (fiyat etkisi tahmini) |
| `kyle_lambda` | float | ≥0 | hacim başına fiyat etkisi |
| `iceberg_score` | float | [0,1] | gizli likidite şüphesi |
| `spoof_score` | float | [0,1] | yanıltıcı katmanlama şüphesi |
| `absorption` | float | [-1,1] | + = satış baskısı emiliyor (bid güçlü) |
| `liq_map_skew` | float | [-1,1] | likidasyon yoğunluğu üstte/altta (mıknatıs etkisi) |
| `ts`, `symbol` | — | — | UTC tz-aware, sembol anahtarı |

- Yeni paket: `lob_microstructure/book/` → `state.py` (BookState), `keeper.py` (defter tutucu, boş), `sim.py` (deterministik sentetik defter üreteci), `features.py` (boş).
- **`SimBookFeed`**: seed'li, rejim-anahtarlamalı sentetik L2 defteri (sakin/toksik/geniş-spread rejimleri) — mevcut `SimOrderbookFeed`'in ciddi hâli. Aynı seed → aynı defter dizisi.
- `FlowFeed`'e paralel **`BookFeed`** okuma arayüzü: `latest(symbol) -> BookState`.

**Bitti:** `BookFeed(mode="simulation").latest("BTCUSDT")` geçerli BookState döndürüyor, testli.

---

## Faz D1 — Gerçek L2 defter yeniden inşası (CEX) *(2–3 gün)*

**Amaç:** anahtarsız, ücretsiz kaynaklardan canlı yerel defter kopyası.

- **Binance WS** (anahtar gerektirmez — projenin "keyless-first" felsefesine uygun):
  - `depth@100ms` diff akışı + REST snapshot → klasik *snapshot + diff senkronizasyonu* (sequence gap kontrolü, kopuşta yeniden senkron).
  - `aggTrade` → işlem bandı (tape).
  - `forceOrder` → **gerçek likidasyon olayları** (likidasyon haritası için altın kaynak).
  - REST: funding, open interest (likidasyon kümeleri tahmini için).
- `lob_microstructure/book/keeper.py::BookKeeper` — seviye→miktar sözlüğü, en iyi N seviye görünümü, O(1) güncelleme; saf Python.
- `lead_lag.py`'deki stub CEX fiyatı gerçek best-bid/ask orta noktasıyla beslenir → **lead-lag artık canlıda gerçek** (mevcut tahmin katmanına ilk somut katkı, bedavaya gelir).
- WSS yoksa otomatik `SimBookFeed`'e düşer (mevcut `WSS_URL` deseniyle aynı).

**Bitti:** canlıda yerel defter CEX arayüzüyle birebir tutarlı (checksum/derinlik karşılaştırma testi); sim modda deterministik.

---

## Faz D2 — Çekirdek L2 okuma özellikleri *(2–3 gün)*

**Amaç:** `lob_microstructure/book/features.py` — her biri tek tek testli, saf fonksiyonlar.

1. **Çok-seviyeli derinlik dengesizliği** — mid'e uzaklıkla üstel sönümlü ağırlık: `imb = (Σw·bid_qty − Σw·ask_qty) / (Σw·bid_qty + Σw·ask_qty)`. Tek seviyeli naif imbalance'tan (stub'daki gibi) çok daha dayanıklı.
2. **Microprice (Stoikov)** — `(ask_qty·bid + bid_qty·ask)/(bid_qty+ask_qty)`; mid yerine adil fiyat. Lead-lag ve slipaj modeli bunu kullanır.
3. **Event-bazlı OFI (Cont-Kukanov-Stoikov)** — best quote değişim olaylarından; mevcut işlem-bazlı OFI'yi tamamlar (ikisi ayrı faktör, korelasyonu factor_tracker ölçer).
4. **Defter eğimi / Kyle's λ** — kümülatif derinlik eğrisine regresyon → "X USD market emri fiyatı kaç bps oynatır". `PaperExecutor` slipaj modelinin gerçek girdisi olur.
5. **Kuyruk dengesizliği + spread dinamiği** — spread'in kendi z-skoru (ani genişleme = bilgili akış/haber öncüsü; macro-sentiment `ShockEvent` ile çapraz doğrulanır).
6. **Likidite boşlukları** — derinlik profilinde delikler (ince bölgeye girince hızlanma beklentisi).

**Bitti:** 6 özellik sentetik defter senaryolarıyla (bilinen cevaplı) test edilmiş; `BookState` alanları gerçek hesaplamayla doluyor.

---

## Faz D3 — L3-türevi okumalar: defterin "niyeti" *(3–4 gün)*

**Amaç:** ham L3 verisi olmadan, diff+tape kesişiminden emir-davranışı çıkarımı. Hepsi kural-tabanlı, parametrik, şeffaf (projenin "sezgi kodlama yok" kuralı).

1. **Iceberg tespiti** — aynı seviyede işlem sonrası anında yenilenen miktar (refill oranı + tekrar sayısı → `iceberg_score`). Gizli alıcı/satıcı = güçlü seviye.
2. **Spoofing/katmanlama sezgisi** — mid'den uzakta beliren büyük pasif blokların işlem görmeden iptali; iptal/işlem oranı, emir ömrü dağılımı (fleeting orders). `spoof_score` yüksekken `depth_imbalance`'ın güveni **kısılır** (yanıltıcı derinliğe kanma).
3. **Absorption (emilim)** — tape'te yoğun agresif satış + fiyat düşmüyor + bid yenileniyor → pozitif absorption (dip sinyali adayı). Tersi dağıtım.
4. **Sweep tespiti** — tek yönde çok seviyeyi süpüren agresif market emirleri (momentum ateşleyici; mevcut whale sinyaliyle çapraz teyit).
5. **Likidasyon haritası** — `forceOrder` gerçek olayları + OI/funding'den tahmini kaldıraç kümeleri → fiyat üstü/altı likidasyon yoğunluğu (`liq_map_skew`, mıknatıs etkisi). Stub'daki rastgele skew gerçek veriye bağlanır.
6. **DEX-mempool çaprazı (bize özgü avantaj):** pending whale emri görüldüğünde CEX defterinin **önceden** nasıl konumlandığını ölç — MM'ler çekiliyorsa bilgili akış teyidi; JIT likidite = DEX'in iceberg'i olarak etiketlenir (mevcut `jit_liquidity.py` ile birleşir).

**Bitti:** her sezgi bilinen senaryolu sentetik testle doğrulanmış; skorlar 0..1 normalize.

---

## Faz D4 — DEX sanal defteri *(2 gün)*

**Amaç:** AMM'yi defter gibi okumak — projenin en özgün parçası.

- **Uniswap V3 tick likiditesi → derinlik merdiveni:** havuzun tick başına likidite dağılımı, birebir bir L2 defteridir. `slot0` + tick bitmap okuması (canlı) / sentetik dağılım (sim) → `BookState`'in DEX versiyonu.
- **Mempool = gelen market emirleri kuyruğu:** pending swap'lar henüz eşleşmemiş agresif emirlerdir. Sanal defter derinliği + pending akış → **beklenen fiyat etkisi** (kaç tick kayar, hangi likidasyon/LP bölgesine girer).
- CEX defteri ↔ DEX sanal defteri **karşılaştırmalı derinlik**: hangi taraf ince → arbitraj yönü ve lead-lag'in nedensel açıklaması.

**Bitti:** `BookFeed` `venue="dex"` ile V3 havuzundan sanal BookState üretiyor; pending emir etkisi deterministik hesaplanıyor.

---

## Faz D5 — Tahmin katmanına bağlama *(1–2 gün)*

**Amaç:** okuma → karar desteği; sözleşme disiplini korunur.

- `predict/direction.py` özellik vektörüne defter özellikleri **düşük ağırlıkla** eklenir; `train.py` yeniden kalibre eder.
- `predict/regime.py`: spread_z + depth → **likidite rejimi** boyutu (normal/ince/toksik). İnce defterde MLP'ye yönlendirme mantıklı (mevcut VPIN yönlendirmesine ek).
- `predict/meta.py` (pozisyon boyutu): Kyle's λ ve derinlik → "bu boyut bu defterde taşınır mı" — economic.py'nin maliyet filtresi gerçek slipaj tahminiyle güçlenir.
- Dashboard: derinlik ısı haritası (Bookmap-vari), likidasyon haritası şeridi, iceberg/spoof rozetleri.

**Bitti:** defter özellikli model, defter özelliksiz baseline'a karşı walk-forward'da raporlanıyor (kazanım iddiası değil, **ölçüm**).

---

## Faz D6 — signalcore sensörü gerçeğe bağlama *(1 gün)*

**Amaç:** FAZ-PLANI Faz 4 #2'nin hakkını vermek.

- `SimOrderbookFeed` korunur (sim birinci sınıf) ama `orderbook_factor` girdisi genişler: `OrderbookState`'e `microprice_dev`, `ofi`, `spoof_score`, `absorption` eklenir.
- Adaptör: `microstructure-analyzer.BookFeed` → signalcore `OrderbookState` (00-SOZLESME'ye ek).
- Oy mantığı: `absorption` ve `iceberg` teyit edici (güven çarpanı), `spoof_score` cezalandırıcı; yön hâlâ `depth_imbalance + liq_skew`.
- **Kural:** yeni faktör düşük ağırlıkla girer; `factor_tracker` pozitif katkı gösterene kadar ağırlık artmaz.

**Bitti:** kartta gerçek defter oyu; factor_tracker IC/hit-rate izliyor.

---

## Faz D7 — Simülatörde gerçek emir defteri çevresi *(3–4 gün, opsiyonel ama değerli)*

**Amaç:** ajanlar defterle etkileşir → emergence gerçekçiliği + okuma stratejisinin laboratuvarı.

- `environment/orderbook.py`: basit fiyat-zaman öncelikli eşleşme motoru (limit/market/iptal). Saf Python, ~200 satır hedef.
- `market_maker.py` gerçek kotasyon verir, `liquidation_engine` defteri süpürür → flash crash artık **defter mekaniğiyle** üretilir (tick-sonu fiyat varsayımı yerine).
- **Kapalı devre doğrulama:** D2-D3 okuma özellikleri simülatör defterinde çalıştırılır — spoofing ajanı eklediğinde `spoof_score` yükseliyor mu? Iceberg ajanı `iceberg_score`'u tetikliyor mu? Okuma stratejisinin **kontrollü deneyle** doğrulanabildiği tek yer burası.
- Kalibrasyon (Faz 9 ilkesi): defterden çıkan spread/derinlik/etki dağılımları stilize gerçeklerle karşılaştırılır.

**Bitti:** scriptli spoof/iceberg senaryoları ilgili skorları ölçülebilir şekilde tetikliyor.

---

## Faz D8 — Kayıt/replay + doğrulama disiplini *(1–2 gün)*

- **Defter kaydedici:** canlı diff+tape akışını sıkıştırılmış olarak diske yaz → deterministik replay (aynı kayıt → aynı BookState dizisi). Test ve backtest'in temeli.
- Journal (`analysis/journal.py`) forward-test defterine defter-kaynaklı sinyaller ayrı etiketle girer → hangi okuma gerçekten katkı veriyor, defter konuşur.
- Leakage kontrolü: defter özellikleri yalnızca t anına kadar olan olaylardan (bar-içi sızıntı testi).

---

## Öncelik ve sıra önerisi

| Sıra | Faz | Neden önce |
|---|---|---|
| 1 | D0 + D1 | Temel; lead-lag'i bedavaya gerçekleştirir |
| 2 | D2 | En yüksek sinyal/emek oranı (microprice, OFI, λ) |
| 3 | D5 (kısmi) | Erken ölçüm: katkı var mı yok mu erken görülür |
| 4 | D3 | Ayırt edici okumalar (iceberg/spoof/absorption/likidasyon) |
| 5 | D6 | signalcore kartına bağla, factor_tracker'a teslim et |
| 6 | D4 | Özgün DEX avantajı |
| 7 | D7 + D8 | Laboratuvar + disiplin |

**Toplam kaba tahmin:** ~2.5–3 hafta tek kişilik efor; her faz tek başına değer üretir, D2'de bile durulabilir.

## Kritik uyarılar

1. **Çift sayım:** `depth_imbalance` ile mevcut `flow_imbalance` korelasyonlu çıkabilir — ikisi de düşük ağırlıkla girer, factor_tracker karar verir.
2. **HFT yanılgısı:** 100ms diff akışıyla gerçek HFT sinyali (mikrosaniye) yakalanmaz; hedef **saniye-dakika ufku** yön/rejim okuması. Plan buna göre boyutlandırıldı.
3. **Spoof/iceberg skorları kanıt değil şüphedir** — kartta tek başına yön oyu vermez, güven çarpanı olarak kullanılır.
4. **Sim ≠ kehanet:** D7 defteri kalibre edilmeden ondan strateji sonucu çıkarılmaz.
