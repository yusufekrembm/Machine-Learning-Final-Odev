# HIGGS ML Pipeline — Feature Selection & Hyperparameter Optimization

Üsküdar Üniversitesi · Fen Bilimleri Enstitüsü · **Makine Öğrenmesi Final Ödevi**

HIGGS veri seti üzerinde uçtan uca makine öğrenmesi pipeline'ı: aykırı değer analizi,
ölçekleme, ANOVA tabanlı öznitelik seçimi ve **Nested Cross-Validation** ile
hiperparametre optimizasyonu. KNN, SVM, MLP ve XGBoost karşılaştırılır.

Kod, okunabilirlik için **iki sade dosyada** toplanmıştır:

| Dosya | Görev |
|-------|-------|
| `egitim.py` | Veri yükleme → ön işleme (IQR + MinMax) → ANOVA seçimi → nested CV → grafikler/tablolar → en iyi modeli `model.joblib` olarak kaydeder |
| `tahmin.py` | `model.joblib`'i yükler ve verilen örnekler için tahmin üretir |

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
# 1) Eğitim + değerlendirme (gerçek veri yoksa otomatik sentetik veri üretir)
python egitim.py

# Gerçek HIGGS ile (UCI'dan HIGGS.csv.gz indirip):
HIGGS_PATH=/yol/HIGGS.csv.gz python egitim.py

# Tam 100.000 örnek (SVM-RBF nested CV'de çok yavaştır):
HIGGS_N=100000 HIGGS_PATH=/yol/HIGGS.csv.gz python egitim.py

# 2) Kaydedilen modelle tahmin
python tahmin.py
```

Ortam değişkenleri: `HIGGS_N` (örnek sayısı, varsayılan 4000), `HIGGS_PATH` (gerçek veri yolu).

## Çıktılar

- `results/figures/` — boxplot, öznitelik skorları, ROC eğrileri, metrik karşılaştırma, akış diyagramları (Figure 1 & 2)
- `results/tables/` — aykırı değer raporu, öznitelik sıralaması, model özeti, fold bazında metrikler
- `results/summary.json` — özet · `model.joblib` — kaydedilen en iyi model
- `report/HIGGS_ML_Rapor.docx` — kısa rapor (kod + grafik + yorum)

## Veri Seti

[HIGGS Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/HIGGS) — 11M örnek, 28 özellik,
ikili sınıflandırma. `egitim.py` gerçek dosyayı bulamazsa aynı yapıda sentetik veri üretip
pipeline'ı uçtan uca çalıştırır; gerçek sonuçlar için `HIGGS_PATH` verin.

## Proje Yapısı

```
higgs-ml-pipeline/
├── egitim.py            # eğitim + değerlendirme (tek dosya)
├── tahmin.py            # tahmin (tek dosya)
├── requirements.txt
├── README.md
├── results/
│   ├── figures/         # grafikler (flowchart_A/B dahil)
│   └── tables/          # CSV tablolar
└── report/
    └── HIGGS_ML_Rapor.docx
```
