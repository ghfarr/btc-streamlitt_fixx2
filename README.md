# ₿ BTC Price Prediction Dashboard

Dashboard prediksi harga Bitcoin menggunakan pendekatan Machine Learning berbasis:

* Data historis Bitcoin (CoinGecko)
* Sentimen pasar dari Twitter/X (HuggingFace Dataset)
* Tiga model Machine Learning:

  * Linear Regression
  * Random Forest
  * Long Short-Term Memory (LSTM)

Penelitian membandingkan tiga skenario data:

1. Historical
2. Sentiment
3. Combined (Historical + Sentiment)

dengan dua horizon prediksi:

* t+1 hari
* t+30 hari

---

# 📂 Struktur Proyek

```text
btc_streamlit/
│
├── app.py
├── precompute.py
├── requirements.txt
├── requirements-train.txt
├── runtime.txt
├── README.md
│
├── data/
│   ├── btc_2021_01_01_to_2024_09_30.csv
│   ├── btc.parquet
│   ├── sentiment.parquet
│   ├── features.parquet
│   ├── predictions.parquet
│   ├── metrics.parquet
│   ├── metrics.csv
│   ├── tweets_sample.parquet
│   └── feature_cols.json
│
└── notebook/
    └── Analisis Prediksi Harga Bitcoin.ipynb
```

---

# 📊 Dataset

## Data Historis Bitcoin

Sumber:

* CoinGecko

Periode:

* 1 Januari 2021 – 30 September 2024

Variabel utama:

* Close Price
* Market Capitalization
* Trading Volume

Catatan:

* Dataset dibatasi sampai September 2024 mengikuti batas ketersediaan data sentimen dari HuggingFace

---

## Data Sentimen Pasar

Sumber:

* HuggingFace Dataset
* StephanAkkerman/financial-tweets-crypto

Periode:

* 27 Desember 2021 – 5 September 2024

Filter:

* Tweet yang mengandung kata:

  * Bitcoin
  * BTC

Jumlah tweet yang digunakan:

* ±20.539 tweet

Analisis sentimen:

* VADER Sentiment Analysis

Fitur sentimen:

* sentiment_mean
* sentiment_pos
* sentiment_neg

---

# ⚙️ Feature Engineering

## Historical Features (13 fitur)

* close
* market_cap
* volume
* close_lag1
* close_lag3
* close_lag7
* ma7
* ma14
* ma30
* std7
* std14
* ret_1d
* ret_7d

## Sentiment Features (3 fitur)

* sentiment_mean
* sentiment_pos
* sentiment_neg

## Combined Features (16 fitur)

Gabungan seluruh fitur Historical dan Sentiment.

---

# 🎯 Target Prediksi

## Horizon t+1

Prediksi harga Bitcoin 1 hari ke depan.

## Horizon t+30

Prediksi harga Bitcoin 30 hari ke depan.

---

# 🤖 Model Machine Learning

## Linear Regression

Model regresi linear sebagai baseline.

## Random Forest

Model ensemble berbasis decision tree.

Parameter utama:

* n_estimators = 300
* max_depth = 10
* min_samples_leaf = 2

## LSTM

Model deep learning untuk data time series.

Parameter utama:

* Window = 30
* Units = 64
* Dropout = 0.2
* Epochs = 30
* Batch Size = 32

---

# 🔀 Data Split

Metode pembagian data menggunakan Time Series Split tanpa shuffle (80-10-10 kronologis).

| Dataset    | Periode                          | Proporsi |
| ---------- | -------------------------------- | -------- |
| Training   | 1 Januari 2021 – 31 Desember 2023 | 80%      |
| Validation | 1 Januari 2024 – 16 Mei 2024     | 10%      |
| Testing    | 17 Mei 2024 – 30 September 2024  | 10%      |

Untuk menghindari data leakage:

* Split dilakukan secara kronologis tanpa pengacakan data
* StandardScaler hanya di-fit pada data training
* Lag feature dihitung secara causal
* Rolling statistic dihitung secara causal
* LSTM menggunakan windowing one-step-ahead

---

# 📈 Dashboard Features

## Overview

* Ringkasan dataset
* Statistik penelitian
* Visualisasi harga Bitcoin
* Best model per skenario dan horizon

## Data & Sentiment

* Visualisasi harga Bitcoin (2021-2024)
* Visualisasi sentimen harian
* Distribusi tweet per tahun
* Sample tweet

## Feature Engineering

* Daftar fitur tiap skenario
* Korelasi fitur terhadap target

## Model Predictions

* Actual vs Predicted
* Error per tanggal
* Ringkasan error model

## Scenario Comparison

* Perbandingan Historical vs Sentiment vs Combined
* Heatmap performa model
* Best result per skenario

## Metrics & Comparison

* Tabel lengkap RMSE, MAE, MAPE
* Bar chart perbandingan antar model
* Best model per horizon

## Kesimpulan

Ringkasan hasil penelitian dan rekomendasi pengembangan lanjutan.

---

# 🏆 Hasil Utama Penelitian

Hasil pengujian menunjukkan:

* Skenario Historical menghasilkan performa terbaik.
* Skenario Sentiment menghasilkan performa terendah.
* Skenario Combined belum memberikan peningkatan yang signifikan dibanding Historical.
* Linear Regression menjadi model terbaik pada sebagian besar eksperimen berdasarkan nilai RMSE terendah.

Kesimpulan utama:

Data historis Bitcoin memiliki kontribusi yang lebih besar terhadap akurasi prediksi dibandingkan data sentimen pasar pada dataset dan periode penelitian ini.

---

# 🚀 Menjalankan Dashboard Secara Lokal

Install dependencies:

```bash
pip install -r requirements.txt
```

Jalankan Streamlit:

```bash
streamlit run app.py
```

Dashboard akan tersedia di:

```text
http://localhost:8501
```

---

# 🔄 Re-generate Artifacts

Jika ingin menjalankan ulang proses training dan menyimpan ulang artifacts:

```bash
pip install -r requirements-train.txt
python precompute.py
```

Pastikan file `btc_2021_01_01_to_2024_09_30.csv` sudah ada di folder `data/`.

---

# ☁️ Deploy ke Streamlit Community Cloud

1. Push project ke GitHub
2. Login ke Streamlit Community Cloud
3. Pilih repository GitHub
4. Main file path:

```text
app.py
```

5. Klik Deploy

---

# 👨‍🎓 Penelitian

Judul:

Analisis Prediksi Harga Bitcoin Menggunakan Machine Learning Berdasarkan Data Historis dan Sentimen Pasar

Program Studi Informatika
Universitas Amikom Yogyakarta

---

# License

MIT License
