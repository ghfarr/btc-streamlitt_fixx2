"""
BTC Price Prediction Dashboard — Streamlit App
Berbasis data CoinGecko + sentimen Twitter (HuggingFace dataset) + ML (LR / RF / LSTM).

Cara pakai:
    streamlit run app.py

Catatan deployment:
- Artifacts (predictions, metrics, sentiment) sudah di-precompute di folder `data/`,
  jadi app ringan & cepat di Streamlit Community Cloud.
- Kalau mau retrain: jalankan `python precompute.py` lokal dulu, lalu push
  ulang folder `data/` ke GitHub.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# Config

st.set_page_config(
    page_title="BTC Price Prediction Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"

MODEL_COLORS = {
    "LinearRegression": "#1f77b4",
    "RandomForest": "#2ca02c",
    "LSTM": "#d62728",
}
ACTUAL_COLOR = "#111111"



# Data loaders (cached)

@st.cache_data(show_spinner=False)
def load_btc() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "btc.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_sentiment() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "sentiment.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "features.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "predictions.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "metrics.parquet")


@st.cache_data(show_spinner=False)
def load_tweet_sample() -> pd.DataFrame:
    p = DATA_DIR / "tweets_sample.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_feature_cols():
    with open(DATA_DIR / "feature_cols.json") as f:
        return json.load(f)



# Sidebar — global controls

st.sidebar.title("₿ BTC Predictor")
st.sidebar.caption("Bitcoin price prediction — ML + sentiment Twitter")

page = st.sidebar.radio(
    "Halaman",
    [
        " Overview",
        " Data & Sentiment",
        " Feature Engineering",
        " Model Predictions",
        " Scenario Comparison",
        " Metrics & Comparison",
        " Kesimpulan",
    ],
)

st.sidebar.divider()
st.sidebar.markdown(
    """
**Setup**
- Train (80%): 2021-01 → 2023-12
- Validation (10%): 2024-01 → 2024-05
- Test (10%): 2024-05 → 2024-09
- Models: Linear Regression, Random Forest, LSTM
- Sentimen: VADER atas tweet BTC dari HuggingFace dataset
"""
)
st.sidebar.divider()
st.sidebar.caption("Built with Streamlit · Data: CoinGecko + HuggingFace")



# Page: Overview

def page_overview():
    st.title("₿ BTC Price Prediction Dashboard")
    st.markdown(
    "Memprediksi harga **Bitcoin** menggunakan tiga skenario data: "
    "**Historical**, **Sentiment**, dan **Historical + Sentiment** "
    "dengan model **Linear Regression**, **Random Forest**, dan **LSTM** "
    "untuk horizon **t+1 hari** dan **t+30 hari**."
)

    btc = load_btc()
    sentiment = load_sentiment()
    metrics = load_metrics()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BTC rows", f"{len(btc):,}")
    c2.metric("Date range", f"{btc['date'].min():%Y-%m-%d}\n→ {btc['date'].max():%Y-%m-%d}")
    c3.metric("Total tweets dipakai", f"{int(sentiment['n_tweets'].sum()):,}")
    c4.metric(
    "Skenario",
    metrics["scenario"].nunique()
)

    st.divider()

    # Key chart: BTC price
    st.subheader("Harga BTC sepanjang periode (2021–2024)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=btc["date"], y=btc["close"], mode="lines",
                             name="BTC Close", line=dict(color="#f7931a", width=1.6)))
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Tanggal", yaxis_title="USD",
        hovermode="x unified",
    )
    fig.add_vrect(x0="2024-01-01", x1="2024-05-16", fillcolor="LightSkyBlue",
                  opacity=0.2, line_width=0, annotation_text="Validation", annotation_position="top left")
    fig.add_vrect(x0="2024-05-17", x1="2024-09-30", fillcolor="LightSalmon",
                  opacity=0.2, line_width=0, annotation_text="Test", annotation_position="top left")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Best Model per Scenario dan Horizon (Test RMSE Terendah)")
    best = (
    metrics[metrics["split"] == "test"]
    .sort_values(
        ["scenario", "horizon", "RMSE"]
    )
    .groupby(
        ["scenario", "horizon"]
    )
    .head(1)
    .reset_index(drop=True)
)
    st.dataframe(
        best[
    [
        "scenario",
        "horizon",
        "model",
        "RMSE",
        "MAE",
        "MAPE"
    ]
].style.format({"RMSE": "{:.2f}", "MAE": "{:.2f}", "MAPE": "{:.2f}%"}),
        use_container_width=True, hide_index=True,
    )

    st.info(
        "Pakai sidebar untuk pindah ke halaman lain: "
        "lihat data + sentimen, feature engineering, prediksi per model, "
        "tabel metrik lengkap, dan kesimpulan akhir."
    )



# Page: Data & Sentiment

def page_data_sentiment():
    st.title("📊 Data Historis & Sentimen Twitter")

    btc = load_btc()
    sentiment = load_sentiment()
    tweets = load_tweet_sample()

    tab_btc, tab_sent, tab_tweets = st.tabs(["BTC Historical", "Daily Sentiment", "Sample Tweets"])

    # --- BTC tab ---
    with tab_btc:
        st.subheader("Bitcoin daily OHLC-derived (CoinGecko)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(btc):,}")
        c2.metric("Min close", f"${btc['close'].min():,.0f}")
        c3.metric("Max close", f"${btc['close'].max():,.0f}")

        col = st.selectbox("Kolom yang ingin di-plot", ["close", "market_cap", "volume"], index=0)
        fig = px.line(btc, x="date", y=col, title=f"BTC {col} (2021-2024)")
        fig.update_traces(line=dict(color="#f7931a", width=1.5))
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Preview data"):
            st.dataframe(btc.tail(50), use_container_width=True, hide_index=True)

    # --- Sentiment tab ---
    with tab_sent:
        st.subheader("Daily aggregated sentiment (VADER)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total hari", f"{len(sentiment):,}")
        c2.metric("Hari ada tweet", f"{int((sentiment['n_tweets']>0).sum()):,}")
        c3.metric("Total tweets", f"{int(sentiment['n_tweets'].sum()):,}")

        st.caption(
            "Hari tanpa tweet di-set ke 0 (netral / no-signal). "
            "Compound score VADER: −1 (negatif) → +1 (positif)."
        )

        # Smooth sentiment for visibility
        smooth = sentiment.copy()
        smooth["sentiment_mean_30d"] = smooth["sentiment_mean"].rolling(30, min_periods=5).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=smooth["date"], y=smooth["sentiment_mean"],
                                 mode="markers", name="daily mean",
                                 marker=dict(size=3, color="lightgray"), opacity=0.6))
        fig.add_trace(go.Scatter(x=smooth["date"], y=smooth["sentiment_mean_30d"],
                                 mode="lines", name="30-day MA", line=dict(color="#d62728", width=2)))
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10),
                          yaxis_title="Compound score", xaxis_title="Tanggal", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # Distribusi tweet per tahun
        sentiment["year"] = sentiment["date"].dt.year
        per_year = sentiment.groupby("year")["n_tweets"].sum().reset_index()
        fig2 = px.bar(per_year, x="year", y="n_tweets", title="Total tweet BTC per tahun",
                      color="n_tweets", color_continuous_scale="Oranges")
        fig2.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    # --- Tweets tab ---
    with tab_tweets:
        if tweets.empty:
            st.warning("Sample tweet tidak tersedia (jalankan precompute.py untuk generate).")
            return
        st.subheader(f"Random sample {len(tweets):,} tweets BTC")
        f1, f2 = st.columns(2)
        with f1:
            year_pick = st.multiselect("Filter tahun", sorted(tweets["date"].dt.year.unique()),
                                        default=sorted(tweets["date"].dt.year.unique()))
        with f2:
            sent_filter = st.select_slider("Sentimen (compound)", options=["semua", "negatif", "netral", "positif"], value="semua")
        view = tweets[tweets["date"].dt.year.isin(year_pick)].copy()
        if sent_filter == "negatif":
            view = view[view["compound"] <= -0.05]
        elif sent_filter == "positif":
            view = view[view["compound"] >= 0.05]
        elif sent_filter == "netral":
            view = view[view["compound"].between(-0.05, 0.05)]
        st.dataframe(view.head(500), use_container_width=True, hide_index=True)

# Page: Feature Engineering

def page_features():
    st.title("⚙️ Feature Engineering")

    df = load_features()

    scenario_features = load_feature_cols()

    scenario = st.selectbox(
        "Scenario",
        ["Historical", "Sentiment", "Combined"]
    )

    fcols = scenario_features[scenario]

    st.markdown(
        f"Scenario **{scenario}** menggunakan "
        f"**{len(fcols)} fitur**."
    )

    st.code("\n".join(fcols))

    st.subheader("Korelasi fitur dengan target t+1 / t+30")

    target_pick = st.radio(
        "Target",
        ["target_t1", "target_t30"],
        horizontal=True
    )

    corr = (
        df[fcols + [target_pick]]
        .corr()[target_pick]
        .drop(target_pick)
        .sort_values(key=abs, ascending=False)
    )

    fig = px.bar(
        corr.reset_index().rename(
            columns={
                "index": "feature",
                target_pick: "corr"
            }
        ),
        x="corr",
        y="feature",
        orientation="h",
        color="corr",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        title=f"Pearson Correlation vs {target_pick}"
    )

    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    with st.expander("Preview feature dataframe"):
        st.dataframe(
            df.head(100),
            use_container_width=True,
            hide_index=True
        )

# Page: Predictions

def page_predictions():

    st.title("🤖 Model Predictions — Actual vs Predicted")

    preds = load_predictions()
    metrics = load_metrics()

    all_models = sorted(metrics["model"].unique())
    pred_models = sorted(preds["model"].unique())

    metrics_only_models = [
        m for m in all_models
        if m not in pred_models
    ]

    if metrics_only_models:
        st.info(
            f"ℹ️ Model **{', '.join(metrics_only_models)}** ada di tabel metrik tapi prediksi-nya "
            "belum di-precompute (butuh TensorFlow)."
        )

    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])

    scenario = c1.selectbox(
        "Scenario",
        ["Historical", "Sentiment", "Combined"]
    )

    horizon = c2.selectbox(
        "Horizon",
        ["t+1", "t+30"],
        index=0
    )

    split = c3.selectbox(
        "Split",
        ["test", "validation"],
        index=0
    )

    models_pick = c4.multiselect(
        "Model",
        pred_models,
        default=pred_models
    )

    if not models_pick:
        st.warning("Pilih minimal satu model.")
        return

    sub = preds[
        (preds["scenario"] == scenario)
        &
        (preds["horizon"] == horizon)
        &
        (preds["split"] == split)
        &
        (preds["model"].isin(models_pick))
    ].sort_values("date")

    if sub.empty:
        st.warning("Tidak ada data untuk filter ini.")
        return

    actual = (
        sub[sub["model"] == models_pick[0]]
        [["date", "y_true"]]
        .rename(columns={"y_true": "actual"})
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=actual["date"],
            y=actual["actual"],
            mode="lines",
            name="Actual",
            line=dict(
                color=ACTUAL_COLOR,
                width=2
            )
        )
    )

    for m in models_pick:

        sm = sub[sub["model"] == m]

        fig.add_trace(
            go.Scatter(
                x=sm["date"],
                y=sm["y_pred"],
                mode="lines",
                name=m,
                line=dict(
                    color=MODEL_COLORS.get(m, "#999"),
                    width=1.5
                ),
                opacity=0.85
            )
        )

    fig.update_layout(
        title=(
            f"{scenario} | {horizon} | {split.title()} "
            f"({sub['date'].min():%Y-%m-%d} → "
            f"{sub['date'].max():%Y-%m-%d})"
        ),
        xaxis_title="Tanggal",
        yaxis_title="BTC Close (USD)",
        hovermode="x unified",
        height=520,
        margin=dict(
            l=10,
            r=10,
            t=50,
            b=10
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    err_rows = []

    for m in models_pick:

        sm = sub[sub["model"] == m].copy()

        sm["error"] = sm["y_pred"] - sm["y_true"]
        sm["abs_error"] = sm["error"].abs()

        err_rows.append(
            sm[
                [
                    "date",
                    "model",
                    "error",
                    "abs_error"
                ]
            ]
        )

    err_df = pd.concat(err_rows)

    st.subheader("Error per tanggal")

    fig2 = px.line(
        err_df,
        x="date",
        y="error",
        color="model",
        color_discrete_map=MODEL_COLORS,
        title="Prediction Error (Predicted - Actual)"
    )

    fig2.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray"
    )

    fig2.update_layout(
        height=360,
        margin=dict(
            l=10,
            r=10,
            t=40,
            b=10
        )
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.subheader("Ringkasan Error")

    summary = (
        err_df
        .groupby("model")["abs_error"]
        .agg(
            ["mean", "median", "max"]
        )
        .round(2)
        .reset_index()
    )

    summary.columns = [
        "Model",
        "MAE",
        "Median |Error|",
        "Max |Error|"
    ]

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

# Page: Metrics

def page_metrics():

    st.title("📈 Metrics & Model Comparison")

    metrics = load_metrics()

    scenario = st.selectbox(
        "Scenario",
        ["Historical", "Sentiment", "Combined"]
    )

    metrics = metrics[
        metrics["scenario"] == scenario
    ]

    st.subheader("Tabel Lengkap (RMSE / MAE / MAPE)")

    st.dataframe(
        metrics.style.format(
            {
                "RMSE": "{:.2f}",
                "MAE": "{:.2f}",
                "MAPE": "{:.2f}%"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Visual Perbandingan")

    metric = st.radio(
        "Metric",
        ["RMSE", "MAE", "MAPE"],
        horizontal=True
    )

    c1, c2 = st.columns(2)

    for i, h in enumerate(["t+1", "t+30"]):

        sub = metrics[
            metrics["horizon"] == h
        ]

        fig = px.bar(
            sub,
            x="model",
            y=metric,
            color="split",
            barmode="group",
            color_discrete_map={
                "validation": "#1f77b4",
                "test": "#d62728"
            },
            title=f"{scenario} | Horizon {h}"
        )

        fig.update_layout(
            height=360,
            margin=dict(
                l=10,
                r=10,
                t=50,
                b=10
            ),
            yaxis_title=metric
        )

        if i == 0:
            c1.plotly_chart(
                fig,
                use_container_width=True
            )
        else:
            c2.plotly_chart(
                fig,
                use_container_width=True
            )

    st.divider()

    st.subheader(
        "Best Model per Horizon (Test RMSE)"
    )

    best = (
    metrics[
        metrics["split"] == "test"
    ]
    .sort_values(
        ["horizon", "RMSE"]
    )
    .groupby(
        ["horizon"]
    )
    .head(1)
    .reset_index(drop=True)
    )

    st.dataframe(
        best[
            [
                "scenario",
                "horizon",
                "model",
                "RMSE",
                "MAE",
                "MAPE"
            ]
        ].style.format(
            {
                "RMSE": "{:.2f}",
                "MAE": "{:.2f}",
                "MAPE": "{:.2f}%"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

# Page: Scenario Comparison

def page_scenario_comparison():

    st.title("📊 Scenario Comparison")

    metrics = load_metrics()

    metric = st.selectbox(
        "Metric",
        ["RMSE", "MAE", "MAPE"]
    )

    split = st.selectbox(
        "Dataset",
        ["test", "validation"],
        index=0
    )

    sub = metrics[
        metrics["split"] == split
    ]

    st.subheader(
        f"Perbandingan {metric} Antar Skenario"
    )

    fig = px.bar(
        sub,
        x="scenario",
        y=metric,
        color="model",
        facet_col="horizon",
        barmode="group",
        title=f"{metric} Comparison"
    )

    fig.update_layout(
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Heatmap Performa"
    )

    heat = (
        sub
        .pivot_table(
            index="scenario",
            columns=["horizon", "model"],
            values=metric
        )
        .round(2)
    )

    st.dataframe(
        heat,
        use_container_width=True
    )

    st.subheader(
        "Best Result"
    )

    best = (
        sub
        .sort_values(metric)
        .head(10)
    )

    best_row = sub.sort_values(metric).iloc[0]

    st.success(
    f"Best {metric}: "
    f"{best_row['scenario']} | "
    f"{best_row['model']} | "
    f"{best_row['horizon']} "
    f"({best_row[metric]:.2f})"
)

    st.dataframe(
        best,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Halaman ini digunakan untuk membandingkan kontribusi "
        "data Historical, Sentiment, dan Combined terhadap "
        "akurasi prediksi Bitcoin."
    )

# Page: Conclusion

def page_conclusion():

    st.title("✅ Kesimpulan")

    st.markdown("""
### Kesimpulan

Berdasarkan hasil pengujian pada tiga skenario data, yaitu Historical, Sentiment, dan Combined, diperoleh beberapa temuan utama.

1. Data historis memberikan performa prediksi terbaik dibandingkan data sentimen saja pada seluruh model dan horizon prediksi.

2. Data sentimen saja belum mampu menghasilkan prediksi harga Bitcoin yang akurat. Hal ini ditunjukkan oleh nilai RMSE, MAE, dan MAPE yang jauh lebih tinggi dibandingkan skenario lainnya.

3. Kombinasi data historis dan sentimen memberikan performa yang kompetitif, namun peningkatannya tidak selalu signifikan dibandingkan penggunaan data historis saja.

4. Linear Regression menjadi model dengan performa terbaik pada sebagian besar skenario dan horizon prediksi berdasarkan nilai RMSE terendah pada data pengujian.

5. Hasil penelitian menunjukkan bahwa informasi historis harga Bitcoin masih menjadi faktor yang paling dominan dalam proses prediksi harga Bitcoin dibandingkan sentimen pasar yang digunakan pada penelitian ini.

### Implikasi Penelitian

Penelitian ini menunjukkan bahwa penggunaan data historis masih menjadi pendekatan yang efektif untuk prediksi harga Bitcoin. Penambahan data sentimen dapat digunakan sebagai fitur pendukung, namun pada penelitian ini belum mampu meningkatkan performa model secara signifikan.

### Pengembangan Selanjutnya

- Menambahkan indikator makroekonomi seperti DXY, CPI, Fear & Greed Index, dan S&P500.
- Menggunakan metode walk-forward validation.
- Menguji model deep learning yang lebih kompleks.
- Menggunakan data sentimen real-time dari berbagai platform media sosial.
""")

    st.info(
        "Seluruh eksperimen menggunakan split time-series tanpa shuffle dan "
        "StandardScaler yang hanya di-fit pada data training untuk menghindari data leakage."
    )

# Router

ROUTES = {
    " Overview": page_overview,
    " Data & Sentiment": page_data_sentiment,
    " Feature Engineering": page_features,
    " Model Predictions": page_predictions,
    " Scenario Comparison": page_scenario_comparison,
    " Metrics & Comparison": page_metrics,
    " Kesimpulan": page_conclusion,
}

try:
    ROUTES[page]()
except FileNotFoundError as e:
    st.error(
        "Artifacts belum di-generate. Jalankan `python precompute.py` dulu, "
        "lalu re-run streamlit.\n\nDetail: " + str(e)
    )
