import json
import os
import re
import zipfile
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except Exception:
    KaggleApi = None


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SAAD Dealer | Automotive Intelligence",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

APP_TITLE = "SAAD Dealer"
APP_SUBTITLE = "Automotive Sales Intelligence Command Center"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

DATASETS = {
    "vehicle_sales": {
        "label": "Vehicle Sales Data",
        "slug": "syedanwarafridi/vehicle-sales-data",
        "business_area": "Vehicle Market Pricing",
    },
    "car_sales": {
        "label": "Car Sales Data",
        "slug": "suraj520/car-sales-data",
        "business_area": "Dealer Sales Performance",
    },
    "automobile_sales": {
        "label": "Automobile Sales Data",
        "slug": "ddosad/auto-sales-data",
        "business_area": "Automobile Order Analytics",
    },
}


# ============================================================
# THEME CSS
# ============================================================

st.markdown(
    textwrap.dedent("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 45% 20%, rgba(46, 113, 255, 0.16), transparent 30%),
                radial-gradient(circle at 78% 18%, rgba(126, 231, 135, 0.10), transparent 25%),
                radial-gradient(circle at 20% 85%, rgba(255, 176, 59, 0.10), transparent 30%),
                linear-gradient(135deg, #070a12 0%, #0a1020 48%, #05070d 100%);
            color: #e5eefb;
        }

        .main .block-container {
            padding-top: 1.1rem;
            padding-left: 1.6rem;
            padding-right: 1.6rem;
            padding-bottom: 2rem;
            max-width: 1480px;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(11, 15, 30, 0.98), rgba(5, 8, 17, 0.98));
            border-right: 1px solid rgba(148, 163, 184, 0.16);
        }

        section[data-testid="stSidebar"] * {
            color: #d8e6ff;
        }

        div[data-testid="stMetric"] {
            background:
                linear-gradient(160deg, rgba(17, 25, 45, 0.82), rgba(12, 18, 33, 0.56));
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 20px;
            padding: 18px 18px;
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.30);
            backdrop-filter: blur(12px);
        }

        div[data-testid="stMetric"] label {
            color: #94a3b8 !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #f8fafc !important;
            font-size: 1.55rem !important;
            font-weight: 800 !important;
        }

        div[data-testid="stMetricDelta"] {
            color: #91f28d !important;
        }

        .top-shell {
            display: grid;
            grid-template-columns: 1.05fr 2fr 1.05fr;
            gap: 18px;
            align-items: stretch;
            margin-bottom: 22px;
        }

        .glass-card {
            background:
                linear-gradient(160deg, rgba(17, 25, 45, 0.86), rgba(9, 14, 27, 0.64));
            border: 1px solid rgba(148, 163, 184, 0.18);
            box-shadow: 0 22px 65px rgba(0, 0, 0, 0.36);
            border-radius: 24px;
            padding: 20px;
            position: relative;
            overflow: hidden;
        }

        .glass-card::before {
            content: "";
            position: absolute;
            inset: -1px;
            background:
                radial-gradient(circle at 12% 0%, rgba(90, 169, 255, 0.18), transparent 28%),
                radial-gradient(circle at 90% 10%, rgba(144, 244, 135, 0.10), transparent 24%);
            pointer-events: none;
        }

        .glass-card > * {
            position: relative;
            z-index: 1;
        }

        .hero-card {
            min-height: 520px;
            display: flex;
            align-items: center;
            justify-content: center;
            background:
                radial-gradient(circle at center, rgba(73, 145, 255, 0.32), transparent 34%),
                radial-gradient(circle at center, rgba(127, 231, 255, 0.12), transparent 54%),
                linear-gradient(160deg, rgba(7, 15, 29, 0.88), rgba(5, 8, 17, 0.70));
        }

        .hero-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 18px;
        }

        .brand-title {
            font-size: 1.35rem;
            font-weight: 800;
            color: #f8fafc;
            letter-spacing: -0.03em;
        }

        .breadcrumb {
            color: #93a4bd;
            font-size: 0.78rem;
            margin-top: 4px;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            border: 1px solid rgba(126, 231, 135, 0.35);
            background: rgba(126, 231, 135, 0.10);
            color: #b8ffb1;
            border-radius: 999px;
            padding: 7px 11px;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .orb-wrap {
            width: 430px;
            height: 430px;
            max-width: 100%;
            border-radius: 50%;
            position: relative;
            background:
                radial-gradient(circle at 33% 28%, rgba(199, 230, 255, 0.88), rgba(73, 145, 255, 0.44) 18%, rgba(19, 55, 92, 0.55) 42%, rgba(5, 12, 25, 0.96) 72%),
                radial-gradient(circle at 52% 52%, transparent 0 42%, rgba(126, 231, 255, 0.12) 43%, transparent 44%),
                linear-gradient(135deg, rgba(29, 78, 216, 0.44), rgba(15, 23, 42, 0.90));
            box-shadow:
                inset -36px -30px 80px rgba(0, 0, 0, 0.72),
                inset 28px 20px 80px rgba(186, 230, 253, 0.18),
                0 0 120px rgba(56, 189, 248, 0.22),
                0 0 12px rgba(255, 255, 255, 0.40);
            overflow: hidden;
        }

        .orb-wrap::before {
            content: "";
            position: absolute;
            inset: 16px;
            border-radius: 50%;
            background:
                repeating-radial-gradient(ellipse at center, rgba(147, 197, 253, 0.16) 0 1px, transparent 1px 24px),
                repeating-linear-gradient(32deg, rgba(147, 197, 253, 0.12) 0 1px, transparent 1px 28px),
                repeating-linear-gradient(-18deg, rgba(147, 197, 253, 0.08) 0 1px, transparent 1px 36px);
            opacity: 0.82;
            mix-blend-mode: screen;
        }

        .orb-wrap::after {
            content: "";
            position: absolute;
            width: 520px;
            height: 180px;
            border: 1px solid rgba(125, 211, 252, 0.30);
            border-radius: 50%;
            left: -46px;
            top: 120px;
            transform: rotate(-18deg);
            box-shadow:
                0 0 36px rgba(125, 211, 252, 0.22),
                inset 0 0 25px rgba(125, 211, 252, 0.10);
        }

        .orb-dot {
            position: absolute;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: #f8fafc;
            box-shadow: 0 0 22px rgba(255,255,255,0.95), 0 0 45px rgba(96,165,250,0.7);
            left: 49%;
            top: 47%;
            z-index: 3;
        }

        .signal-ring {
            position: absolute;
            inset: 190px;
            border: 1px dashed rgba(248, 250, 252, 0.75);
            border-radius: 50%;
            box-shadow: 0 0 32px rgba(248, 250, 252, 0.22);
            z-index: 2;
        }

        .mini-stat {
            display: flex;
            gap: 12px;
            align-items: center;
            margin-bottom: 18px;
        }

        .mini-icon {
            width: 36px;
            height: 36px;
            border-radius: 12px;
            display: grid;
            place-items: center;
            font-size: 1rem;
            font-weight: 800;
        }

        .blue-icon { background: linear-gradient(135deg, #2563eb, #60a5fa); }
        .green-icon { background: linear-gradient(135deg, #4ade80, #bef264); color: #052e16; }
        .orange-icon { background: linear-gradient(135deg, #f59e0b, #f97316); color: #111827; }

        .mini-label {
            font-size: 0.78rem;
            color: #94a3b8;
            margin-bottom: 3px;
        }

        .mini-value {
            font-size: 1.05rem;
            color: #f8fafc;
            font-weight: 800;
        }

        .panel-title {
            color: #f8fafc;
            font-weight: 800;
            font-size: 0.95rem;
            margin-bottom: 4px;
        }

        .panel-subtitle {
            color: #94a3b8;
            font-size: 0.76rem;
            margin-bottom: 16px;
        }

        .large-number {
            color: #ffffff;
            font-size: 2.3rem;
            font-weight: 800;
            letter-spacing: -0.05em;
            line-height: 1;
        }

        .delta-up {
            color: #9af58a;
            font-size: 0.78rem;
            font-weight: 700;
            margin-top: 8px;
        }

        .dashboard-section-title {
            color: #f8fafc;
            font-size: 1.05rem;
            font-weight: 800;
            margin: 20px 0 8px 0;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 999px;
            color: #cbd5e1;
            padding: 8px 16px;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: white;
            border-color: rgba(96, 165, 250, 0.60);
        }

        div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(148, 163, 184, 0.18);
        }

        .small-note {
            color: #94a3b8;
            font-size: 0.78rem;
            line-height: 1.45;
        }

        @media (max-width: 1100px) {
            .top-shell {
                grid-template-columns: 1fr;
            }

            .hero-card {
                min-height: 430px;
            }

            .orb-wrap {
                width: 320px;
                height: 320px;
            }
        }
    </style>
    """),
    unsafe_allow_html=True,
)


# ============================================================
# DATA UTILITIES
# ============================================================

def normalize_column_name(col: str) -> str:
    col = str(col).strip().lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    col = re.sub(r"_+", "_", col)
    return col.strip("_")


def clean_numeric_series(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace("", np.nan)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def clean_money_series(series: pd.Series) -> pd.Series:
    return clean_numeric_series(series)


def safe_to_datetime(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(series, errors="coerce")


def find_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    normalized_columns = [normalize_column_name(c) for c in columns]
    col_map = dict(zip(normalized_columns, columns))
    normalized_candidates = [normalize_column_name(c) for c in candidates]

    for candidate in normalized_candidates:
        if candidate in col_map:
            return col_map[candidate]

    for candidate in normalized_candidates:
        for norm_col, original_col in col_map.items():
            if candidate in norm_col or norm_col in candidate:
                return original_col

    return None


def format_currency(value: float) -> str:
    if pd.isna(value):
        return "-"

    value = float(value)

    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.2f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.1f}K"

    return f"${value:,.0f}"


def format_number(value: float) -> str:
    if pd.isna(value):
        return "-"

    value = float(value)

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.1f}K"

    return f"{value:,.0f}"


def safe_group_sum(df: pd.DataFrame, group_col: str, value_col: str, top_n: int = 10) -> pd.DataFrame:
    if group_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame(columns=[group_col, value_col])

    return (
        df.dropna(subset=[group_col])
        .groupby(group_col, as_index=False)[value_col]
        .sum()
        .sort_values(value_col, ascending=False)
        .head(top_n)
    )


def safe_group_mean(df: pd.DataFrame, group_col: str, value_col: str, top_n: int = 10) -> pd.DataFrame:
    if group_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame(columns=[group_col, value_col])

    return (
        df.dropna(subset=[group_col])
        .groupby(group_col, as_index=False)[value_col]
        .mean()
        .sort_values(value_col, ascending=False)
        .head(top_n)
    )


def apply_dark_layout(fig, height: int = 430):
    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbeafe", family="Inter"),
        title=dict(font=dict(size=17, color="#f8fafc")),
        margin=dict(l=20, r=20, t=55, b=28),
        legend=dict(
            bgcolor="rgba(15,23,42,0.35)",
            bordercolor="rgba(148,163,184,0.18)",
            borderwidth=1,
        ),
    )

    fig.update_xaxes(
        gridcolor="rgba(148,163,184,0.11)",
        zerolinecolor="rgba(148,163,184,0.20)",
    )
    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.11)",
        zerolinecolor="rgba(148,163,184,0.20)",
    )

    return fig


# ============================================================
# KAGGLE API
# ============================================================

def _read_local_kaggle_json() -> Tuple[Optional[str], Optional[str]]:
    """Local-only fallback. These files are gitignored and must not be uploaded to GitHub."""
    candidate_paths = [
        BASE_DIR / "kaggle.json",
        BASE_DIR / ".streamlit" / "kaggle.json",
    ]

    for path in candidate_paths:
        if path.exists():
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
                return obj.get("username"), obj.get("key")
            except Exception:
                pass

    return None, None


def configure_kaggle_credentials() -> Tuple[Optional[str], Optional[str]]:
    """
    Credential priority:
    1. Streamlit Cloud Secrets: [kaggle] username/key
    2. Flat Streamlit Secrets: KAGGLE_USERNAME/KAGGLE_KEY
    3. Environment variables
    4. Local-only kaggle.json for local testing
    """
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    try:
        if "kaggle" in st.secrets:
            kaggle_secret = st.secrets.get("kaggle", {})
            username = kaggle_secret.get("username", username)
            key = kaggle_secret.get("key", key)

        username = st.secrets.get("KAGGLE_USERNAME", username)
        key = st.secrets.get("KAGGLE_KEY", key)
    except Exception:
        pass

    if not username or not key:
        local_username, local_key = _read_local_kaggle_json()
        username = username or local_username
        key = key or local_key

    if username and key:
        os.environ["KAGGLE_USERNAME"] = str(username)
        os.environ["KAGGLE_KEY"] = str(key)

    return username, key


def has_kaggle_credentials() -> bool:
    username, key = configure_kaggle_credentials()
    return bool(username and key)


@st.cache_resource(show_spinner=False)
def get_kaggle_api():
    username, key = configure_kaggle_credentials()

    if KaggleApi is None:
        raise RuntimeError("Package 'kaggle' belum tersedia. Tambahkan 'kaggle' ke requirements.txt.")

    if not username or not key:
        raise RuntimeError("Kaggle credential belum ditemukan. Isi Streamlit Secrets dengan [kaggle] username dan key.")

    api = KaggleApi()
    api.authenticate()
    return api


def unzip_all_files(folder: Path) -> None:
    for zip_path in folder.glob("*.zip"):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(folder)


def dataset_folder_has_data(folder: Path) -> bool:
    extensions = ["*.csv", "*.xlsx", "*.xls", "*.json", "*.parquet"]
    return any(list(folder.glob(ext)) for ext in extensions)


def download_dataset(dataset_key: str) -> Path:
    dataset_info = DATASETS[dataset_key]
    slug = dataset_info["slug"]

    dataset_folder = RAW_DIR / dataset_key
    dataset_folder.mkdir(parents=True, exist_ok=True)

    if dataset_folder_has_data(dataset_folder):
        return dataset_folder

    api = get_kaggle_api()
    api.dataset_download_files(
        slug,
        path=str(dataset_folder),
        unzip=True,
        quiet=False,
    )

    unzip_all_files(dataset_folder)
    return dataset_folder


def list_data_files(folder: Path) -> List[Path]:
    files = []
    for pattern in ["*.csv", "*.xlsx", "*.xls", "*.parquet", "*.json"]:
        files.extend(folder.glob(pattern))
    return sorted(files)


def read_table_file(file_path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(file_path, nrows=max_rows, low_memory=False)

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file_path, nrows=max_rows)

    if suffix == ".parquet":
        df = pd.read_parquet(file_path)
        return df.head(max_rows) if max_rows else df

    if suffix == ".json":
        df = pd.read_json(file_path)
        return df.head(max_rows) if max_rows else df

    raise ValueError(f"Format file tidak didukung: {file_path.name}")


@st.cache_data(show_spinner=False)
def load_dataset_from_kaggle(dataset_key: str, max_rows: Optional[int]) -> pd.DataFrame:
    folder = download_dataset(dataset_key)
    files = list_data_files(folder)

    if not files:
        raise FileNotFoundError(f"Tidak ada file data untuk dataset: {DATASETS[dataset_key]['label']}")

    loaded_tables = []

    for file_path in files:
        try:
            df = read_table_file(file_path, max_rows=max_rows)
            if not df.empty:
                df["_source_file"] = file_path.name
                loaded_tables.append(df)
        except Exception:
            continue

    if not loaded_tables:
        raise RuntimeError(f"Dataset ditemukan, tetapi tidak ada file yang berhasil dibaca: {DATASETS[dataset_key]['label']}")

    loaded_tables = sorted(
        loaded_tables,
        key=lambda x: x.shape[0] * max(x.shape[1], 1),
        reverse=True,
    )

    return loaded_tables[0]


# ============================================================
# DEMO FALLBACK DATA
# ============================================================

@st.cache_data(show_spinner=False)
def generate_demo_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    brands = ["Toyota", "Ford", "BMW", "Mercedes-Benz", "Honda", "Nissan", "Chevrolet", "Audi", "Hyundai", "Kia"]
    regions = ["CA", "TX", "FL", "NY", "WA", "IL", "AZ", "NV", "GA", "London"]
    conditions = ["Excellent", "Good", "Fair", "Like New", "Reconditioned"]
    salespeople = ["A. Rahman", "N. Putri", "T. Malik", "S. Khan", "D. Morgan", "L. Chen"]
    product_lines = ["Classic Cars", "Vintage Cars", "Motorcycles", "Trucks and Buses", "Planes", "Ships"]

    n1, n2, n3 = 3500, 1800, 1600

    vehicle = pd.DataFrame({
        "source": "Vehicle Sales Data",
        "business_area": "Vehicle Market Pricing",
        "date": pd.date_range("2023-01-01", periods=n1, freq="D").to_series().sample(n1, replace=True, random_state=seed).values,
        "year": rng.integers(2008, 2025, n1),
        "brand": rng.choice(brands, n1),
        "model": rng.choice(["Sedan", "SUV", "Coupe", "Truck", "Hatchback", "Crossover"], n1),
        "region": rng.choice(regions, n1),
        "salesperson": "Unknown",
        "customer": "Unknown",
        "quantity": 1,
        "revenue": rng.normal(24500, 8200, n1).clip(2500, 78000),
        "unit_price": np.nan,
        "condition": rng.choice(conditions, n1),
        "odometer": rng.normal(65000, 35000, n1).clip(100, 220000),
        "mmr": np.nan,
        "deal_size": "Unknown",
        "transaction_count": 1,
    })
    vehicle["mmr"] = vehicle["revenue"] * rng.normal(0.88, 0.07, n1).clip(0.62, 1.08)
    vehicle["margin"] = vehicle["revenue"] - vehicle["mmr"]
    vehicle["unit_price"] = vehicle["revenue"]

    dealer = pd.DataFrame({
        "source": "Car Sales Data",
        "business_area": "Dealer Sales Performance",
        "date": pd.date_range("2023-01-01", periods=n2, freq="D").to_series().sample(n2, replace=True, random_state=seed+1).values,
        "year": rng.integers(2016, 2025, n2),
        "brand": rng.choice(brands, n2),
        "model": rng.choice(["Model A", "Model B", "Model C", "Model X", "Model Z"], n2),
        "region": rng.choice(regions, n2),
        "salesperson": rng.choice(salespeople, n2),
        "customer": [f"Customer {i:04d}" for i in range(n2)],
        "quantity": 1,
        "revenue": rng.normal(31200, 9800, n2).clip(5000, 92000),
        "unit_price": np.nan,
        "condition": "Unknown",
        "odometer": np.nan,
        "mmr": np.nan,
        "deal_size": "Unknown",
        "transaction_count": 1,
    })
    dealer["unit_price"] = dealer["revenue"]
    dealer["margin"] = np.nan

    orders = pd.DataFrame({
        "source": "Automobile Sales Data",
        "business_area": "Automobile Order Analytics",
        "date": pd.date_range("2022-01-01", periods=n3, freq="D").to_series().sample(n3, replace=True, random_state=seed+2).values,
        "year": rng.integers(2020, 2025, n3),
        "brand": rng.choice(product_lines, n3),
        "model": rng.choice(["S10", "S18", "A44", "GT7", "LX2"], n3),
        "region": rng.choice(["USA", "UK", "France", "Germany", "Japan", "Australia", "Singapore"], n3),
        "salesperson": "Unknown",
        "customer": [f"Corporate Client {i:04d}" for i in range(n3)],
        "quantity": rng.integers(1, 12, n3),
        "revenue": rng.normal(58000, 21000, n3).clip(8000, 170000),
        "unit_price": np.nan,
        "condition": "Unknown",
        "odometer": np.nan,
        "mmr": np.nan,
        "deal_size": rng.choice(["Small", "Medium", "Large"], n3, p=[0.44, 0.38, 0.18]),
        "transaction_count": 1,
    })
    orders["unit_price"] = orders["revenue"] / orders["quantity"]
    orders["margin"] = np.nan

    return pd.concat([vehicle, dealer, orders], ignore_index=True)


# ============================================================
# STANDARDIZATION
# ============================================================

def standardize_sales_data(df: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    original = df.copy()
    original.columns = [normalize_column_name(c) for c in original.columns]
    columns = list(original.columns)
    dataset_info = DATASETS[dataset_key]

    date_col = find_column(columns, ["saledate", "sale_date", "date", "orderdate", "order_date", "transaction_date", "sales_date"])
    year_col = find_column(columns, ["year", "car_year", "model_year", "vehicle_year"])
    brand_col = find_column(columns, ["make", "car_make", "manufacturer", "brand", "productline", "product_line"])
    model_col = find_column(columns, ["model", "car_model", "vehicle_model", "trim", "productcode", "product_code"])
    region_col = find_column(columns, ["state", "region", "territory", "country", "city", "postalcode", "postal_code", "dealer_region"])
    salesperson_col = find_column(columns, ["salesperson", "sales_person", "sales_rep", "sales_representative", "employee", "seller"])
    customer_col = find_column(columns, ["customer_name", "customername", "customer", "client", "buyer"])
    quantity_col = find_column(columns, ["quantityordered", "quantity_ordered", "quantity", "qty", "units", "unit_sold", "units_sold"])
    revenue_col = find_column(columns, ["sellingprice", "selling_price", "sale_price", "sales", "revenue", "total_sales", "price", "amount"])
    unit_price_col = find_column(columns, ["priceeach", "price_each", "unit_price", "price", "sellingprice", "selling_price", "sale_price"])
    condition_col = find_column(columns, ["condition", "vehicle_condition", "car_condition"])
    odometer_col = find_column(columns, ["odometer", "mileage", "miles", "kilometer", "kilometers"])
    mmr_col = find_column(columns, ["mmr", "market_value", "market_price"])
    deal_size_col = find_column(columns, ["dealsize", "deal_size", "deal", "segment"])

    std = pd.DataFrame(index=original.index)
    std["source"] = dataset_info["label"]
    std["business_area"] = dataset_info["business_area"]
    std["date"] = safe_to_datetime(original[date_col]) if date_col else pd.NaT
    std["year"] = clean_numeric_series(original[year_col]) if year_col else std["date"].dt.year
    std["brand"] = original[brand_col].astype(str).str.strip() if brand_col else "Unknown"
    std["model"] = original[model_col].astype(str).str.strip() if model_col else "Unknown"
    std["region"] = original[region_col].astype(str).str.strip() if region_col else "Unknown"
    std["salesperson"] = original[salesperson_col].astype(str).str.strip() if salesperson_col else "Unknown"
    std["customer"] = original[customer_col].astype(str).str.strip() if customer_col else "Unknown"
    std["quantity"] = clean_numeric_series(original[quantity_col]) if quantity_col else 1
    std["revenue"] = clean_money_series(original[revenue_col]) if revenue_col else np.nan
    std["unit_price"] = clean_money_series(original[unit_price_col]) if unit_price_col else std["revenue"] / std["quantity"].replace(0, np.nan)

    if std["revenue"].isna().all() and std["unit_price"].notna().any():
        std["revenue"] = std["unit_price"] * std["quantity"]

    std["condition"] = original[condition_col].astype(str).str.strip() if condition_col else "Unknown"
    std["odometer"] = clean_numeric_series(original[odometer_col]) if odometer_col else np.nan
    std["mmr"] = clean_money_series(original[mmr_col]) if mmr_col else np.nan
    std["margin"] = std["revenue"] - std["mmr"]
    std["deal_size"] = original[deal_size_col].astype(str).str.strip() if deal_size_col else "Unknown"
    std["transaction_count"] = 1

    for text_col in ["brand", "model", "region", "salesperson", "customer", "condition", "deal_size"]:
        std[text_col] = std[text_col].replace(["nan", "None", "NaN", ""], "Unknown").fillna("Unknown")

    std["quantity"] = std["quantity"].fillna(1)
    std["revenue"] = std["revenue"].replace([np.inf, -np.inf], np.nan)
    std["unit_price"] = std["unit_price"].replace([np.inf, -np.inf], np.nan)

    return std


@st.cache_data(show_spinner=False)
def load_all_standardized_data(
    selected_dataset_keys: Tuple[str, ...],
    max_rows_per_dataset: Optional[int],
    use_demo_data: bool,
) -> pd.DataFrame:
    if use_demo_data:
        return generate_demo_data()

    frames = []

    for dataset_key in selected_dataset_keys:
        raw_df = load_dataset_from_kaggle(dataset_key, max_rows=max_rows_per_dataset)
        std_df = standardize_sales_data(raw_df, dataset_key)
        frames.append(std_df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ============================================================
# SIDEBAR AND FILTERS
# ============================================================

def render_sidebar() -> Tuple[Tuple[str, ...], Optional[int], bool]:
    st.sidebar.markdown("## 🚘 SAAD Dealer")
    st.sidebar.caption("Automotive Intelligence")

    credentials_found = has_kaggle_credentials()
    use_demo_data = st.sidebar.toggle(
        "Use demo data",
        value=not credentials_found,
        help="Jika Kaggle Secrets belum terdeteksi, dashboard otomatis memakai demo data supaya aplikasi tetap terbuka.",
    )

    if credentials_found:
        st.sidebar.success("Kaggle credential detected.")
    else:
        st.sidebar.warning("Kaggle credential belum terdeteksi. Isi Streamlit Cloud Secrets untuk memakai data Kaggle asli.")

    st.sidebar.markdown("### Kaggle Datasets")

    dataset_labels = {key: value["label"] for key, value in DATASETS.items()}
    label_to_key = {value: key for key, value in dataset_labels.items()}

    selected_labels = st.sidebar.multiselect(
        "Select datasets",
        options=list(label_to_key.keys()),
        default=list(label_to_key.keys()),
        disabled=use_demo_data,
    )

    selected_keys = tuple(label_to_key[label] for label in selected_labels)

    st.sidebar.markdown("### Loading")
    row_limit_choice = st.sidebar.selectbox(
        "Rows per dataset",
        options=["10,000 rows", "50,000 rows", "100,000 rows", "200,000 rows", "Full dataset"],
        index=1,
        disabled=use_demo_data,
    )

    row_limit_map = {
        "10,000 rows": 10_000,
        "50,000 rows": 50_000,
        "100,000 rows": 100_000,
        "200,000 rows": 200_000,
        "Full dataset": None,
    }

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div class="small-note">
        Kaggle credential disimpan melalui Streamlit Secrets. Jangan upload <b>kaggle.json</b> ke GitHub.
        </div>
        """,
        unsafe_allow_html=True,
    )

    return selected_keys, row_limit_map[row_limit_choice], use_demo_data


def apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()

    st.sidebar.markdown("### Filters")

    source_options = sorted(filtered["source"].dropna().unique().tolist())
    selected_sources = st.sidebar.multiselect("Dataset Source", source_options, default=source_options)

    if selected_sources:
        filtered = filtered[filtered["source"].isin(selected_sources)]

    brand_options = sorted(filtered["brand"].dropna().unique().tolist())
    selected_brands = st.sidebar.multiselect("Brand / Product Line", brand_options, default=[])

    if selected_brands:
        filtered = filtered[filtered["brand"].isin(selected_brands)]

    region_options = sorted(filtered["region"].dropna().unique().tolist())
    selected_regions = st.sidebar.multiselect("Region", region_options, default=[])

    if selected_regions:
        filtered = filtered[filtered["region"].isin(selected_regions)]

    valid_dates = filtered["date"].dropna()

    if not valid_dates.empty:
        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()

        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            filtered = filtered[
                (filtered["date"].isna())
                | ((filtered["date"].dt.date >= start_date) & (filtered["date"].dt.date <= end_date))
            ]

    return filtered


# ============================================================
# TOP GLOBE COMMAND CENTER
# ============================================================

def render_top_command_center(df: pd.DataFrame) -> None:
    total_revenue = df["revenue"].sum(skipna=True)
    transactions = df["transaction_count"].sum(skipna=True)
    units = df["quantity"].sum(skipna=True)
    avg_price = df["revenue"].mean(skipna=True)
    avg_margin = df["margin"].mean(skipna=True)
    delivered_rate = min(99, max(10, 60 + int((df["revenue"].notna().mean()) * 30)))

    top_region = (
        df.groupby("region")["revenue"].sum().sort_values(ascending=False).index[0]
        if df["region"].notna().any()
        else "-"
    )

    top_brand = (
        df.groupby("brand")["revenue"].sum().sort_values(ascending=False).index[0]
        if df["brand"].notna().any()
        else "-"
    )

    st.markdown(
        textwrap.dedent(f"""
        <div class="hero-header">
            <div>
                <div class="brand-title">Departments</div>
                <div class="breadcrumb">Map / SAAD Dealer / Automotive Command Center</div>
            </div>
            <div class="status-pill">● Premium Analytics</div>
        </div>

        <div class="top-shell">
            <div class="glass-card">
                <div class="panel-title">Total Revenue</div>
                <div class="panel-subtitle">Selected automotive datasets</div>
                <div class="large-number">{format_currency(total_revenue)}</div>
                <div class="delta-up">▲ Integrated market, dealer, and order data</div>
                <br>
                <div class="mini-stat">
                    <div class="mini-icon blue-icon">↗</div>
                    <div>
                        <div class="mini-label">Transactions</div>
                        <div class="mini-value">{format_number(transactions)}</div>
                    </div>
                </div>
                <div class="mini-stat">
                    <div class="mini-icon green-icon">✓</div>
                    <div>
                        <div class="mini-label">Units / Quantity</div>
                        <div class="mini-value">{format_number(units)}</div>
                    </div>
                </div>
                <div class="mini-stat">
                    <div class="mini-icon orange-icon">$</div>
                    <div>
                        <div class="mini-label">Average Deal Value</div>
                        <div class="mini-value">{format_currency(avg_price)}</div>
                    </div>
                </div>
                <br>
                <div class="panel-title">Market Leader</div>
                <div class="large-number" style="font-size:1.55rem;">{top_brand}</div>
                <div class="panel-subtitle">Top region: {top_region}</div>
            </div>

            <div class="glass-card hero-card">
                <div class="orb-wrap">
                    <div class="orb-dot"></div>
                    <div class="signal-ring"></div>
                </div>
            </div>

            <div class="glass-card">
                <div class="panel-title">Operational Status</div>
                <div class="panel-subtitle">Sales data readiness</div>
                <div class="large-number">{delivered_rate}%</div>
                <div class="delta-up">▲ Data availability index</div>
                <br>
                <div class="panel-title">Average Margin</div>
                <div class="large-number" style="font-size:1.85rem;">{format_currency(avg_margin)}</div>
                <div class="panel-subtitle">Available mainly from vehicle market pricing</div>
                <br>
                <div class="mini-stat">
                    <div class="mini-icon blue-icon">3D</div>
                    <div>
                        <div class="mini-label">Dashboard Mode</div>
                        <div class="mini-value">Globe Intelligence</div>
                    </div>
                </div>
                <div class="mini-stat">
                    <div class="mini-icon green-icon">API</div>
                    <div>
                        <div class="mini-label">Data Source</div>
                        <div class="mini-value">Kaggle API</div>
                    </div>
                </div>
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )


# ============================================================
# DASHBOARD SECTIONS
# ============================================================

def render_kpis(df: pd.DataFrame) -> None:
    total_revenue = df["revenue"].sum(skipna=True)
    total_units = df["quantity"].sum(skipna=True)
    total_transactions = df["transaction_count"].sum(skipna=True)
    avg_revenue = df["revenue"].mean(skipna=True)
    unique_brands = df["brand"].replace("Unknown", np.nan).nunique()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Revenue", format_currency(total_revenue), "portfolio value")
    col2.metric("Units Sold", format_number(total_units), "quantity volume")
    col3.metric("Transactions", format_number(total_transactions), "records")
    col4.metric("Avg. Revenue", format_currency(avg_revenue), "per transaction")
    col5.metric("Brand Lines", format_number(unique_brands), "active lines")


def render_overview(df: pd.DataFrame) -> None:
    st.markdown('<div class="dashboard-section-title">Executive Overview</div>', unsafe_allow_html=True)
    render_kpis(df)

    col1, col2 = st.columns([1.35, 1])

    with col1:
        monthly = (
            df.dropna(subset=["date"])
            .assign(month=lambda x: x["date"].dt.to_period("M").dt.to_timestamp())
            .groupby("month", as_index=False)["revenue"]
            .sum()
            .sort_values("month")
        )

        if not monthly.empty:
            fig = px.area(
                monthly,
                x="month",
                y="revenue",
                title="Monthly Revenue Flow",
                labels={"month": "Month", "revenue": "Revenue"},
            )
            fig = apply_dark_layout(fig, 430)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Date column belum cukup untuk monthly trend.")

    with col2:
        source_revenue = safe_group_sum(df, "source", "revenue", 10)

        if not source_revenue.empty:
            fig = px.pie(
                source_revenue,
                names="source",
                values="revenue",
                hole=0.62,
                title="Revenue Mix by Dataset",
            )
            fig = apply_dark_layout(fig, 430)
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        brand_revenue = safe_group_sum(df, "brand", "revenue", 12)

        if not brand_revenue.empty:
            fig = px.bar(
                brand_revenue.sort_values("revenue"),
                x="revenue",
                y="brand",
                orientation="h",
                title="Top Brand / Product Line",
                labels={"brand": "Brand / Product Line", "revenue": "Revenue"},
            )
            fig = apply_dark_layout(fig, 480)
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        region_revenue = safe_group_sum(df, "region", "revenue", 12)

        if not region_revenue.empty:
            fig = px.bar(
                region_revenue.sort_values("revenue"),
                x="revenue",
                y="region",
                orientation="h",
                title="Top Region",
                labels={"region": "Region", "revenue": "Revenue"},
            )
            fig = apply_dark_layout(fig, 480)
            st.plotly_chart(fig, use_container_width=True)


def render_vehicle_market(df: pd.DataFrame) -> None:
    vehicle_df = df[df["source"].eq("Vehicle Sales Data")].copy()

    if vehicle_df.empty:
        st.warning("Vehicle Sales Data tidak tersedia pada filter saat ini.")
        return

    st.markdown('<div class="dashboard-section-title">Vehicle Market Pricing</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average Selling Price", format_currency(vehicle_df["revenue"].mean()))
    col2.metric("Average MMR", format_currency(vehicle_df["mmr"].mean()))
    col3.metric("Average Margin", format_currency(vehicle_df["margin"].mean()))
    col4.metric("Average Odometer", format_number(vehicle_df["odometer"].mean()))

    col5, col6 = st.columns(2)

    with col5:
        plot_df = vehicle_df.dropna(subset=["odometer", "revenue"])
        if not plot_df.empty:
            sample_df = plot_df.sample(min(len(plot_df), 7000), random_state=42)
            fig = px.scatter(
                sample_df,
                x="odometer",
                y="revenue",
                color="brand",
                hover_data=["model", "year", "condition", "region"],
                title="Odometer vs Selling Price",
                labels={"odometer": "Odometer", "revenue": "Selling Price"},
            )
            fig = apply_dark_layout(fig, 480)
            st.plotly_chart(fig, use_container_width=True)

    with col6:
        condition_price = safe_group_mean(vehicle_df, "condition", "revenue", 12)
        if not condition_price.empty:
            fig = px.bar(
                condition_price.sort_values("revenue", ascending=False),
                x="condition",
                y="revenue",
                title="Average Selling Price by Condition",
                labels={"condition": "Condition", "revenue": "Average Selling Price"},
            )
            fig = apply_dark_layout(fig, 480)
            st.plotly_chart(fig, use_container_width=True)

    col7, col8 = st.columns(2)

    with col7:
        margin_brand = safe_group_mean(vehicle_df.dropna(subset=["margin"]), "brand", "margin", 12)
        if not margin_brand.empty:
            fig = px.bar(
                margin_brand.sort_values("margin"),
                x="margin",
                y="brand",
                orientation="h",
                title="Average Margin by Brand",
            )
            fig = apply_dark_layout(fig, 460)
            st.plotly_chart(fig, use_container_width=True)

    with col8:
        year_price = (
            vehicle_df.dropna(subset=["year", "revenue"])
            .groupby("year", as_index=False)["revenue"]
            .mean()
            .sort_values("year")
        )
        if not year_price.empty:
            fig = px.line(
                year_price,
                x="year",
                y="revenue",
                markers=True,
                title="Average Selling Price by Vehicle Year",
            )
            fig = apply_dark_layout(fig, 460)
            st.plotly_chart(fig, use_container_width=True)


def render_dealer_sales(df: pd.DataFrame) -> None:
    dealer_df = df[df["source"].eq("Car Sales Data")].copy()

    if dealer_df.empty:
        st.warning("Car Sales Data tidak tersedia pada filter saat ini.")
        return

    st.markdown('<div class="dashboard-section-title">Dealer Sales Performance</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dealer Revenue", format_currency(dealer_df["revenue"].sum()))
    col2.metric("Dealer Units", format_number(dealer_df["quantity"].sum()))
    col3.metric("Average Deal", format_currency(dealer_df["revenue"].mean()))
    col4.metric("Salespeople", format_number(dealer_df["salesperson"].replace("Unknown", np.nan).nunique()))

    col5, col6 = st.columns(2)

    with col5:
        salesperson = safe_group_sum(dealer_df, "salesperson", "revenue", 12)
        if not salesperson.empty and not salesperson["salesperson"].eq("Unknown").all():
            fig = px.bar(
                salesperson.sort_values("revenue"),
                x="revenue",
                y="salesperson",
                orientation="h",
                title="Top Salesperson by Revenue",
            )
            fig = apply_dark_layout(fig, 500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Kolom salesperson tidak tersedia pada versi dataset saat ini.")

    with col6:
        model_revenue = safe_group_sum(dealer_df, "model", "revenue", 12)
        if not model_revenue.empty:
            fig = px.bar(
                model_revenue.sort_values("revenue"),
                x="revenue",
                y="model",
                orientation="h",
                title="Top Car Model by Revenue",
            )
            fig = apply_dark_layout(fig, 500)
            st.plotly_chart(fig, use_container_width=True)

    col7, col8 = st.columns(2)

    with col7:
        brand_unit = (
            dealer_df.dropna(subset=["brand"])
            .groupby("brand", as_index=False)["quantity"]
            .sum()
            .sort_values("quantity", ascending=False)
            .head(12)
        )
        if not brand_unit.empty:
            fig = px.bar(
                brand_unit.sort_values("quantity"),
                x="quantity",
                y="brand",
                orientation="h",
                title="Units by Brand",
            )
            fig = apply_dark_layout(fig, 460)
            st.plotly_chart(fig, use_container_width=True)

    with col8:
        year_sales = (
            dealer_df.dropna(subset=["year", "revenue"])
            .groupby("year", as_index=False)["revenue"]
            .sum()
            .sort_values("year")
        )
        if not year_sales.empty:
            fig = px.line(year_sales, x="year", y="revenue", markers=True, title="Dealer Revenue by Car Year")
            fig = apply_dark_layout(fig, 460)
            st.plotly_chart(fig, use_container_width=True)


def render_order_analytics(df: pd.DataFrame) -> None:
    order_df = df[df["source"].eq("Automobile Sales Data")].copy()

    if order_df.empty:
        st.warning("Automobile Sales Data tidak tersedia pada filter saat ini.")
        return

    st.markdown('<div class="dashboard-section-title">Automobile Order Analytics</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Order Revenue", format_currency(order_df["revenue"].sum()))
    col2.metric("Quantity Ordered", format_number(order_df["quantity"].sum()))
    col3.metric("Average Order", format_currency(order_df["revenue"].mean()))
    col4.metric("Customers", format_number(order_df["customer"].replace("Unknown", np.nan).nunique()))

    col5, col6 = st.columns(2)

    with col5:
        product_revenue = safe_group_sum(order_df, "brand", "revenue", 12)
        if not product_revenue.empty:
            fig = px.bar(
                product_revenue.sort_values("revenue"),
                x="revenue",
                y="brand",
                orientation="h",
                title="Revenue by Product Line",
            )
            fig = apply_dark_layout(fig, 500)
            st.plotly_chart(fig, use_container_width=True)

    with col6:
        deal_size = safe_group_sum(order_df, "deal_size", "revenue", 10)
        if not deal_size.empty and not deal_size["deal_size"].eq("Unknown").all():
            fig = px.pie(deal_size, names="deal_size", values="revenue", hole=0.58, title="Revenue Share by Deal Size")
            fig = apply_dark_layout(fig, 500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Kolom deal size tidak tersedia pada versi dataset saat ini.")

    col7, col8 = st.columns(2)

    with col7:
        region_revenue = safe_group_sum(order_df, "region", "revenue", 12)
        if not region_revenue.empty:
            fig = px.bar(
                region_revenue.sort_values("revenue"),
                x="revenue",
                y="region",
                orientation="h",
                title="Revenue by Region / Country",
            )
            fig = apply_dark_layout(fig, 460)
            st.plotly_chart(fig, use_container_width=True)

    with col8:
        monthly = (
            order_df.dropna(subset=["date"])
            .assign(month=lambda x: x["date"].dt.to_period("M").dt.to_timestamp())
            .groupby("month", as_index=False)["revenue"]
            .sum()
            .sort_values("month")
        )
        if not monthly.empty:
            fig = px.line(monthly, x="month", y="revenue", markers=True, title="Monthly Automobile Order Revenue")
            fig = apply_dark_layout(fig, 460)
            st.plotly_chart(fig, use_container_width=True)


def render_data_quality(df: pd.DataFrame) -> None:
    st.markdown('<div class="dashboard-section-title">Data Quality & Explorer</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", format_number(len(df)))
    col2.metric("Columns", format_number(df.shape[1]))
    col3.metric("Duplicates", format_number(df.duplicated().sum()))
    col4.metric("Missing Cells", format_number(df.isna().sum().sum()))

    missing_summary = (
        df.isna()
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_percentage"})
        .sort_values("missing_percentage", ascending=False)
    )

    col5, col6 = st.columns([1, 1.4])

    with col5:
        st.subheader("Missing Value Summary")
        st.dataframe(missing_summary, use_container_width=True, hide_index=True)

    with col6:
        st.subheader("Filtered Data Preview")
        st.dataframe(df.head(600), use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered CSV",
        data=csv,
        file_name="saad_dealer_standardized_sales_data.csv",
        mime="text/csv",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    selected_dataset_keys, max_rows_per_dataset, use_demo_data = render_sidebar()

    try:
        with st.spinner("Loading SAAD Dealer automotive intelligence data..."):
            if not selected_dataset_keys and not use_demo_data:
                st.warning("Pilih minimal satu dataset Kaggle di sidebar.")
                st.stop()

            df = load_all_standardized_data(
                selected_dataset_keys=selected_dataset_keys,
                max_rows_per_dataset=max_rows_per_dataset,
                use_demo_data=use_demo_data,
            )

    except Exception as error:
        st.error("Dashboard gagal memuat data Kaggle.")
        st.exception(error)
        st.markdown(
            """
            **Solusi cepat:**

            1. Buka Streamlit Cloud → Manage app → Settings → Secrets.
            2. Isi format TOML berikut:
            ```toml
            [kaggle]
            username = "username_kaggle_kamu"
            key = "api_key_kaggle_kamu"
            ```
            3. Klik Save, lalu Reboot app.
            4. Jangan upload `kaggle.json` atau `.streamlit/secrets.toml` ke GitHub.
            5. Untuk cek tampilan tanpa Kaggle API, aktifkan **Use demo data** di sidebar.
            """
        )
        st.stop()

    if df.empty:
        st.warning("Data kosong setelah proses loading.")
        st.stop()

    filtered_df = apply_sidebar_filters(df)

    if filtered_df.empty:
        st.warning("Tidak ada data yang sesuai dengan filter saat ini.")
        st.stop()

    render_top_command_center(filtered_df)

    tabs = st.tabs(
        [
            "Executive Overview",
            "Vehicle Market",
            "Dealer Sales",
            "Automobile Orders",
            "Data Quality",
        ]
    )

    with tabs[0]:
        render_overview(filtered_df)

    with tabs[1]:
        render_vehicle_market(filtered_df)

    with tabs[2]:
        render_dealer_sales(filtered_df)

    with tabs[3]:
        render_order_analytics(filtered_df)

    with tabs[4]:
        render_data_quality(filtered_df)


if __name__ == "__main__":
    main()
