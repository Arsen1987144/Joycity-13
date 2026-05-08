import os
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ----------------------------------------------------------------------------
# 1. CONFIG
# ----------------------------------------------------------------------------

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("wb-dashboard")

WB_API_TOKEN = os.getenv("WB_API_TOKEN", "")

# Base URLs for different WB APIs
WB_STATS_BASE   = "https://statistics-api.wildberries.ru"
WB_ADVERT_BASE  = "https://advert-api.wildberries.ru"
WB_ANALYTICS    = "https://seller-analytics-api.wildberries.ru"

REQUEST_TIMEOUT = 60
CACHE_TTL = 60 * 15  # 15 minutes


@dataclass
class Period:
    label: str
    days: int


PERIODS = {
    "День":     Period("День", 1),
    "7 дней":   Period("7 дней", 7),
    "30 дней":  Period("30 дней", 30),
}


# ----------------------------------------------------------------------------
# 2. WB API CLIENT
# ----------------------------------------------------------------------------

class WBClient:
    """Thin client for WB API with retries and unified authorization header."""

    def __init__(self, token: str):
        if not token:
            raise ValueError(
                "WB_API_TOKEN not set. Generate a token in WB seller portal → Profile → API Access."
            )
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": token,
            "Content-Type": "application/json",
        })

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type((requests.RequestException,)),
        reraise=True,
    )
    def _get(self, url: str, params: dict | None = None) -> list | dict:
        log.info("GET %s params=%s", url, params)
        r = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if r.status_code == 429:
            time.sleep(20)  # respect rate limits
            r.raise_for_status()
        r.raise_for_status()
        return r.json()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type((requests.RequestException,)),
        reraise=True,
    )
    def _post(self, url: str, json_body: dict) -> list | dict:
        log.info("POST %s body=%s", url, json_body)
        r = self.session.post(url, json=json_body, timeout=REQUEST_TIMEOUT)
        if r.status_code == 429:
            time.sleep(20)
            r.raise_for_status()
        r.raise_for_status()
        return r.json()

    # Sales (paid orders)
    def sales(self, date_from: datetime) -> list:
        url = f"{WB_STATS_BASE}/api/v1/supplier/sales"
        return self._get(url, {"dateFrom": date_from.strftime("%Y-%m-%dT%H:%M:%S")})

    # Orders (created, not yet purchased)
    def orders(self, date_from: datetime) -> list:
        url = f"{WB_STATS_BASE}/api/v1/supplier/orders"
        return self._get(url, {"dateFrom": date_from.strftime("%Y-%m-%dT%H:%M:%S")})

    # Stocks at warehouses
    def stocks(self, date_from: datetime) -> list:
        url = f"{WB_STATS_BASE}/api/v1/supplier/stocks"
        return self._get(url, {"dateFrom": date_from.strftime("%Y-%m-%dT%H:%M:%S")})

    # Detailed realisation report with commissions
    def report_detail(self, date_from: date, date_to: date) -> list:
        url = f"{WB_STATS_BASE}/api/v5/supplier/reportDetailByPeriod"
        return self._get(url, {
            "dateFrom": date_from.isoformat(),
            "dateTo":   date_to.isoformat(),
            "limit":    100000,
        })

    # Funnel for product cards (views → clicks → cart → order)
    def funnel(self, nm_ids: list[int], date_from: date, date_to: date) -> dict:
        url = f"{WB_ANALYTICS}/api/v2/nm-report/detail"
        body = {
            "nmIDs": nm_ids[:1000],  # API limit – 1000 nmID per call
            "period": {"begin": date_from.isoformat(), "end": date_to.isoformat()},
            "page": 1,
        }
        return self._post(url, body)

    # Active advertising campaigns
    def adverts(self, status: int = 9) -> list:
        url = f"{WB_ADVERT_BASE}/adv/v1/promotion/count"
        return self._get(url)

    # Advertising statistics
    def advert_stats(self, campaign_ids: list[int], date_from: date, date_to: date) -> list:
        url = f"{WB_ADVERT_BASE}/adv/v2/fullstats"
        body = [{"id": cid, "dates": [date_from.isoformat(), date_to.isoformat()]} for cid in campaign_ids]
        return self._post(url, body)


# ----------------------------------------------------------------------------
# 3. ETL: loading and normalisation
# ----------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL, show_spinner="Загружаю продажи WB…")
def load_sales(date_from: datetime) -> pd.DataFrame:
    client = WBClient(WB_API_TOKEN)
    raw = client.sales(date_from)
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["date"])
    df.rename(columns={
        "nmId": "nm_id",
        "supplierArticle": "art",
        "subject": "name",
        "category": "cat",
        "priceWithDisc": "revenue",  # price with discount = actual revenue per line
        "warehouseName": "warehouse",
    }, inplace=True)
    return df


@st.cache_data(ttl=CACHE_TTL, show_spinner="Загружаю заказы…")
def load_orders(date_from: datetime) -> pd.DataFrame:
    client = WBClient(WB_API_TOKEN)
    raw = client.orders(date_from)
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["date"])
    df.rename(columns={"nmId": "nm_id", "supplierArticle": "art", "isCancel": "is_cancel"}, inplace=True)
    return df


@st.cache_data(ttl=CACHE_TTL, show_spinner="Загружаю остатки…")
def load_stocks(date_from: datetime) -> pd.DataFrame:
    client = WBClient(WB_API_TOKEN)
    raw = client.stocks(date_from)
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw)
    df.rename(columns={
        "nmId": "nm_id",
        "supplierArticle": "art",
        "warehouseName": "warehouse",
        "quantity": "qty",
        "subject": "name",
        "category": "cat",
    }, inplace=True)
    return df


@st.cache_data(ttl=CACHE_TTL, show_spinner="Загружаю отчёт по реализации (комиссии)…")
def load_report(date_from: date, date_to: date) -> pd.DataFrame:
    client = WBClient(WB_API_TOKEN)
    raw = client.report_detail(date_from, date_to)
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw)
    df.rename(columns={
        "nm_id": "nm_id",
        "sa_name": "art",
        "ppvz_for_pay": "to_pay",
        "delivery_rub": "logistics",
        "commission_percent": "commission_pct",
        "retail_amount": "retail",
        "subject_name": "name",
    }, inplace=True)
    return df


@st.cache_data(ttl=CACHE_TTL, show_spinner="Загружаю воронку по карточкам…")
def load_funnel(nm_ids: list[int], date_from: date, date_to: date) -> pd.DataFrame:
    if not nm_ids:
        return pd.DataFrame()
    client = WBClient(WB_API_TOKEN)
    raw = client.funnel(nm_ids, date_from, date_to)
    cards = (raw or {}).get("data", {}).get("cards", [])
    rows = []
    for c in cards:
        st_block = (c.get("statistics") or [{}])[0]
        sel = st_block.get("selectedPeriod", {})
        conv = sel.get("conversions", {})
        rows.append({
            "nm_id":     c.get("nmID"),
            "shows":     sel.get("openCardCount", 0),
            "clicks":    sel.get("openCardCount", 0),  # for compatibility
            "add_cart":  sel.get("addToCartCount", 0),
            "orders":    sel.get("ordersCount", 0),
            "buyouts":   sel.get("buyoutsCount", 0),
            "revenue":   sel.get("ordersSumRub", 0),
            "buyout_sum": sel.get("buyoutsSumRub", 0),
            "c2cart":    conv.get("addToCartPercent", 0),
            "c2order":   conv.get("cartToOrderPercent", 0),
            "buyout":    conv.get("buyoutsPercent", 0),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 4. KPI calculation
# ----------------------------------------------------------------------------

def calc_kpis(funnel_df: pd.DataFrame, ads_total: float, cogs_total: float) -> dict:
    """Calculate top-level KPIs for selected period."""
    if funnel_df.empty:
        return {}
    revenue = funnel_df["revenue"].sum()
    orders  = funnel_df["orders"].sum()
    buyouts = funnel_df["buyouts"].sum()
    add_cart = funnel_df["add_cart"].sum()
    shows   = funnel_df["shows"].sum()

    ctr      = (funnel_df["clicks"].sum() / shows * 100) if shows else 0
    c2cart   = (add_cart / shows * 100) if shows else 0
    c2order  = (orders / add_cart * 100) if add_cart else 0
    buyout   = (buyouts / orders * 100) if orders else 0
    margin_abs = revenue - cogs_total - ads_total
    margin_pct = (margin_abs / revenue * 100) if revenue else 0

    return {
        "revenue": revenue,
        "orders":  orders,
        "ads":     ads_total,
        "ctr":     ctr,
        "c2cart":  c2cart,
        "c2order": c2order,
        "buyout":  buyout,
        "margin_abs": margin_abs,
        "margin_pct": margin_pct,
    }


def calc_delta(curr: dict, prev: dict, key: str) -> float:
    """Return % delta. If no base for comparison, returns 0."""
    if not prev or not prev.get(key):
        return 0.0
    return (curr.get(key, 0) - prev[key]) / prev[key] * 100


def detect_anomalies(skus: pd.DataFrame) -> list[dict]:
    """Simple rules + Z-score on revenue for anomaly detection."""
    insights = []
    if skus.empty:
        return insights

    # Rule 1: DRR > 25% and CTR < 2%
    bad_ad = skus[(skus["drr"] > 25) & (skus["ctr"] < 2)]
    if len(bad_ad):
        insights.append({
            "type": "danger",
            "icon": "🔥",
            "text": f"{len(bad_ad)} SKU с ДРР > 25% и CTR < 2%: {', '.join(bad_ad['art'].head(5))}. "
                    f"Корректируйте ставки/креативы или ставьте на паузу.",
        })

    # Rule 2: Margin < 0
    losers = skus[skus["margin_pct"] < 0]
    if len(losers):
        insights.append({
            "type": "danger",
            "icon": "📉",
            "text": f"{len(losers)} SKU убыточны: {', '.join(losers['art'].head(5))}. "
                    f"Пересчитайте цену или отключите рекламу.",
        })

    # Rule 3: Days of coverage < 10
    low_stock = skus[(skus["days_cover"] > 0) & (skus["days_cover"] < 10)]
    if len(low_stock):
        insights.append({
            "type": "warning",
            "icon": "📦",
            "text": f"{len(low_stock)} SKU с остатком < 10 дней: {', '.join(low_stock['art'].head(5))}. "
                    f"Срочная поставка из Китая.",
        })

    # Rule 4: Out of stock
    oos = skus[skus["days_cover"] == 0]
    if len(oos):
        insights.append({
            "type": "danger",
            "icon": "🛑",
            "text": f"{len(oos)} SKU без остатка: {', '.join(oos['art'].head(5))}. Теряете продажи прямо сейчас.",
        })

    # Rule 5: Z-score for revenue (anomalous drop for top SKUs)
    top = skus[skus["revenue"] > skus["revenue"].median()].copy()
    if len(top) > 5:
        top["z"] = (top["revenue"] - top["revenue"].mean()) / top["revenue"].std()
        anom = top[top["z"] < -1.5]
        if len(anom):
            insights.append({
                "type": "warning",
                "icon": "⚠️",
                "text": f"Аномальная просадка выручки: {', '.join(anom['art'].head(5))}. "
                        f"Проверьте остатки, отзывы, цену конкурентов.",
            })

    # Rule 6: Low buyout rate
    bad_buyout = skus[skus["buyout"] < 75]
    if len(bad_buyout):
        insights.append({
            "type": "warning",
            "icon": "↩️",
            "text": f"{len(bad_buyout)} SKU с выкупом < 75%: {', '.join(bad_buyout['art'].head(5))}. "
                    f"Проверьте размерную сетку, описание, фото.",
        })

    return insights


def build_sku_table(
    funnel_df: pd.DataFrame,
    stocks_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    ads_per_sku: pd.DataFrame,
    cogs_per_sku: pd.DataFrame,
    period_days: int,
) -> pd.DataFrame:
    """Aggregate SKU-level metrics for UI."""
    df = funnel_df.copy()

    # Pull article, name, category and stock from stocks
    if not stocks_df.empty:
        meta = stocks_df.groupby("nm_id").agg(
            art=("art", "first"),
            name=("name", "first"),
            cat=("cat", "first"),
            stock=("qty", "sum"),
        ).reset_index()
        df = df.merge(meta, on="nm_id", how="left")
    else:
        df["art"] = df["nm_id"].astype(str)
        df["name"] = ""
        df["cat"] = ""
        df["stock"] = 0

    # CTR from funnel (WB API uses openCardCount as both shows and clicks)
    df["ctr"] = np.where(df["shows"] > 0, df["clicks"] / df["shows"] * 100, 0)

    # Advertising costs per SKU
    if not ads_per_sku.empty:
        df = df.merge(ads_per_sku, on="nm_id", how="left")
    df["ads"] = df.get("ads", 0).fillna(0)

    # Cost of goods sold per SKU
    if not cogs_per_sku.empty:
        df = df.merge(cogs_per_sku, on="nm_id", how="left")
    df["cogs"] = df.get("cogs", 0).fillna(0)

    # Derived metrics
    df["drr"] = np.where(df["revenue"] > 0, df["ads"] / df["revenue"] * 100, 0)
    df["margin_abs"] = df["revenue"] - df["cogs"] - df["ads"]
    df["margin_pct"] = np.where(df["revenue"] > 0, df["margin_abs"] / df["revenue"] * 100, 0)

    # Days of stock coverage
    avg_orders_per_day = df["orders"] / max(period_days, 1)
    df["days_cover"] = np.where(
        avg_orders_per_day > 0,
        (df["stock"] / avg_orders_per_day).round(0),
        0,
    )

    return df.sort_values("revenue", ascending=False)


# ----------------------------------------------------------------------------
# 5. UI
# ----------------------------------------------------------------------------

st.set_page_config(page_title="WB · ТАЙКОНГ", page_icon="📊", layout="wide")

st.markdown(
    """
<style>
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }
    .stMetric { background: #F7F7F5; padding: 12px 16px; border-radius: 8px; }
    h1 { font-weight: 500; font-size: 24px; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("📊 ООО «ТАЙКОНГ» · Wildberries")
st.caption("Оперативный мониторинг · обновляется каждые 15 минут")

# Sidebar
with st.sidebar:
    st.subheader("Период")
    period_label = st.radio("Период", list(PERIODS.keys()), index=1, label_visibility="collapsed")
    period = PERIODS[period_label]

    st.subheader("Фильтры")
    cat_filter = st.multiselect("Категория", [], placeholder="Все")
    only_active = st.checkbox("Только активная реклама", value=False)

    st.divider()
    st.caption("WB API")
    if WB_API_TOKEN:
        st.success("Токен подключён")
    else:
        st.error("Нет WB_API_TOKEN. Добавьте в .env")

    if st.button("🔄 Обновить данные"):
        st.cache_data.clear()
        st.rerun()


# Date ranges
today = datetime.now()
date_from = today - timedelta(days=period.days)
prev_date_from = today - timedelta(days=period.days * 2)
prev_date_to   = today - timedelta(days=period.days)

# Load data
if not WB_API_TOKEN:
    st.warning(
        "Настройте `.env` с WB_API_TOKEN, чтобы увидеть реальные данные. Сейчас приложение работает в демо-режиме."
    )
    st.stop()

try:
    sales_df  = load_sales(date_from)
    orders_df = load_orders(date_from)
    stocks_df = load_stocks(today - timedelta(days=1))

    # nmID for funnel: union of sales and stock nmIDs
    nm_ids = list(set(
        (sales_df["nm_id"].tolist() if not sales_df.empty else []) +
        (stocks_df["nm_id"].tolist() if not stocks_df.empty else [])
    ))

    funnel_curr = load_funnel(nm_ids, date_from.date(), today.date())
    funnel_prev = load_funnel(nm_ids, prev_date_from.date(), prev_date_to.date())

    # TODO: integrate advertising via WBClient.advert_stats
    ads_per_sku = pd.DataFrame(columns=["nm_id", "ads"])
    ads_total_curr = 0
    ads_total_prev = 0

    # TODO: cost of goods sold dictionary; currently empty
    cogs_per_sku = pd.DataFrame(columns=["nm_id", "cogs"])
    cogs_total = 0

except Exception as e:
    st.error(f"Ошибка загрузки данных: {e}")
    log.exception(e)
    st.stop()

# KPI calculation
kpi_curr = calc_kpis(funnel_curr, ads_total_curr, cogs_total)
kpi_prev = calc_kpis(funnel_prev, ads_total_prev, cogs_total)

st.subheader("Ключевые показатели")
c1, c2, c3, c4 = st.columns(4)
c5, c6, c7, c8 = st.columns(4)


def kpi_card(col, label, key, fmt, invert=False):
    val = kpi_curr.get(key, 0)
    delta = calc_delta(kpi_curr, kpi_prev, key)
    delta_color = "inverse" if invert else "normal"
    col.metric(label, fmt(val), f"{delta:+.1f}%", delta_color=delta_color)


kpi_card(c1, "Выручка",         "revenue",    lambda v: f"{v:,.0f} ₽".replace(",", " "))
kpi_card(c2, "Маржа, %",        "margin_pct", lambda v: f"{v:.1f}%")
kpi_card(c3, "Реклама",         "ads",        lambda v: f"{v:,.0f} ₽".replace(",", " "), invert=True)
kpi_card(c4, "CTR, %",          "ctr",        lambda v: f"{v:.2f}%")
kpi_card(c5, "Конв. в корзину", "c2cart",     lambda v: f"{v:.1f}%")
kpi_card(c6, "Конв. в заказ",   "c2order",    lambda v: f"{v:.1f}%")
kpi_card(c7, "% выкупа",        "buyout",     lambda v: f"{v:.1f}%")
kpi_card(c8, "Заказы",          "orders",     lambda v: f"{v:,.0f}".replace(",", " "))

# Build SKU table
sku_df = build_sku_table(funnel_curr, stocks_df, sales_df, ads_per_sku, cogs_per_sku, period.days)

# Auto-insights
st.subheader("🔍 Авто-инсайты")
insights = detect_anomalies(sku_df)
if not insights:
    st.success("Аномалий не обнаружено — все ключевые метрики в норме.")
else:
    for ins in insights:
        if ins["type"] == "danger":
            st.error(f"{ins['icon']} {ins['text']}")
        elif ins["type"] == "warning":
            st.warning(f"{ins['icon']} {ins['text']}")
        else:
            st.info(f"{ins['icon']} {ins['text']}")

# Charts
st.subheader("📈 Динамика")
if not sales_df.empty:
    daily = sales_df.groupby(sales_df["date"].dt.date).agg(
        revenue=("revenue", "sum"),
        qty=("revenue", "count")
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=daily["date"], y=daily["revenue"], name="Выручка, ₽", marker_color="#185FA5", yaxis="y"))
    fig.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title="₽"),
        showlegend=False,
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

# Waterfall chart for margin breakdown
st.subheader("💰 Разбор маржи")
revenue_val = kpi_curr.get("revenue", 0)
if revenue_val > 0:
    cogs = cogs_total
    commission = revenue_val * 0.17  # average WB commission; adjust for your category
    ads = kpi_curr.get("ads", 0)
    logistics = revenue_val * 0.06
    returns = revenue_val * 0.04
    profit = revenue_val - cogs - commission - ads - logistics - returns

    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "relative", "relative", "total"],
        x=["Выручка", "Себестоимость", "Комиссии WB", "Реклама", "Логистика", "Возвраты", "Прибыль"],
        y=[revenue_val, -cogs, -commission, -ads, -logistics, -returns, profit],
        connector={"line": {"color": "#B4B2A9"}},
        increasing={"marker": {"color": "#0F6E56"}},
        decreasing={"marker": {"color": "#A32D2D"}},
        totals={"marker": {"color": "#185FA5"}},
    ))
    fig_wf.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="white")
    st.plotly_chart(fig_wf, use_container_width=True)

# Scatter: advertising vs CTR
if not sku_df.empty and sku_df["ads"].sum() > 0:
    st.subheader("🎯 Реклама vs CTR")
    fig_sc = px.scatter(
        sku_df,
        x="ads",
        y="ctr",
        size="revenue",
        color="margin_pct",
        hover_data=["art", "name", "drr", "margin_pct"],
        color_continuous_scale=[(0, "#A32D2D"), (0.5, "#BA7517"), (1, "#0F6E56")],
        labels={"ads": "Расход на рекламу, ₽", "ctr": "CTR, %", "margin_pct": "Маржа, %"},
    )
    fig_sc.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_sc, use_container_width=True)

# SKU table
st.subheader("📋 Артикулы")
search = st.text_input("Поиск по артикулу или названию", "")
if search:
    sku_df = sku_df[
        sku_df["art"].str.contains(search, case=False, na=False) |
        sku_df["name"].str.contains(search, case=False, na=False)
    ]

display_cols = ["art", "name", "cat", "shows", "ctr", "c2cart", "c2order",
                "buyout", "orders", "revenue", "ads", "drr", "margin_pct",
                "stock", "days_cover"]
display_cols = [c for c in display_cols if c in sku_df.columns]

st.dataframe(
    sku_df[display_cols],
    use_container_width=True,
    hide_index=True,
    column_config={
        "art":        st.column_config.TextColumn("Артикул", width="small"),
        "name":       st.column_config.TextColumn("Название"),
        "cat":        st.column_config.TextColumn("Категория", width="small"),
        "shows":      st.column_config.NumberColumn("Показы", format="%d"),
        "ctr":        st.column_config.NumberColumn("CTR %", format="%.1f"),
        "c2cart":     st.column_config.NumberColumn("Корз. %", format="%.1f"),
        "c2order":    st.column_config.NumberColumn("Зак. %", format="%.1f"),
        "buyout":     st.column_config.NumberColumn("Выкуп %", format="%.1f"),
        "orders":     st.column_config.NumberColumn("Заказы", format="%d"),
        "revenue":    st.column_config.NumberColumn("Выручка ₽", format="%.0f"),
        "ads":        st.column_config.NumberColumn("Реклама ₽", format="%.0f"),
        "drr":        st.column_config.NumberColumn("ДРР %", format="%.1f"),
        "margin_pct": st.column_config.NumberColumn("Маржа %", format="%.1f"),
        "stock":      st.column_config.NumberColumn("Остаток", format="%d"),
        "days_cover": st.column_config.NumberColumn("Дни покр.", format="%d"),
    },
)

csv = sku_df.to_csv(index=False).encode("utf-8-sig")
st.download_button("⬇️ Скачать CSV", csv, f"taikong_skus_{today.date()}.csv", "text/csv")

st.caption(
    f"Обновлено: {today.strftime('%d.%m.%Y %H:%M')} · Период: {period.label} · SKU в выборке: {len(sku_df)}"
)