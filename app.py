import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import requests

# ── Load model ───────────────────────────────────────────────
model = joblib.load("model.pkl")
with open("model_features.json") as f:
    model_features = json.load(f)

# ── API Config ───────────────────────────────────────────────
API_KEY       = "579b464db66ec23bdd000001f825acb3b2094b196a7677458c9f9c1c"
MANDI_API     = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"

# ── State-wise lease rates (Rs/acre/year) ────────────────────
LEASE_RATES = {
    "Andhra Pradesh":  {"min": 12000, "avg": 18000, "max": 28000},
    "Karnataka":       {"min": 10000, "avg": 16000, "max": 25000},
    "Tamil Nadu":      {"min": 8000,  "avg": 13000, "max": 20000},
    "Telangana":       {"min": 10000, "avg": 15000, "max": 22000},
    "Kerala":          {"min": 15000, "avg": 25000, "max": 47000},
    "Maharashtra":     {"min": 8000,  "avg": 14000, "max": 22000},
    "Gujarat":         {"min": 7000,  "avg": 12000, "max": 18000},
    "Punjab":          {"min": 18000, "avg": 28000, "max": 40000},
    "Haryana":         {"min": 15000, "avg": 24000, "max": 35000},
    "Uttar Pradesh":   {"min": 6000,  "avg": 10000, "max": 16000},
    "Madhya Pradesh":  {"min": 5000,  "avg": 9000,  "max": 14000},
    "Rajasthan":       {"min": 4000,  "avg": 8000,  "max": 13000},
    "Bihar":           {"min": 5000,  "avg": 9000,  "max": 14000},
    "West Bengal":     {"min": 6000,  "avg": 11000, "max": 18000},
    "Odisha":          {"min": 4000,  "avg": 8000,  "max": 13000},
    "Assam":           {"min": 4000,  "avg": 7000,  "max": 12000},
    "Chhattisgarh":    {"min": 4000,  "avg": 7000,  "max": 11000},
    "Jharkhand":       {"min": 4000,  "avg": 7000,  "max": 11000},
    "Uttarakhand":     {"min": 6000,  "avg": 11000, "max": 18000},
    "Himachal Pradesh":{"min": 5000,  "avg": 9000,  "max": 15000},
}
DEFAULT_LEASE = {"min": 6000, "avg": 10000, "max": 16000}

# ── Live Mandi Price Fetch ───────────────────────────────────
@st.cache_data(ttl=3600)  # cache for 1 hour
def fetch_live_price(state, commodity):
    try:
        params = {
            "api-key": API_KEY,
            "format":  "json",
            "limit":   100,
            "filters[state]":     state,
            "filters[commodity]": commodity,
        }
        response = requests.get(MANDI_API, params=params, timeout=10)
        data     = response.json()
        records  = data.get("records", [])

        if not records:
            return None, "No data found for this state & crop combination."

        prices = [float(r["modal_price"]) for r in records if r.get("modal_price")]
        if not prices:
            return None, "Price data unavailable."

        avg_price = round(sum(prices) / len(prices), 2)
        return avg_price, f"✅ Fetched from {len(prices)} mandi records for {commodity} in {state}"

    except Exception as e:
        return None, f"❌ API error: {str(e)}"

# ── Predict function ─────────────────────────────────────────
def predict_income(price, yield_kg, month, state, commodity):
    sample = pd.DataFrame(columns=model_features)
    sample.loc[0] = 0
    sample["price_modal"]  = price
    sample["yield_kg_ha"]  = yield_kg
    sample["month"]        = month
    state_col = "state_" + state
    if state_col in sample.columns:
        sample[state_col] = 1
    crop_col = "commodity_" + commodity
    if crop_col in sample.columns:
        sample[crop_col] = 1
    return model.predict(sample)[0]

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="AgriConnect+", page_icon="🌾", layout="centered")
st.title("🌾 AgriConnect+ | Crop vs Lease Income Comparison")
st.markdown("**Objective 1 — Help farmers decide: Should I farm or lease my land?**")
st.divider()

# ── Inputs ───────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    state    = st.selectbox("State", sorted(LEASE_RATES.keys()), index=list(sorted(LEASE_RATES.keys())).index("Haryana"))
    crop     = st.text_input("Crop", value="Tomato")
    month    = st.slider("Month", min_value=1, max_value=12, value=6)

with col2:
    yield_kg = st.number_input("Yield (kg/hectare)", min_value=0.0, value=25000.0, step=500.0)
    acres    = st.number_input("Land Size (Acres)", min_value=0.1, value=2.0, step=0.5)

# ── Auto-fetch crop price ─────────────────────────────────────
st.subheader("📡 Live Crop Price")

if st.button("🔄 Fetch Live Price from Mandi API", use_container_width=True):
    with st.spinner(f"Fetching live price for {crop} in {state}..."):
        fetched_price, msg = fetch_live_price(state, crop)
    if fetched_price:
        st.session_state["crop_price"] = fetched_price
        st.success(msg)
    else:
        st.warning(f"{msg} — Please enter price manually below.")

# Show fetched price or allow manual entry
default_price = st.session_state.get("crop_price", 1200.0)
price = st.number_input(
    "Crop Price (Rs/quintal) — auto-fetched or enter manually:",
    min_value=0.0,
    value=float(default_price),
    step=50.0
)

if "crop_price" in st.session_state:
    st.caption(f"📊 Live average mandi price for **{crop}** in **{state}**: ₹{st.session_state['crop_price']:,}/quintal")

st.divider()

# ── Auto lease rate ───────────────────────────────────────────
lease_data  = LEASE_RATES.get(state, DEFAULT_LEASE)
lease_min   = lease_data["min"]
lease_avg   = lease_data["avg"]
lease_max   = lease_data["max"]

st.info(f"📍 **{state} Lease Rate (per acre/year):** "
        f"Min ₹{lease_min:,} | Avg ₹{lease_avg:,} | Max ₹{lease_max:,}  "
        f"*(Source: NITI Aayog Agricultural Land Leasing Report)*")

lease_choice = st.radio(
    "Use which lease rate for comparison?",
    ["Minimum", "Average", "Maximum"],
    index=1,
    horizontal=True
)
lease_price = {"Minimum": lease_min, "Average": lease_avg, "Maximum": lease_max}[lease_choice]

st.divider()

# ── Compare button ───────────────────────────────────────────
if st.button("🔍 Compare Income", use_container_width=True):

    crop_income_per_acre = predict_income(price, yield_kg, month, state.strip().title(), crop.strip().title())
    total_crop_income    = crop_income_per_acre * acres
    total_lease_income   = lease_price * acres
    difference           = total_crop_income - total_lease_income
    margin_pct           = abs(difference / total_lease_income) * 100 if total_lease_income > 0 else 0
    better               = "🌾 Crop Farming" if difference > 0 else "🏠 Leasing Land"

    # ── Metrics ──────────────────────────────────────────────
    st.subheader("📊 Results")
    m1, m2, m3 = st.columns(3)
    m1.metric("Crop Income (Total)",  f"₹{total_crop_income:,.0f}",  f"₹{crop_income_per_acre:,.0f}/acre")
    m2.metric("Lease Income (Total)", f"₹{total_lease_income:,.0f}", f"₹{lease_price:,}/acre")
    m3.metric("Difference",           f"₹{abs(difference):,.0f}",    f"{margin_pct:.1f}% advantage")

    # ── Recommendation ────────────────────────────────────────
    if difference > 0:
        st.success(f"✅ **Recommendation: {better}** — Growing {crop} earns ₹{abs(difference):,.0f} MORE than leasing.")
    else:
        st.warning(f"⚠️ **Recommendation: {better}** — Leasing earns ₹{abs(difference):,.0f} MORE than growing {crop}.")

    # ── Bar Chart ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    categories = ['Crop Farming', 'Leasing Land']
    values     = [total_crop_income, total_lease_income]
    colors     = ['#2ecc71' if total_crop_income >= total_lease_income else '#e74c3c',
                  '#2ecc71' if total_lease_income > total_crop_income  else '#e74c3c']

    bars = ax.bar(categories, values, color=colors, width=0.4, edgecolor='white')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(values) * 0.01,
                f'₹{val:,.0f}', ha='center', fontsize=11, fontweight='bold')

    ax.set_title(f'{crop} Farming vs Leasing — {state} ({acres} acres)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Income (₹)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
    plt.tight_layout()
    st.pyplot(fig)

    st.caption(f"ℹ️ Lease rate for {state}: ₹{lease_min:,} – ₹{lease_max:,}/acre/year. "
               f"Compared using **{lease_choice.lower()}** rate of ₹{lease_price:,}.")
