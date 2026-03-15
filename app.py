import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

# ── Load model ──────────────────────────────────────────
model = joblib.load("model.pkl")
with open("model_features.json") as f:
    model_features = json.load(f)

# ── Predict function ─────────────────────────────────────
def predict_income(price, yield_kg, month, state, commodity):
    sample = pd.DataFrame(columns=model_features)
    sample.loc[0] = 0
    sample["price_modal"] = price
    sample["yield_kg_ha"] = yield_kg
    sample["month"] = month
    state_col = "state_" + state
    if state_col in sample.columns:
        sample[state_col] = 1
    crop_col = "commodity_" + commodity
    if crop_col in sample.columns:
        sample[crop_col] = 1
    return model.predict(sample)[0]

# ── Page config ───────────────────────────────────────────
st.set_page_config(page_title="AgriConnect+", page_icon="🌾", layout="centered")
st.title("🌾 AgriConnect+ | Crop vs Lease Income Comparison")
st.markdown("**Objective 1 — Help farmers decide: Should I farm or lease my land?**")
st.divider()

# ── Inputs ────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    price      = st.number_input("Crop Price (Rs/quintal)", min_value=0.0, value=1200.0, step=50.0)
    yield_kg   = st.number_input("Yield (kg/hectare)", min_value=0.0, value=25000.0, step=500.0)
    month      = st.slider("Month", min_value=1, max_value=12, value=6)

with col2:
    state      = st.text_input("State", value="Haryana")
    crop       = st.text_input("Crop", value="Tomato")
    acres      = st.number_input("Land Size (Acres)", min_value=0.1, value=2.0, step=0.5)
    lease_price = st.number_input("Lease Price (Rs/acre/year)", min_value=0.0, value=15000.0, step=500.0)

st.divider()

# ── Compare button ────────────────────────────────────────
if st.button("🔍 Compare Income", use_container_width=True):

    crop_income_per_acre = predict_income(price, yield_kg, month, state.strip().title(), crop.strip().title())
    total_crop_income    = crop_income_per_acre * acres
    total_lease_income   = lease_price * acres
    difference           = total_crop_income - total_lease_income
    margin_pct           = abs(difference / total_lease_income) * 100 if total_lease_income > 0 else 0
    better               = "🌾 Crop Farming" if difference > 0 else "🏠 Leasing Land"

    # ── Metrics ───────────────────────────────────────────
    st.subheader("📊 Results")
    m1, m2, m3 = st.columns(3)
    m1.metric("Crop Income (Total)", f"₹{total_crop_income:,.0f}", f"₹{crop_income_per_acre:,.0f}/acre")
    m2.metric("Lease Income (Total)", f"₹{total_lease_income:,.0f}", f"₹{lease_price:,.0f}/acre")
    m3.metric("Difference", f"₹{abs(difference):,.0f}", f"{margin_pct:.1f}% advantage")

    # ── Recommendation ────────────────────────────────────
    if difference > 0:
        st.success(f"✅ **Recommendation: {better}** — Growing {crop} earns ₹{abs(difference):,.0f} MORE than leasing.")
    else:
        st.warning(f"⚠️ **Recommendation: {better}** — Leasing earns ₹{abs(difference):,.0f} MORE than growing {crop}.")

    # ── Bar Chart ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    categories = ['Crop Farming', 'Leasing Land']
    values     = [total_crop_income, total_lease_income]
    colors     = ['#2ecc71' if total_crop_income >= total_lease_income else '#e74c3c',
                  '#2ecc71' if total_lease_income > total_crop_income  else '#e74c3c']

    bars = ax.bar(categories, values, color=colors, width=0.4, edgecolor='white')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(values)*0.01,
                f'₹{val:,.0f}', ha='center', fontsize=11, fontweight='bold')

    ax.set_title(f'{crop} Farming vs Leasing ({acres} acres)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Total Income (₹)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
    plt.tight_layout()
    st.pyplot(fig)
