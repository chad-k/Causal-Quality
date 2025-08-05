# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 15:09:43 2025

@author: ckaln
"""


import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from econml.dml import LinearDML

st.title("📈 Full Causal Inference Dashboard (Interpretation + What-If + Visualization)")

@st.cache_data
def load_data():
    return pd.read_csv("c:/users/ckaln/downloads/causal_data_strong_effect.csv")

@st.cache_resource
def train_models(df_encoded, X, y):
    models = {}
    for treatment in ["Pressure", "Speed", "Temperature"]:
        T = df_encoded[[treatment]]
        model = LinearDML(
            model_y=RandomForestRegressor(),
            model_t=RandomForestRegressor(),
            random_state=42
        )
        model.fit(y, T, X=X)
        models[treatment] = model
    return models

df = load_data()
part_selected = st.selectbox("Select Part Number", df["PartNo"].unique())
machine_selected = st.selectbox("Select Machine", df["Machine"].unique())

subset = df[(df["PartNo"] == part_selected) & (df["Machine"] == machine_selected)]

if len(subset) < 200:
    st.warning("Not enough data for selected Part Number + Machine. Try another combination.")
else:
    avg_pressure = round(subset["Pressure"].mean(), 2)
    avg_speed = round(subset["Speed"].mean(), 2)
    avg_temp = round(subset["Temperature"].mean(), 2)

    st.markdown(f"### 📊 Averages for {part_selected} on {machine_selected}")
    st.markdown(f"- Pressure: `{avg_pressure}` bar")
    st.markdown(f"- Speed: `{avg_speed}` RPM")
    st.markdown(f"- Temperature: `{avg_temp}` °C")

    df_encoded = pd.get_dummies(subset, columns=["Operator", "MaterialLot", "Tool"], drop_first=True)
    st.subheader("📄 Sample Data")
    st.dataframe(df_encoded.head())

    X = df_encoded.drop(columns=["DefectRate", "PartNo", "Machine"])
    y = df_encoded["DefectRate"]

    causal_models = train_models(df_encoded, X, y)

    st.subheader("📌 Causal Effect Summary with Interpretation")

    effect_means, ci_lowers, ci_uppers = [], [], []

    def interpret(treatment, effect, ci_low, ci_high):
        if ci_low > 0:
            return "📈 Significant Increase"
        elif ci_high < 0:
            return "📉 Significant Decrease"
        elif ci_high - ci_low < 2.0:
            return "⚠️ Marginal (CI narrow but includes 0)"
        else:
            return "❓ Uncertain (wide CI, includes 0)"

    for treatment in ["Pressure", "Speed", "Temperature"]:
        model = causal_models[treatment]
        effect = model.effect(X)
        lower, upper = model.effect_interval(X)

        mean_eff = np.mean(effect)
        lower_mean = np.mean(lower)
        upper_mean = np.mean(upper)
        tag = interpret(treatment, mean_eff, lower_mean, upper_mean)

        st.markdown(
            f"**{treatment}**  "
            f"- Effect: `{mean_eff:.2f}`  "
            f"- 95% CI: `[ {lower_mean:.2f}, {upper_mean:.2f} ]`  "
            f"- Interpretation: **{tag}**"
        )

        effect_means.append(mean_eff)
        ci_lowers.append(lower_mean)
        ci_uppers.append(upper_mean)

    st.subheader("📊 Visual Causal Summary")
    fig, ax = plt.subplots(figsize=(8, 5))
    means = np.array(effect_means)
    lowers = means - np.array(ci_lowers)
    uppers = np.array(ci_uppers) - means
    ax.bar(["Pressure", "Speed", "Temperature"], means, yerr=[lowers, uppers], capsize=5, color="skyblue")
    ax.set_title(f"Causal Effects on Defect Rate for {part_selected} on {machine_selected}")
    ax.set_ylabel("Effect on Defect Rate (%)")
    ax.grid(axis="y")
    st.pyplot(fig)


    st.subheader("🧪 What-If Scenario Simulator")

    input_pressure = st.slider("Pressure (bar)", 3.0, 8.0, float(avg_pressure), 0.1)
    input_speed = st.slider("Speed (RPM)", 80, 120, int(avg_speed), 1)
    input_temp = st.slider("Temperature (°C)", 190, 210, int(avg_temp), 1)

    baseline = pd.DataFrame({col: [0] for col in X.columns})
    for col in baseline.columns:
        if col.startswith("Operator_"):
            baseline[col] = 1 if "O124" in col else 0
        if col.startswith("MaterialLot_"):
            baseline[col] = 1 if "Lot-B" in col else 0
        if col.startswith("Tool_"):
            baseline[col] = 1 if "T2" in col else 0
    baseline["Pressure"] = float(avg_pressure)
    baseline["Speed"] = float(avg_speed)
    baseline["Temperature"] = float(avg_temp)

    whatif = baseline.copy()
    whatif["Pressure"] = input_pressure
    whatif["Speed"] = input_speed
    whatif["Temperature"] = input_temp

    st.subheader("📈 What-If Causal Effects")
    effects = {}
    for treatment in ["Pressure", "Speed", "Temperature"]:
        model = causal_models[treatment]
        effect = model.effect(whatif)[0]
        effects[treatment] = effect
        st.markdown(f"- `{treatment}` → effect = `{effect:.2f}`")

    base_rate = 10 - 0.8 * avg_pressure + 0.2 * (avg_speed - 100) + 0.3 * (avg_temp - 200)
    whatif_rate = base_rate + sum(effects.values())
    base_rate = np.clip(base_rate, 0, 100)
    whatif_rate = np.clip(whatif_rate, 0, 100)

    st.subheader("📉 Baseline vs What-If Defect Rate")
    fig2, ax2 = plt.subplots()
    ax2.bar(["Baseline", "What-If"], [base_rate, whatif_rate], color=["gray", "blue"])
    ax2.set_ylabel("Defect Rate (%)")
    ax2.set_title("Scenario Comparison")
    st.pyplot(fig2)

    st.markdown(f"📌 **Baseline Defect Rate**: `{base_rate:.2f}%`")
    st.markdown(f"📌 **What-If Defect Rate**: `{whatif_rate:.2f}%`")

    st.success("Full causal analysis complete.")
