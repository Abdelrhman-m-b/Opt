import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ========== Utility Functions ==========

def load_model_and_scaler(diameter, stage):
    diameter = float(diameter)  # Ensure float format
    model_path = f"models/model_{stage}_d{diameter}.pkl"
    scaler_path = f"scalers/scaler_d{diameter}.pkl"
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict_mechanical_properties(input_df, diameter):
    rm_model, scaler = load_model_and_scaler(diameter, "Rm")
    reh_model, _ = load_model_and_scaler(diameter, "ReH")

    chem_features = ['C', 'Si', 'Mn', 'P', 'S', 'Ni', 'Cr', 'Mo', 'Cu', 'V', 'N',
                     'MnS', 'MnSi', 'CE', 'CEC', 'CEMn', 'C_Mn']

    input_df_scaled = input_df[chem_features].copy()
    input_df_scaled = pd.DataFrame(scaler.transform(input_df_scaled), columns=chem_features)

    rm_pred = rm_model.predict(input_df_scaled)
    input_df_scaled["Predicted_Rm"] = rm_pred
    reh_pred = reh_model.predict(input_df_scaled)

    return rm_pred[0], reh_pred[0]

def apply_engineered_features(chem):
    chem = chem.copy()
    chem["MnS"] = chem["Mn"] * chem["S"]
    chem["MnSi"] = chem["Mn"] * chem["Si"]
    chem["CE"] = chem["C"] + chem["Mn"] / 6 + (chem["Cr"] + chem["Mo"] + chem["V"]) / 5 + (chem["Ni"] + chem["Cu"]) / 15
    chem["CEC"] = chem["CE"] / (chem["C"] if chem["C"] != 0 else 1e-8)
    chem["CEMn"] = chem["CE"] / (chem["Mn"] if chem["Mn"] != 0 else 1e-8)
    chem["C_Mn"] = chem["C"] * chem["Mn"]
    return chem

def optimize_C_Mn(base_comp, diameter_list, rm_min, reh_min, step=0.01):
    results = []
    for dia in diameter_list:
        for c in np.arange(0.15, 0.31, step):
            for mn in np.arange(0.5, 2.01, step):
                chem = base_comp.copy()
                chem["C"] = c
                chem["Mn"] = mn
                chem = apply_engineered_features(chem)
                try:
                    rm, reh = predict_mechanical_properties(pd.DataFrame([chem]), dia)
                    if rm >= rm_min and reh >= reh_min:
                        results.append((dia, c, mn, rm, reh, c + mn))
                except:
                    continue
    df = pd.DataFrame(results, columns=["Diameter", "C", "Mn", "Rm", "ReH", "C+Mn"])
    return df.sort_values("C+Mn").reset_index(drop=True)

def optimize_for_selected_diameter(first_input, selected_d, rm_min, reh_min=500, step=0.01):
    results = []
    for c in np.arange(0.15, 0.31, step):
        for mn in np.arange(0.5, 2.01, step):
            comp = first_input.copy()
            comp["C"] = c
            comp["Mn"] = mn
            comp = apply_engineered_features(comp)
            try:
                rm, reh = predict_mechanical_properties(pd.DataFrame([comp]), selected_d)
                if rm >= rm_min and reh >= reh_min:
                    results.append((c, mn, rm, reh, c + mn))
            except:
                continue
    df = pd.DataFrame(results, columns=["C", "Mn", "Rm", "ReH", "C+Mn"])
    return df.sort_values("C+Mn")

# ========== Streamlit App ==========

st.set_page_config(page_title="Steel Alloy Optimizer", layout="centered")
st.title("🔩 Steel Alloy Optimization Tool")
st.markdown("Estimate mechanical properties and **optimize alloying additions** for EAF rebars.")

diameter = st.selectbox("Select initial test diameter", [10.0, 12.0, 16.0, 18.0, 32.0])
rm_min = st.number_input("Target Rm (MPa)", min_value=400, value=550)
reh_min = st.number_input("Target ReH (MPa)", min_value=300, value=500)
step = st.select_slider("Optimization Step", options=[0.005, 0.01, 0.02], value=0.01)

st.markdown("## Enter First Composition (Initial Test) — Low C & Mn Expected")

default_chem = {
    'C': 0.18, 'Si': 0.2, 'Mn': 0.8, 'P': 0.01, 'S': 0.01,
    'Ni': 0.02, 'Cr': 0.03, 'Mo': 0.01, 'Cu': 0.02, 'V': 0.01, 'N': 0.008
}
if "chem" not in st.session_state:
    st.session_state.chem = default_chem.copy()

cols = st.columns(6)
for i, el in enumerate(default_chem):
    st.session_state.chem[el] = cols[i % 6].number_input(el, value=st.session_state.chem[el])

if st.button("Reset Composition"):
    st.session_state.chem = default_chem.copy()
    st.experimental_rerun()

# Prediction from test
input_comp = apply_engineered_features(st.session_state.chem.copy())
rm_pred, reh_pred = predict_mechanical_properties(pd.DataFrame([input_comp]), diameter)

st.markdown(f"### 🔍 Prediction for Initial Composition (Ø {diameter} mm)")
st.write(f"**Predicted Rm**: {rm_pred:.1f} MPa")
st.write(f"**Predicted ReH**: {reh_pred:.1f} MPa")

# Optimization Trigger
if st.button("Run Optimization"):
    with st.spinner("🔧 Optimizing C and Mn across valid combinations..."):

        # Option 1: Any Diameter (min total alloying)
        all_d_opt = optimize_C_Mn(base_comp=st.session_state.chem, diameter_list=[10, 12, 16, 18, 32],
                                  rm_min=rm_min, reh_min=reh_min, step=step)
        
        st.subheader("✅ Option 1: Lowest C+Mn across any diameter")
        if not all_d_opt.empty:
            st.dataframe(all_d_opt.head(1))
            st.download_button("📥 Download All Combinations", all_d_opt.to_csv(index=False), "option1_lowest_alloy.csv")
        else:
            st.warning("No valid combinations found for any diameter.")

        # Option 2: Same Diameter
        same_d_opt = optimize_for_selected_diameter(first_input=st.session_state.chem,
                                                    selected_d=diameter, rm_min=rm_min,
                                                    reh_min=reh_min, step=step)

        st.subheader(f"✅ Option 2: Optimize within diameter Ø {diameter} mm")
        if not same_d_opt.empty:
            st.dataframe(same_d_opt.head(1))
            st.download_button("📥 Download Same Diameter Results", same_d_opt.to_csv(index=False), "option2_same_d.csv")
        else:
            st.warning("No valid compositions found for this diameter.")

