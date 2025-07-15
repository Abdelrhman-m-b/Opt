import streamlit as st
import pandas as pd
import numpy as np
import joblib


def load_model_and_scaler(diameter, stage):
    model_path = f"models/{stage}_model_d{diameter}.pkl"
    scaler_path = f"scalers/scaler_d{diameter}.pkl"
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_mechanical_properties(input_df, diameter):
    # Load models & scaler
    rm_model, scaler = load_model_and_scaler(diameter, "Rm")
    reh_model, _ = load_model_and_scaler(diameter, "ReH")

    # Derived features
    input_df["MnS"] = input_df["Mn"] * input_df["S"]
    input_df["MnSi"] = input_df["Mn"] * input_df["Si"]
    input_df["CEC"] = input_df["CE"] / input_df["C"].replace(0, 1e-8)
    input_df["CEMn"] = input_df["CE"] / input_df["Mn"].replace(0, 1e-8)
    input_df["C_Mn"] = input_df["C"] * input_df["Mn"]

    chem_features = ['C', 'Si', 'Mn', 'P', 'S', 'Ni', 'Cr', 'Mo', 'Cu', 'V', 'N',
                     'MnS', 'MnSi', 'CE', 'CEC', 'CEMn', 'C_Mn']

    # Scale
    X_scaled = input_df[chem_features].copy()
    X_scaled = pd.DataFrame(scaler.transform(X_scaled), columns=chem_features)

    # Predict
    rm_pred = rm_model.predict(X_scaled)
    X_scaled["Predicted_Rm"] = rm_pred
    reh_pred = reh_model.predict(X_scaled)

    return rm_pred[0], reh_pred[0]


def optimize_C_Mn(base_comp, diameter, rm_min, reh_min=500, c_range=(0.15, 0.30), mn_range=(0.5, 2.0), step=0.01):
    results = []
    for c in np.arange(*c_range, step):
        for mn in np.arange(*mn_range, step):
            comp = base_comp.copy()
            comp["C"] = c
            comp["Mn"] = mn
            comp["CE"] = c + mn / 6
            try:
                rm, reh = predict_mechanical_properties(pd.DataFrame([comp]), diameter)
                if rm >= rm_min and reh >= reh_min:
                    results.append((c, mn, rm, reh, c + mn))
            except:
                continue
    df = pd.DataFrame(results, columns=["C", "Mn", "Rm", "ReH", "C+Mn"])
    return df.sort_values("C+Mn").reset_index(drop=True)


# Streamlit UI
st.set_page_config(page_title="Steel Alloy Optimizer", layout="centered")
st.title("⚙️ Steel Alloy Optimizer")
st.markdown("Optimize **C** and **Mn** additions for mechanical performance and cost.")

# Sidebar controls
diameter = st.selectbox("Select Rebar Diameter (mm)", [10, 12, 16, 18, 20])
rm_min = st.number_input("Target Rm (MPa)", min_value=400, value=550)
reh_min = st.number_input("Target ReH (MPa)", min_value=300, value=500)
c_min, c_max = st.slider("C range (%)", 0.10, 0.40, (0.15, 0.30), 0.01)
mn_min, mn_max = st.slider("Mn range (%)", 0.30, 2.00, (0.50, 1.80), 0.01)
step = st.select_slider("Search Step", options=[0.005, 0.01, 0.02], value=0.01)

# Input area
st.markdown("---")
st.subheader("🧪 Enter Chemical Composition (excluding C and Mn)")
default_comp = {
    "Si": 0.25, "P": 0.02, "S": 0.02, "Ni": 0.05, "Cr": 0.05, "Mo": 0.01,
    "Cu": 0.05, "V": 0.01, "N": 0.01
}
comp = {}
cols = st.columns(3)
for i, el in enumerate(default_comp):
    with cols[i % 3]:
        comp[el] = st.number_input(f"{el} (%)", value=default_comp[el], key=el)

# Reset button
if st.button("🔄 Reset to Default Composition"):
    st.experimental_rerun()

# Run button
if st.button("Run Optimization"):
    with st.spinner("Optimizing... Please wait..."):
        base = comp.copy()
        base["C"] = 0.0
        base["Mn"] = 0.0
        base["CE"] = 0.0  # Placeholder

        df_results = optimize_C_Mn(
            base_comp=base,
            diameter=diameter,
            rm_min=rm_min,
            reh_min=reh_min,
            c_range=(c_min, c_max),
            mn_range=(mn_min, mn_max),
            step=step
        )

        if not df_results.empty:
            st.success("✅ Optimization Complete!")
            st.write("### Top Result")
            st.dataframe(df_results.head(1))
            st.write("### All Valid Combinations")
            st.dataframe(df_results)
            st.download_button("📥 Download Results", df_results.to_csv(index=False), file_name="optimized_alloy.csv")
        else:
            st.error("❌ No combinations found that meet the criteria.")
