import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Multi-Gas Thermal MFC Digital Twin", layout="wide")
R = 8.314462618
P_N = 101325.0
T_N = 273.15
T_REF = 293.15

# Illustrative gas properties. Replace with validated property data for design work.
GASES = {
    "Air": {"MW": 0.0289652, "cp": 1005.0, "k": 0.0262, "mu": 1.81e-5, "gamma": 1.400},
    "N2":  {"MW": 0.0280134, "cp": 1040.0, "k": 0.0258, "mu": 1.76e-5, "gamma": 1.400},
    "CO2": {"MW": 0.0440095, "cp": 844.0,  "k": 0.0168, "mu": 1.48e-5, "gamma": 1.289},
    "H2":  {"MW": 0.0020159, "cp": 14300.0,"k": 0.180,  "mu": 8.9e-6,  "gamma": 1.405},
    "CH4": {"MW": 0.016043,  "cp": 2220.0, "k": 0.034,  "mu": 1.10e-5, "gamma": 1.304},
    "CO":  {"MW": 0.028010,  "cp": 1040.0, "k": 0.024,  "mu": 1.75e-5, "gamma": 1.400},
    "Ar":  {"MW": 0.039948,  "cp": 520.3,  "k": 0.0177, "mu": 2.23e-5, "gamma": 1.667},
}

def mass_from_nlpm(q_nlpm, mw):
    return (q_nlpm / 60000.0) * (P_N * mw / (R * T_N))

def nlpm_from_mass(mdot, mw):
    return (mdot / mw) * (R * T_N / P_N) * 60000.0

def thermal_sensor_signal(q_mass, gas, params):
    """Illustrative calibrated bridge signal model; not a real sensor's transfer function."""
    p = GASES[gas]
    cp_ratio = p["cp"] / GASES["Air"]["cp"]
    k_ratio = p["k"] / GASES["Air"]["k"]
    # A simple gas-dependent sensitivity model, deliberately requiring calibration.
    sensitivity = params["base_sensitivity"] * (0.65 * cp_ratio**0.35 + 0.35 * k_ratio**0.65)
    return params["zero_v"] + sensitivity * (q_mass * 1e6) + params["nonlinearity"] * (q_mass * 1e6)**2

def fit_gas_calibration(gas, params, q_points):
    mdots = mass_from_nlpm(q_points, GASES[gas]["MW"])
    signals = np.array([thermal_sensor_signal(m, gas, params) for m in mdots])
    # In real use, fit measured signal against reference mass flow, per gas.
    coeff = np.polyfit(signals - params["zero_v"], q_points, 2)
    return coeff, signals

def inverse_calibration(signal, coeff, zero_v):
    x = max(0.0, signal - zero_v)
    return max(0.0, float(np.polyval(coeff, x)))

def simulate(cfg, cal_coeff):
    dt, duration = cfg["dt"], cfg["duration"]
    t = np.arange(0, duration + dt, dt)
    mw = GASES[cfg["gas"]]["MW"]
    # Sensor calibration mapping returns standard volumetric flow NLPM.
    opening = cfg["initial_opening"] / 100.0
    sensor_lag = 0.0
    integral = 0.0
    prev_error = 0.0
    rows = []
    for i, ti in enumerate(t):
        sp = cfg["setpoint"] if ti >= cfg["step_time"] else cfg["initial_setpoint"]
        # Simple valve capacity model: capacity proportional to opening^1.6 and sqrt pressure head.
        dp_bar = max(cfg["pin_bar"] - cfg["pout_bar"], 0.0)
        q_capacity = cfg["valve_capacity"] * opening**1.6 * np.sqrt(dp_bar / 0.2)
        q_true = max(0.0, min(q_capacity, cfg["max_flow"]))
        mdot = mass_from_nlpm(q_true, mw)

        # Virtual thermal bridge response and gas-specific calibration.
        raw = thermal_sensor_signal(mdot, cfg["gas"], cfg)
        raw += np.random.normal(0, cfg["noise_v"])
        raw += cfg["zero_drift_v"]
        raw += cfg["gain_error_pct"] / 100.0 * (raw - cfg["zero_v"])
        tau_s = cfg["sensor_tau"]
        sensor_lag += (raw - sensor_lag) * min(1.0, dt / max(tau_s, dt))
        q_indicated = inverse_calibration(sensor_lag, cal_coeff, cfg["zero_v"])

        error = sp - q_indicated
        if i > 0:
            integral += error * dt
        integral = np.clip(integral, -cfg["i_limit"], cfg["i_limit"])
        derivative = (error - prev_error) / dt if i else 0.0
        prev_error = error
        u = cfg["kp"] * error + cfg["ki"] * integral + cfg["kd"] * derivative
        command = np.clip(opening + cfg["controller_gain"] * u * dt, 0.0, 1.0)
        opening += (command - opening) * min(1.0, dt / max(cfg["valve_tau"], dt))
        opening = float(np.clip(opening, 0.0, 1.0))
        rows.append([ti, sp, q_true, q_indicated, mdot*1e6, sensor_lag,
                     opening*100, command*100, cfg["pin_bar"], cfg["pout_bar"]])
    return pd.DataFrame(rows, columns=["Time (s)", "Setpoint (NLPM)", "True flow (NLPM)",
        "Indicated flow (NLPM)", "Mass flow (mg/s)", "Bridge signal (V)",
        "Valve opening (%)", "Valve command (%)", "Inlet pressure (bar abs)", "Outlet pressure (bar abs)"])

st.title("Multi-gas Thermal MFC — Digital Twin")
st.caption("Educational model for seven selectable gases. The sensor transfer function and calibration data are illustrative, not manufacturer data or a certified instrument model.")

with st.sidebar:
    st.header("Gas and operating point")
    gas = st.selectbox("Gas", list(GASES.keys()), index=0)
    setpoint = st.slider("Flow setpoint (NLPM)", 0.0, 30.0, 5.0, 0.5)
    pin = st.number_input("Inlet pressure (bar absolute)", 1.01, 5.0, 2.0, 0.05)
    pout = st.number_input("Outlet pressure (bar absolute)", 1.0, 4.99, 1.8, 0.05)
    temp_c = st.number_input("Gas temperature (°C)", 0.0, 60.0, 20.0, 1.0)
    st.header("Thermal sensor (illustrative)")
    zero_v = st.number_input("Zero signal (V)", 0.0, 5.0, 0.5, 0.01)
    base_sensitivity = st.number_input("Base sensitivity", 0.001, 2.0, 0.08, 0.001, format="%.3f")
    nonlinearity = st.number_input("Nonlinearity coefficient", 0.0, 0.1, 0.0001, 0.0001, format="%.4f")
    noise_v = st.number_input("Noise standard deviation (V)", 0.0, 0.1, 0.002, 0.001, format="%.3f")
    sensor_tau = st.slider("Sensor time constant (s)", 0.02, 2.0, 0.20, 0.02)
    zero_drift_v = st.number_input("Zero drift (V)", -0.1, 0.1, 0.0, 0.001, format="%.3f")
    gain_error_pct = st.number_input("Gain error (%)", -10.0, 10.0, 0.0, 0.5)
    st.header("Valve and PID")
    valve_capacity = st.slider("Valve full-open capacity (NLPM at ΔP=0.2 bar)", 5.0, 50.0, 35.0, 1.0)
    valve_tau = st.slider("Valve time constant (s)", 0.05, 3.0, 0.35, 0.05)
    kp = st.slider("Kp", 0.0, 10.0, 2.0, 0.1)
    ki = st.slider("Ki", 0.0, 5.0, 0.5, 0.05)
    kd = st.slider("Kd", 0.0, 2.0, 0.0, 0.01)
    controller_gain = st.slider("Controller output gain", 0.001, 0.5, 0.035, 0.001)
    st.header("Simulation")
    duration = st.slider("Duration (s)", 10, 180, 60, 5)
    dt = st.select_slider("Time step (s)", [0.01, 0.02, 0.05, 0.1], value=0.05)
    initial_setpoint = st.number_input("Initial setpoint (NLPM)", 0.0, 30.0, 0.0, 0.5)
    step_time = st.number_input("Setpoint step time (s)", 0.0, 120.0, 2.0, 0.5)
    initial_opening = st.slider("Initial valve opening (%)", 0, 100, 5)
    max_flow = 30.0

if pin <= pout:
    st.error("Inlet pressure must be greater than outlet pressure.")
    st.stop()

sensor_params = dict(zero_v=zero_v, base_sensitivity=base_sensitivity,
                     nonlinearity=nonlinearity, noise_v=noise_v)

st.subheader("Gas property reference")
prop = GASES[gas]
a,b,c = st.columns(3)
a.metric("Molecular weight", f'{prop["MW"]*1000:.3f} g/mol')
b.metric("Specific heat (approx.)", f'{prop["cp"]:.1f} J/kg/K')
c.metric("Thermal conductivity (approx.)", f'{prop["k"]:.4f} W/m/K')
st.caption("Properties are approximate teaching values near room temperature; use a validated property source for engineering design.")

st.subheader("Gas-specific calibration")
st.write("The example below generates a synthetic calibration curve for the selected gas. For real use, replace it with traceable reference measurements for that sensor, gas, pressure and temperature.")
q_cal = np.linspace(0.5, 30.0, 25)
cal_coeff, cal_signals = fit_gas_calibration(gas, sensor_params, q_cal)
cal_df = pd.DataFrame({"Reference flow (NLPM)": q_cal, "Synthetic bridge signal (V)": cal_signals})
with st.expander("View synthetic calibration points"):
    st.dataframe(cal_df, use_container_width=True, hide_index=True)
    st.download_button("Download calibration template CSV",
        pd.DataFrame({"Gas": [gas]*len(q_cal), "Reference flow NLPM": q_cal,
                      "Measured bridge signal V": [""]*len(q_cal),
                      "Pressure bar abs": [pin]*len(q_cal), "Temperature C": [temp_c]*len(q_cal)}).to_csv(index=False).encode(),
        f"{gas.lower()}_calibration_template.csv", "text/csv")

if st.button("▶ Run multi-gas simulation", type="primary", use_container_width=True):
    np.random.seed(11)
    cfg = dict(gas=gas, setpoint=setpoint, pin_bar=pin, pout_bar=pout,
        temperature_c=temp_c, zero_v=zero_v, base_sensitivity=base_sensitivity,
        nonlinearity=nonlinearity, noise_v=noise_v, sensor_tau=sensor_tau,
        zero_drift_v=zero_drift_v, gain_error_pct=gain_error_pct,
        valve_capacity=valve_capacity, valve_tau=valve_tau, kp=kp, ki=ki, kd=kd,
        controller_gain=controller_gain, duration=duration, dt=dt,
        initial_setpoint=initial_setpoint, step_time=step_time,
        initial_opening=initial_opening, i_limit=100.0, max_flow=max_flow)
    df = simulate(cfg, cal_coeff)
    st.session_state["mg_df"] = df
    st.session_state["mg_gas"] = gas

if "mg_df" in st.session_state:
    df = st.session_state["mg_df"]
    final = df.iloc[-1]
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Final true flow", f'{final["True flow (NLPM)"]:.2f} NLPM')
    m2.metric("Final indicated flow", f'{final["Indicated flow (NLPM)"]:.2f} NLPM')
    m3.metric("Final valve opening", f'{final["Valve opening (%)"]:.1f}%')
    m4.metric("Final mass flow", f'{final["Mass flow (mg/s)"]:.3f} mg/s')

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df["Time (s)"], df["Setpoint (NLPM)"], "--", label="Setpoint")
    ax.plot(df["Time (s)"], df["True flow (NLPM)"], label="True flow")
    ax.plot(df["Time (s)"], df["Indicated flow (NLPM)"], label="Indicated flow")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("NLPM"); ax.grid(True); ax.legend()
    st.pyplot(fig)

    c1,c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(df["Time (s)"], df["Valve opening (%)"])
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Valve opening (%)"); ax.grid(True)
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(df["Time (s)"], df["Bridge signal (V)"])
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Bridge signal (V)"); ax.grid(True)
        st.pyplot(fig)

    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download run results CSV", df.to_csv(index=False).encode(),
                       "multi_gas_mfc_results.csv", "text/csv")
else:
    st.info("Choose a gas and operating conditions, then click Run multi-gas simulation.")

st.warning("Validation is required")
