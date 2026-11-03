import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Digital MFC Simulator", page_icon="⚙️", layout="wide")

# ------------------------- Physical constants -------------------------
R = 8.314462618       # J/(mol K)
MW_AIR = 0.0289652    # kg/mol
GAMMA_AIR = 1.4
T_GAS = 293.15        # K
T_NORMAL = 273.15     # K
P_NORMAL = 101325.0   # Pa
RHO_NORMAL_AIR = P_NORMAL * MW_AIR / (R * T_NORMAL)

def gas_properties(gas):
    props = {
        "Air": (0.0289652, 1.4),
        "N₂": (0.0280134, 1.4),
        "CO₂": (0.0440095, 1.289),
        "H₂": (0.00201588, 1.405),
    }
    return props[gas]

def compressible_orifice_mass_flow(p1, p2, temperature, area, cd, mw, gamma):
    """Ideal-gas compressible orifice/nozzle model; pressures in Pa absolute."""
    if p1 <= p2 or area <= 0:
        return 0.0
    ratio = max(0.0, min(p2 / p1, 1.0))
    critical_ratio = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
    factor = cd * area * p1 * np.sqrt(gamma * mw / (R * temperature))
    if ratio <= critical_ratio:  # choked flow
        term = (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
        return factor * term
    bracket = (2.0 * gamma / (gamma - 1.0)) * (
        ratio ** (2.0 / gamma) - ratio ** ((gamma + 1.0) / gamma)
    )
    return factor * np.sqrt(max(bracket, 0.0))

def mass_to_nlpm(mass_flow, mw):
    molar_flow = mass_flow / mw
    return molar_flow * R * T_NORMAL / P_NORMAL * 60000.0

def simulate(cfg):
    dt = cfg["dt"]
    t = np.arange(0, cfg["duration"] + dt, dt)
    n = len(t)
    p1 = cfg["pin_bar"] * 1e5
    p2 = cfg["pout_bar"] * 1e5
    mw, gamma = gas_properties(cfg["gas"])
    # Fixed flow-sensing orifice: selected to give useful DP at 5 NLPM.
    sensor_orifice_d = cfg["sensor_orifice_mm"] / 1000.0
    sensor_area = np.pi * sensor_orifice_d**2 / 4
    sensor_cd = cfg["sensor_cd"]
    # Valve full-open equivalent area; opening fraction scales area nonlinearly.
    valve_d = cfg["valve_diameter_mm"] / 1000.0
    valve_area_full = np.pi * valve_d**2 / 4
    valve_cd = cfg["valve_cd"]

    # Controller state
    opening = cfg["initial_opening"] / 100.0
    integral = 0.0
    prev_error = 0.0
    measured = 0.0
    rows = []

    # Calibration coefficient is generated from the ideal sensing-orifice model.
    # Use a reference sweep at the same gas and temperature.
    q_cal = np.linspace(0.5, 10.0, 20)
    dp_cal = []
    for q in q_cal:
        m = (q / 60000.0) * P_NORMAL / (R * T_NORMAL) * mw
        # Low-Mach differential-pressure approximation for calibration element.
        rho = p1 * mw / (R * T_GAS)
        dp = 0.5 * rho * (m / max(rho * sensor_cd * sensor_area, 1e-12))**2
        dp_cal.append(dp)
    dp_cal = np.asarray(dp_cal)
    # Fit Q = K*sqrt(dp); a fitted calibration coefficient, not a universal constant.
    k_cal = np.sum(np.sqrt(dp_cal) * q_cal) / np.sum(dp_cal)

    for i, ti in enumerate(t):
        setpoint = cfg["setpoint"] if ti >= cfg["step_time"] else cfg["initial_setpoint"]
        # Valve has first-order actuator dynamics.
        # Valve flow area is proportional to opening^1.5 (illustrative valve characteristic).
        effective_area = valve_area_full * max(opening, 0.0)**1.5
        m_valve = compressible_orifice_mass_flow(
            p1, p2, T_GAS, effective_area, valve_cd, mw, gamma
        )
        q_true = mass_to_nlpm(m_valve, mw)

        # Flow element pressure drop; low-Mach approximation used for sensing.
        rho = p1 * mw / (R * T_GAS)
        q_actual_m3s = (q_true / 60000.0) * (P_NORMAL / (R * T_NORMAL)) * (R * T_GAS / p1)
        velocity_term = q_actual_m3s / max(sensor_cd * sensor_area, 1e-12)
        dp_true = 0.5 * rho * velocity_term**2

        # Virtual differential-pressure sensor with offset, gain error and white noise.
        dp_sensor = max(0.0, dp_true * (1.0 + cfg["sensor_gain_error"]/100.0)
                        + cfg["sensor_offset_pa"]
                        + np.random.normal(0.0, cfg["sensor_noise_pa"]))
        q_measured = max(0.0, k_cal * np.sqrt(dp_sensor))

        error = setpoint - q_measured
        if i > 0:
            integral += error * dt
        integral = np.clip(integral, -cfg["integral_limit"], cfg["integral_limit"])
        derivative = (error - prev_error) / dt if i > 0 else 0.0
        control = cfg["kp"] * error + cfg["ki"] * integral + cfg["kd"] * derivative
        prev_error = error

        # Controller output is an opening command. Clamp and apply anti-windup.
        command = np.clip(opening + control * cfg["controller_gain"] * dt, 0.0, 1.0)
        opening += (command - opening) * min(1.0, dt / max(cfg["actuator_tau"], 1e-3))
        opening = float(np.clip(opening, 0.0, 1.0))

        rows.append((ti, setpoint, q_true, q_measured, dp_true, dp_sensor,
                     opening*100, command*100, p1/1e5, p2/1e5))
    return pd.DataFrame(rows, columns=[
        "Time (s)", "Setpoint (NLPM)", "True flow (NLPM)", "Measured flow (NLPM)",
        "True sensor ΔP (Pa)", "Sensor ΔP (Pa)", "Valve opening (%)",
        "Valve command (%)", "Inlet pressure (bar abs)", "Outlet pressure (bar abs)"
    ]), q_cal, dp_cal, k_cal

st.title("⚙️ Digital Mass Flow Controller (MFC) Simulator")
st.caption("Educational digital twin: compressible gas valve model + virtual differential-pressure sensor + PI/PID feedback. Not a certified instrument design.")

with st.sidebar:
    st.header("Operating conditions")
    gas = st.selectbox("Gas", ["Air", "N₂", "CO₂", "H₂"], index=0)
    setpoint = st.slider("Flow setpoint (NLPM)", 0.0, 10.0, 5.0, 0.1)
    pin = st.number_input("Inlet pressure (bar absolute)", 1.01, 10.0, 2.0, 0.05)
    pout = st.number_input("Outlet pressure (bar absolute)", 1.0, 9.9, 1.8, 0.05)
    temperature_c = st.number_input("Gas temperature (°C)", -20.0, 100.0, 20.0, 1.0)
    st.header("Valve and sensing element")
    valve_diameter_mm = st.slider("Valve equivalent diameter (mm)", 1.0, 8.0, 4.0, 0.1)
    valve_cd = st.slider("Valve discharge coefficient", 0.2, 0.9, 0.70, 0.01)
    sensor_orifice_mm = st.slider("Sensor orifice diameter (mm)", 1.0, 6.0, 3.2, 0.1)
    sensor_cd = st.slider("Sensor orifice discharge coefficient", 0.2, 0.9, 0.61, 0.01)
    st.header("Controller")
    kp = st.slider("Kp", 0.0, 10.0, 2.0, 0.1)
    ki = st.slider("Ki", 0.0, 5.0, 0.5, 0.05)
    kd = st.slider("Kd", 0.0, 2.0, 0.0, 0.01)
    controller_gain = st.slider("Controller output gain", 0.01, 2.0, 0.35, 0.01)
    actuator_tau = st.slider("Actuator time constant (s)", 0.05, 3.0, 0.35, 0.05)
    st.header("Virtual sensor imperfections")
    sensor_noise_pa = st.slider("Sensor noise standard deviation (Pa)", 0.0, 30.0, 2.0, 0.5)
    sensor_offset_pa = st.slider("Sensor zero offset (Pa)", -50.0, 50.0, 0.0, 0.5)
    sensor_gain_error = st.slider("Sensor gain error (%)", -10.0, 10.0, 0.0, 0.5)
    st.header("Simulation")
    duration = st.slider("Duration (s)", 10, 120, 40, 5)
    dt = st.select_slider("Time step (s)", options=[0.01, 0.02, 0.05, 0.1], value=0.05)
    initial_setpoint = st.number_input("Initial setpoint before step (NLPM)", 0.0, 10.0, 0.0, 0.1)
    step_time = st.number_input("Setpoint step time (s)", 0.0, 60.0, 2.0, 0.5)
    initial_opening = st.slider("Initial valve opening (%)", 0, 100, 10, 1)
    integral_limit = st.number_input("Integral clamp", 1.0, 1000.0, 100.0, 1.0)

if pin <= pout:
    st.error("Inlet pressure must be greater than outlet pressure for forward flow.")
    st.stop()

if st.button("▶ Run simulation", type="primary", use_container_width=True):
    np.random.seed(7)
    cfg = dict(gas=gas, setpoint=setpoint, pin_bar=pin, pout_bar=pout,
               temperature=T_GAS + (temperature_c - 20.0), kp=kp, ki=ki, kd=kd,
               controller_gain=controller_gain, actuator_tau=actuator_tau,
               sensor_noise_pa=sensor_noise_pa, sensor_offset_pa=sensor_offset_pa,
               sensor_gain_error=sensor_gain_error, duration=duration, dt=dt,
               initial_setpoint=initial_setpoint, step_time=step_time,
               initial_opening=initial_opening, integral_limit=integral_limit,
               valve_diameter_mm=valve_diameter_mm, valve_cd=valve_cd,
               sensor_orifice_mm=sensor_orifice_mm, sensor_cd=sensor_cd)
    df, q_cal, dp_cal, k_cal = simulate(cfg)
    st.session_state["simulation"] = df
    st.session_state["calibration"] = (q_cal, dp_cal, k_cal, cfg)

if "simulation" in st.session_state:
    df = st.session_state["simulation"]
    q_cal, dp_cal, k_cal, cfg = st.session_state["calibration"]
    last = df.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final true flow", f'{last["True flow (NLPM)"]:.2f} NLPM')
    c2.metric("Final measured flow", f'{last["Measured flow (NLPM)"]:.2f} NLPM')
    c3.metric("Final valve opening", f'{last["Valve opening (%)"]:.1f}%')
    c4.metric("Calibration K", f"{k_cal:.4f}")

    st.subheader("Flow response")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Time (s)"], df["Setpoint (NLPM)"], "--", label="Setpoint")
    ax.plot(df["Time (s)"], df["True flow (NLPM)"], label="True flow")
    ax.plot(df["Time (s)"], df["Measured flow (NLPM)"], alpha=0.8, label="Measured flow")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Flow (NLPM)")
    ax.grid(True); ax.legend(); st.pyplot(fig)

    left, right = st.columns(2)
    with left:
        st.subheader("Valve response")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.plot(df["Time (s)"], df["Valve opening (%)"])
        ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Opening (%)")
        ax2.grid(True); st.pyplot(fig2)
    with right:
        st.subheader("Differential pressure")
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        ax3.plot(df["Time (s)"], df["True sensor ΔP (Pa)"], label="True ΔP")
        ax3.plot(df["Time (s)"], df["Sensor ΔP (Pa)"], alpha=0.7, label="Sensor ΔP")
        ax3.set_xlabel("Time (s)"); ax3.set_ylabel("ΔP (Pa)")
        ax3.grid(True); ax3.legend(); st.pyplot(fig3)

    st.subheader("Calibration curve")
    fig4, ax4 = plt.subplots(figsize=(8, 3.5))
    ax4.scatter(np.sqrt(dp_cal), q_cal, label="Synthetic calibration points")
    xfit = np.linspace(0, np.sqrt(max(dp_cal))*1.05, 100)
    ax4.plot(xfit, k_cal*xfit, label=f"Q = {k_cal:.4f} √ΔP")
    ax4.set_xlabel("√Differential pressure (√Pa)")
    ax4.set_ylabel("Flow (NLPM)")
    ax4.grid(True); ax4.legend(); st.pyplot(fig4)

    st.subheader("Simulation data")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download simulation CSV", df.to_csv(index=False).encode("utf-8"),
                       "mfc_simulation_results.csv", "text/csv")
    st.info("Model note: the valve uses an ideal-gas compressible-orifice equation. The virtual flow sensor uses a simplified low-Mach differential-pressure relationship and a synthetic calibration sweep. The simulator is educational, not a validated design tool. It does not model real MFC thermal sensor physics, detailed valve Cv data, regulator dynamics, or certified accuracy.")
else:
    st.info("Set your operating conditions in the sidebar and click **Run simulation**.")
    st.markdown("""
    **Default case:** Air, 5 NLPM setpoint, 2.0 bar(a) inlet, 1.8 bar(a) outlet,
    Kp = 2.0 and Ki = 0.5.

    Try changing the setpoint, pressure difference, valve diameter, sensor offset,
    and PID settings to see how the simulated controller responds.
    """)

