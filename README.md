# Digital-MFC-Simulator

Digital MFC Simulator
Educational Streamlit simulator for a low-pressure gas flow controller.
Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:
pip install -r requirements.txt
5. Launch:
streamlit run app.py
Default case
- Gas: Air
- Flow setpoint: 5 NLPM
- Inlet pressure: 2.0 bar absolute
- Outlet pressure: 1.8 bar absolute
- PI gains: Kp = 2.0, Ki = 0.5
Model scope and limitations
This is an educational digital model, not a certified MFC design or calibration system.
It uses an ideal-gas compressible-orifice equation for the control valve, a simplified
low-Mach differential-pressure flow element, a virtual sensor with configurable noise,
offset and gain error, and a discrete PI/PID controller. The calibration curve is
synthetic and generated from the same simplified sensor model; it is not traceable
calibration. It does not reproduce the thermal mass sensing principle of a Brooks
SLA5800 and must not be used to size or operate real high-pressure equipment.

For real equipment, use manufacturer data, pressure-rated components, and traceable
calibration performed by a qualified laboratory.
