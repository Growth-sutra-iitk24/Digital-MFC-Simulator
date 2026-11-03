# Digital-MFC-Simulator

# Multi-gas Thermal MFC Digital Twin — Streamlit deployment

This repository contains an educational Streamlit digital twin for a selectable-gas
thermal mass-flow controller. It includes Air, N2, CO2, H2, CH4, CO and Ar; a
synthetic gas-specific sensor response/calibration model; valve dynamics; PI/PID
control; plots; and CSV exports.

## Deploy on Streamlit Community Cloud

1. Download and extract this project ZIP.
2. Create a new **public or private GitHub repository** (depending on your plan and
   repository access settings).
3. Upload all files and folders from this project to the repository root. Ensure
   `app.py` and `requirements.txt` are at the root and `.streamlit/config.toml`
   remains inside the `.streamlit` folder.
4. Sign in to Streamlit Community Cloud with GitHub.
5. Select **Create app** / **Deploy an app**.
6. Choose your repository, branch (usually `main`) and main file path `app.py`.
7. Deploy. Streamlit will install packages from `requirements.txt`.
8. Open the generated app URL and test each gas and flow setpoint.

Official deployment guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## Run locally first (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Repository structure

```text
.
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── config.toml
```

## Important scientific and safety limitations

This is a software-only educational digital twin. The thermal bridge response and
gas-specific calibration curves are illustrative and synthetic, not manufacturer
transfer functions or traceable experimental calibrations. The valve model is a
simplified teaching model. The application does not establish the accuracy,
pressure rating, material compatibility, or safety of any physical MFC.

Do not use this simulation to design, certify, or operate real gas equipment.
Hydrogen and methane are flammable; carbon monoxide is acutely toxic; carbon dioxide
can cause asphyxiation. Physical systems require manufacturer-approved components,
appropriate gas detection/ventilation, pressure protection, qualified assembly and
traceable gas-specific calibration.
