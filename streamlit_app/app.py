from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import altair as alt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as _components
from scipy.stats import pearsonr, spearmanr

# Ensures correct imports when launched with:
# streamlit run streamlit_app/app.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multiagent_pipeline.main import run_pipeline
from multiagent_pipeline.config import get_anthropic_api_key


st.set_page_config(
    page_title="Airport Risk Intelligence",
    page_icon=":airplane_departure:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────── IATA Coordinates ────────────────────────────
# (lat, lon) for every airport that appears in the dataset routes
IATA_COORDS: dict[str, tuple[float, float]] = {
    # ── Italian arrival airports ──────────────────────────────────────────────
    "AOI": (43.6163, 13.3622), "BDS": (40.6576, 17.9470), "BGY": (45.6739,  9.7046),
    "BLQ": (44.5354, 11.2887), "BRI": (41.1389, 16.7606), "BZO": (46.4603, 11.3264),
    "CAG": (39.2515,  9.0543), "CIA": (41.7994, 12.5949), "CIY": (36.9962, 14.6071),
    "CTA": (37.4667, 15.0664), "CUF": (44.5470,  7.6232), "FCO": (41.8003, 12.2389),
    "FLR": (43.8100, 11.2051), "GOA": (44.4133,  8.8375), "LIN": (45.4454,  9.2767),
    "MXP": (45.6306,  8.7281), "NAP": (40.8860, 14.2910), "OLB": (40.8987,  9.5176),
    "PEG": (43.0991, 12.5132), "PMF": (44.8246, 10.2964), "PMO": (38.1796, 13.0913),
    "PSA": (43.6836, 10.3927), "PSR": (42.4317, 14.1811), "REG": (38.0712, 15.6516),
    "RMI": (44.0204, 12.6117), "SUF": (38.9057, 16.2423), "TRN": (45.2009,  7.6497),
    "TRS": (45.8275, 13.4722), "TSF": (45.6484, 12.1944), "VBS": (45.4282, 10.3306),
    "VCE": (45.5053, 12.3519), "VRN": (45.3957, 10.8885),
    # ── International departure airports ─────────────────────────────────────
    "ABJ": ( 5.2613,  -3.9262), "ADB": (38.2924,  27.1570), "ADD": ( 8.9779,  38.7993),
    "ADL": (-34.945, 138.5311), "AER": (43.4499,  39.9566), "AGA": (30.3250,  -9.4130),
    "AKL": (-37.008, 174.7917), "ALA": (43.3521,  77.0405), "ALG": (36.6910,   3.2154),
    "AMD": (23.0772,  72.6347), "AMM": (31.7226,  35.9932), "ANU": (17.1368, -61.7927),
    "ARN": (59.6519,  17.9186), "ASB": (37.9868,  58.3610), "ASM": (15.2954,  38.9107),
    "ATL": (33.6407, -84.4277), "ATQ": (31.7096,  74.7973), "AUH": (24.4330,  54.6511),
    "AYT": (36.8987,  30.7995), "BAH": (26.2708,  50.6336), "BEG": (44.8184,  20.3091),
    "BEY": (33.8208,  35.4884), "BFS": (54.6575,  -6.2158), "BHX": (52.4539,  -1.7480),
    "BKK": (13.6811, 100.7470), "BLR": (13.1979,  77.7063), "BNA": (36.1245, -86.6782),
    "BNE": (-27.384, 153.1175), "BOG": ( 4.7016, -74.1469), "BOS": (42.3656, -71.0096),
    "BQH": (51.3382,   0.0313), "BRS": (51.3827,  -2.7191), "BSB": (-15.871, -47.9186),
    "BUF": (42.9405, -78.7322), "CAI": (30.1219,  31.4056), "CAN": (23.3924, 113.2988),
    "CEB": (10.3075, 123.9790), "CGK": (-6.1256, 106.6558), "CGO": (34.5197, 113.8408),
    "CKG": (29.7192, 106.6417), "CLE": (41.4117, -81.8498), "CMN": (33.3675,  -7.5897),
    "CUN": (21.0365, -86.8771), "CVG": (39.0489, -84.6678), "DAR": (-6.8781,  39.2026),
    "DEL": (28.5665,  77.1031), "DFW": (32.8998, -97.0403), "DMM": (26.4712,  49.7979),
    "DOH": (25.2607,  51.6138), "DPS": (-8.7483, 115.1670), "DSS": (14.6700, -17.0727),
    "DTW": (42.2124, -83.3534), "DUR": (-29.614,  31.1197), "DWC": (24.8963,  55.1614),
    "DXB": (25.2532,  55.3657), "EDI": (55.9500,  -3.3725), "ELQ": (26.3023,  43.7742),
    "EMA": (52.8311,  -1.3280), "ESB": (40.1281,  32.9951), "EVN": (40.1473,  44.3959),
    "EWR": (40.6925, -74.1687), "EZE": (-34.822, -58.5358), "FAB": (51.2788,  -0.7764),
    "FEZ": (33.9273,  -4.9779), "FIH": (-4.3857,  15.4446), "FLL": (26.0726, -80.1527),
    "FRA": (50.0379,   8.5622), "FRU": (43.0613,  74.4776), "FUK": (33.5859, 130.4508),
    "GIG": (-22.810, -43.2505), "GLA": (55.8642,  -4.4330), "GRU": (-23.436, -46.4731),
    "GYD": (40.4675,  50.0467), "GZT": (36.9473,  37.4786), "HAN": (21.2212, 105.8072),
    "HGH": (30.2295, 120.4298), "HKG": (22.3080, 113.9185), "HKT": ( 8.1132,  98.3169),
    "HND": (35.5494, 139.7798), "HRG": (27.1783,  33.7994), "IAD": (38.9531, -77.4565),
    "IAH": (29.9844, -95.3414), "ICN": (37.4691, 126.4510), "IFN": (32.7508,  51.8613),
    "IKA": (35.4161,  51.1522), "ISB": (33.6167,  73.1000), "ISL": (40.9769,  28.8146),
    "IST": (41.2758,  28.7519), "JAX": (30.4941, -81.6879), "JED": (21.6796,  39.1565),
    "JFK": (40.6413, -73.7781), "KBL": (34.5659,  69.2123), "KCH": ( 1.4847, 110.3373),
    "KIV": (46.9277,  28.9305), "KUL": ( 2.7456, 101.7099), "KUT": (42.1763,  42.4826),
    "KWI": (29.2267,  47.9689), "KZN": (55.6062,  49.2787), "LAD": (-8.8587,  13.2312),
    "LAS": (36.0840, -115.154), "LAX": (33.9425,-118.4081), "LBA": (53.8659,  -1.6606),
    "LCY": (51.5053,   0.0553), "LGW": (51.1537,  -0.1821), "LHR": (51.4706,  -0.4619),
    "LOS": ( 6.5774,   3.3212), "LPL": (53.3336,  -2.8497), "LRM": (18.2742, -69.1683),
    "LTN": (51.8747,  -0.3683), "LXR": (25.6712,  32.7066), "LYX": (50.9561,   0.9392),
    "MAN": (53.3537,  -2.2750), "MBA": (-4.0348,  39.5942), "MBJ": (18.5037, -77.9134),
    "MCT": (23.5932,  58.2844), "MED": (24.5534,  39.7051), "MEL": (-37.669, 144.8410),
    "MEX": (19.4363, -99.0721), "MHD": (36.2352,  59.6411), "MIA": (25.7959, -80.2870),
    "MJI": (32.8943,  13.2791), "MLE": ( 4.1918,  73.5290), "MNL": (14.5086, 121.0194),
    "MPM": (-25.921,  32.5726), "MRU": (-20.430,  57.6836), "MYR": (33.6797, -78.9283),
    "NAV": (38.9192,  34.5346), "NBO": (-1.3192,  36.9275), "NCL": (55.0375,  -1.6917),
    "NHT": (51.5533,  -0.4186), "NKG": (31.7420, 118.8620), "NOS": (-13.312,  48.3148),
    "NRT": (35.7648, 140.3864), "NSI": ( 3.7226,  11.5531), "OAK": (37.7213,-122.2208),
    "ORD": (41.9742, -87.9073), "OUA": (12.3532,  -1.5124), "OXB": (11.8948, -15.6537),
    "OXF": (51.8368,  -1.3200), "PEK": (40.0799, 116.6031), "PER": (-31.940, 115.9669),
    "PEW": (33.9939,  71.5146), "PHL": (39.8729, -75.2437), "PKX": (39.5097, 116.4104),
    "POA": (-29.994, -51.1714), "PRN": (42.5728,  21.0358), "PVG": (31.1443, 121.8083),
    "PVR": (20.6801,-105.2544), "RAK": (31.6069,  -8.0363), "RBA": (34.0510,  -6.7516),
    "REC": (-8.1265, -34.9232), "RMF": (23.4353,  36.2856), "RMO": (50.3889,  28.7383),
    "RUH": (24.9578,  46.6989), "SAW": (40.8985,  29.3092), "SCL": (-33.393, -70.7858),
    "SDU": (-22.911, -43.1631), "SGN": (10.8188, 106.6520), "SHJ": (25.3286,  55.5172),
    "SID": (16.7414, -22.9494), "SIN": ( 1.3644, 103.9915), "SJJ": (43.8246,  18.3315),
    "SKG": (40.5197,  22.9709), "SKP": (41.9616,  21.6214), "SKT": (32.5356,  74.3636),
    "SLL": (17.0386,  54.0913), "SMF": (38.6954,-121.5908), "SOF": (42.6967,  23.4114),
    "SPX": (30.0792,  31.0111), "SSA": (-12.909, -38.3225), "SSH": (27.9773,  34.3950),
    "STN": (51.8860,   0.2389), "SVO": (55.9726,  37.4146), "SYD": (-33.940, 151.1753),
    "SYR": (43.1112, -76.1063), "SYZ": (29.5392,  52.5898), "SZX": (22.6393, 113.8107),
    "TAS": (41.2579,  69.2813), "TBS": (41.6692,  44.9547), "TFU": (30.5795, 103.8966),
    "TGD": (42.3594,  19.2519), "TIA": (41.4147,  19.7206), "TLV": (32.0114,  34.8867),
    "TNG": (35.7269,  -5.9169), "TPA": (27.9755, -82.5332), "TPE": (25.0777, 121.2322),
    "TSA": (25.0694, 121.5524), "TUN": (36.8510,  10.2272), "VKO": (55.5965,  37.2615),
    "WNZ": (27.9122, 120.6519), "YEG": (53.3097,-113.5797), "YHZ": (44.8808, -63.5086),
    "YUL": (45.4706, -73.7408), "YVR": (49.1947,-123.1792), "YWG": (49.9100, -97.2398),
    "YYC": (51.1315,-114.0106), "YYZ": (43.6777, -79.6248), "ZNZ": (-6.2220,  39.2248),
}


# ──────────────────────────── Shared CSS ─────────────────────────────────────

def _inject_style() -> None:
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">

        <style>
          /* ── Base & typography ─────────────────────────────────────── */
          html, body, [class*="css"] {
            font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif !important;
          }
          .block-container {
            padding-top: 1.4rem !important;
            padding-bottom: 2rem !important;
            max-width: 1280px !important;
          }
          h1, h2, h3 {
            font-family: 'Inter', system-ui, sans-serif !important;
            letter-spacing: -0.3px !important;
            font-weight: 800 !important;
          }

          /* ── Sidebar ───────────────────────────────────────────────── */
          [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #060d1a 0%, #0a1220 100%) !important;
            border-right: 1px solid rgba(14,165,233,0.1) !important;
          }
          [data-testid="stSidebar"] h1,
          [data-testid="stSidebar"] h2,
          [data-testid="stSidebar"] h3,
          [data-testid="stSidebar"] label {
            color: #cbd5e1 !important;
          }
          /* Primary run button */
          [data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"],
          [data-testid="stSidebar"] div[data-testid="stButton"] button {
            background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            font-size: 0.92rem !important;
            letter-spacing: 0.2px !important;
            color: #fff !important;
            box-shadow: 0 0 20px rgba(14,165,233,0.25), 0 4px 12px rgba(0,0,0,0.3) !important;
            transition: box-shadow 0.2s ease, transform 0.15s ease !important;
          }
          [data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
            box-shadow: 0 0 32px rgba(14,165,233,0.45), 0 6px 20px rgba(0,0,0,0.35) !important;
            transform: translateY(-1px) !important;
          }

          /* ── Metric cards ──────────────────────────────────────────── */
          [data-testid="metric-container"] {
            background: linear-gradient(135deg,
              rgba(14,165,233,0.07) 0%,
              rgba(15,23,42,0.92) 100%) !important;
            border: 1px solid rgba(14,165,233,0.18) !important;
            border-radius: 12px !important;
            padding: 14px 18px !important;
            backdrop-filter: blur(8px) !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
          }
          [data-testid="metric-container"]:hover {
            border-color: rgba(14,165,233,0.35) !important;
            box-shadow: 0 0 20px rgba(14,165,233,0.1) !important;
          }
          [data-testid="stMetricLabel"] {
            font-size: 0.7rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.8px !important;
            color: #64748b !important;
          }
          [data-testid="stMetricValue"] {
            font-size: 1.6rem !important;
            font-weight: 800 !important;
            color: #f1f5f9 !important;
          }

          /* ── Tabs ──────────────────────────────────────────────────── */
          .stTabs [data-baseweb="tab-list"] {
            background: rgba(6,13,26,0.9) !important;
            border-radius: 12px !important;
            padding: 4px 5px !important;
            border: 1px solid rgba(30,41,59,0.7) !important;
            gap: 2px !important;
          }
          .stTabs [data-baseweb="tab"] {
            border-radius: 8px !important;
            font-weight: 500 !important;
            font-size: 0.83rem !important;
            color: #475569 !important;
            padding: 7px 14px !important;
            transition: color 0.15s ease, background 0.15s ease !important;
          }
          .stTabs [data-baseweb="tab"]:hover {
            color: #94a3b8 !important;
            background: rgba(30,41,59,0.6) !important;
          }
          .stTabs [aria-selected="true"] {
            background: rgba(14,165,233,0.12) !important;
            color: #38bdf8 !important;
            border: 1px solid rgba(14,165,233,0.25) !important;
          }
          .stTabs [data-baseweb="tab-panel"] {
            padding-top: 20px !important;
          }

          /* ── Dataframes ────────────────────────────────────────────── */
          [data-testid="stDataFrame"] {
            border-radius: 10px !important;
            overflow: hidden !important;
            border: 1px solid rgba(30,41,59,0.7) !important;
          }

          /* ── Status / alerts ───────────────────────────────────────── */
          [data-testid="stAlert"] {
            border-radius: 10px !important;
          }
          [data-testid="stStatusWidget"] {
            border-radius: 12px !important;
          }

          /* ── Expanders ─────────────────────────────────────────────── */
          [data-testid="stExpander"] {
            border: 1px solid rgba(30,41,59,0.6) !important;
            border-radius: 10px !important;
            background: rgba(15,23,42,0.5) !important;
          }

          /* ── Download button ───────────────────────────────────────── */
          [data-testid="stDownloadButton"] > button {
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 0.82rem !important;
            border: 1px solid rgba(30,41,59,0.8) !important;
            color: #94a3b8 !important;
            background: rgba(15,23,42,0.8) !important;
            transition: border-color 0.2s, color 0.2s !important;
          }
          [data-testid="stDownloadButton"] > button:hover {
            border-color: rgba(14,165,233,0.4) !important;
            color: #38bdf8 !important;
          }

          /* ── Selectbox / widgets ───────────────────────────────────── */
          [data-testid="stSelectbox"] > div > div {
            border-radius: 8px !important;
            border-color: rgba(30,41,59,0.8) !important;
          }

          /* ── Custom classes ────────────────────────────────────────── */
          .chip {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 600;
            margin-right: 6px;
            letter-spacing: 0.3px;
          }
          .ok  {
            background: rgba(34,197,94,0.1);
            border: 1px solid rgba(34,197,94,0.3);
            color: #4ade80;
          }
          .err {
            background: rgba(239,68,68,0.1);
            border: 1px solid rgba(239,68,68,0.3);
            color: #f87171;
          }
          .section-card {
            border: 1px solid rgba(30,41,59,0.7);
            border-radius: 12px;
            padding: 14px 18px;
            background: linear-gradient(135deg, rgba(14,165,233,0.05), rgba(15,23,42,0.9));
            font-size: 0.88rem;
            color: #94a3b8;
          }
          .section-card b { color: #cbd5e1; }

          /* ── Dividers ──────────────────────────────────────────────── */
          hr {
            border-color: rgba(30,41,59,0.6) !important;
            margin: 16px 0 !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────── Pipeline graph HTML ──────────────────────────────

_PIPELINE_AGENTS = [
    ("data",     "DataAgent",          "#0ea5e9", "Loads, filters & 54 features"),
    ("baseline", "BaselineAgent",      "#f59e0b", "Robust MAD z-scores"),
    ("outlier",  "OutlierAgent",       "#ef4444", "4-model weighted ensemble"),
    ("risk",     "RiskProfilingAgent", "#ec4899", "5 business rules → final_risk"),
    ("report",   "ReportAgent",        "#a855f7", "LLM narrative (optional)"),
]


def _render_pipeline_graph_html(active_step: int, stage_errors: dict | None = None) -> str:
    """
    Renders the 5-agent pipeline as a pure HTML/CSS card row.

    active_step:
      -1  → nothing started
       0  → DataAgent running
       1  → BaselineAgent running (DataAgent done)
       2  → OutlierAgent running
       3  → RiskProfilingAgent running
       4  → ReportAgent running
       5  → all done
    """
    errors = stage_errors or {}

    # Build CSS keyframes for each agent's pulse animation
    css_keyframes = ""
    for agent_id, _, color, _ in _PIPELINE_AGENTS:
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        css_keyframes += f"""
        @keyframes pulse-{agent_id} {{
          0%,100% {{ box-shadow: 0 0 6px rgba({r},{g},{b},0.3); }}
          50%      {{ box-shadow: 0 0 22px rgba({r},{g},{b},0.9),
                                  0 0 44px rgba({r},{g},{b},0.4); }}
        }}"""

    cards_html = []
    for i, (agent_id, label, color, desc) in enumerate(_PIPELINE_AGENTS):
        has_error = agent_id in errors
        is_done   = (i < active_step) and not has_error
        is_active = (i == active_step) and not has_error
        is_pending = i > active_step and not has_error

        if has_error:
            border     = "#ef4444"
            bg         = "rgba(239,68,68,0.10)"
            label_col  = "#fca5a5"
            status_ico = '<span style="color:#ef4444;font-size:18px;line-height:1;">✗</span>'
            anim       = ""
        elif is_done:
            border     = color
            bg         = "rgba(15,23,42,0.85)"
            label_col  = "#e2e8f0"
            status_ico = '<span style="color:#22c55e;font-size:18px;line-height:1;">✓</span>'
            anim       = ""
        elif is_active:
            border     = color
            bg         = "rgba(15,23,42,0.95)"
            label_col  = "#f8fafc"
            status_ico = f'<span style="color:{color};font-size:15px;line-height:1;">▶</span>'
            anim       = f"animation: pulse-{agent_id} 1.8s ease-in-out infinite;"
        else:  # pending
            border     = "#1e293b"
            bg         = "rgba(15,23,42,0.55)"
            label_col  = "#475569"
            status_ico = '<span style="color:#334155;font-size:16px;line-height:1;">○</span>'
            anim       = ""

        card = f"""
        <div style="flex:1; min-width:100px; max-width:155px;
                    padding:13px 9px 11px; border-radius:12px;
                    border:2px solid {border}; background:{bg};
                    text-align:center; {anim}
                    transition:border-color 0.4s, background 0.4s;">
          <div style="font-size:8px; font-weight:700;
                      color:{color if not is_pending else '#1e293b'};
                      letter-spacing:1.5px; text-transform:uppercase;">
            Agent {i + 1}
          </div>
          <div style="font-size:12px; font-weight:800; color:{label_col};
                      margin:5px 0 2px; white-space:nowrap;">
            {label}
          </div>
          <div style="font-size:9px; line-height:1.35;
                      color:{'#64748b' if not is_pending else '#1e293b'};">
            {desc}
          </div>
          <div style="margin-top:9px;">{status_ico}</div>
        </div>"""
        cards_html.append(card)

        if i < len(_PIPELINE_AGENTS) - 1:
            arrow_color = color if (is_done or is_active) else "#1e293b"
            cards_html.append(
                f'<div style="color:{arrow_color}; font-size:18px; font-weight:300;'
                f' flex-shrink:0; padding:0 3px; align-self:center;'
                f' transition:color 0.4s;">→</div>'
            )

    return f"""
    <style>
    {css_keyframes}
    .agent-pipeline {{
      display: flex;
      align-items: stretch;
      gap: 6px;
      padding: 18px 16px;
      overflow-x: auto;
      background: linear-gradient(160deg, #020617 0%, #0f172a 100%);
      border-radius: 14px;
      border: 1px solid #1e293b;
      font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif;
    }}
    </style>
    <div class="agent-pipeline">
      {''.join(cards_html)}
    </div>
    """


# ─────────────────── Agent graph via React + Babel CDN ───────────────────────

def _render_agent_graph_html(active_step: int = -1, stage_errors: dict | None = None) -> str:
    """Embeds agent_graph.jsx via React 18 + Babel standalone (CDN).

    Patches the JSX source at runtime:
      - Replaces ES6 imports with UMD globals (React is a global on the page)
      - Injects _PIPELINE_STEP and _PIPELINE_ERRORS variables
      - Removes 'export default'
    """
    jsx_path = Path(__file__).parent / "agent_graph.jsx"
    try:
        jsx_content = jsx_path.read_text(encoding="utf-8")
    except Exception:
        return "<div style='color:#64748b;padding:20px;font-family:sans-serif;'>Agent graph unavailable.</div>"

    # Patch imports → UMD globals (React 18 UMD exposes window.React)
    jsx_content = jsx_content.replace(
        'import { useState, useEffect, useRef } from "react";',
        'const { useState, useEffect, useRef } = React;',
    )
    # Remove export default so the function is available globally
    jsx_content = jsx_content.replace(
        'export default function AgentGraph()',
        'function AgentGraph()',
    )
    # Inject active step into initial state so the graph starts at the right position
    jsx_content = jsx_content.replace(
        'const [activeStep, setActiveStep] = useState(-1);',
        'const [activeStep, setActiveStep] = useState(typeof _PIPELINE_STEP !== "undefined" ? _PIPELINE_STEP : -1);',
    )

    errors_json = json.dumps(stage_errors or {})

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://unpkg.com/react@18/umd/react.production.min.js" crossorigin></script>
<script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js" crossorigin></script>
<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 0; background: transparent; overflow-y: auto; }}
</style>
</head>
<body>
<div id="root"></div>
<script>
  var _PIPELINE_STEP = {active_step};
  var _PIPELINE_ERRORS = {errors_json};
</script>
<script type="text/babel">
{jsx_content}
ReactDOM.createRoot(document.getElementById('root')).render(<AgentGraph />);
</script>
</body>
</html>"""


# ─────────────────────── Live pipeline runner ─────────────────────────────────

def _run_pipeline_with_live_ui(
    perimeter: dict,
    run_report: bool,
    use_llm: bool,
    dry_run: bool,
    continue_on_error: bool,
    save_outputs: bool,
) -> tuple[dict, dict]:
    """
    Runs the pipeline agent-by-agent, updating the graph HTML between each step.
    Returns (final_state, summary) with the same structure as run_pipeline().
    """
    from multiagent_pipeline.agents.data_agent import data_agent_node
    from multiagent_pipeline.agents.baseline_agent import run_baseline_agent
    from multiagent_pipeline.agents.outlier_agent import run_outlier_agent
    from multiagent_pipeline.agents.risk_profiling_agent import run_risk_profiling_agent
    from multiagent_pipeline.agents.report_agent import run_report_agent

    # ── Initial state ────────────────────────────────────────────────────────
    state: dict = {
        "perimeter":      perimeter or {},
        "df_raw":         None, "df_allarmi":    None,
        "df_viaggiatori": None, "data_meta":     None,
        "df_features":    None, "feature_meta":  None,
        "df_baseline":    None, "baseline_meta": None,
        "df_anomalies":   None, "anomaly_meta":  None,
        "df_risk":        None, "risk_meta":     None,
        "report":         None, "report_path":   None,
    }

    # ── Agent function closures ───────────────────────────────────────────────
    # DataAgent now also performs feature engineering inline → 5 agents total,
    # matching the spec topology (Data → Baseline → Outlier → Risk → Report).
    def _run_data(s):
        return data_agent_node(s, save_artifacts=save_outputs)
    def _run_baseline(s):
        return run_baseline_agent(s, save_output=save_outputs)
    def _run_outlier(s):
        return run_outlier_agent(s, save_output=save_outputs)
    def _run_risk(s):
        return run_risk_profiling_agent(s, save_output=save_outputs)
    def _run_report(s):
        return run_report_agent(s, save_output=save_outputs, use_llm=use_llm, dry_run=dry_run)

    agent_stages = [
        ("data",     "DataAgent",          0, _run_data,     "data_meta"),
        ("baseline", "BaselineAgent",      1, _run_baseline, "baseline_meta"),
        ("outlier",  "OutlierAgent",       2, _run_outlier,  "anomaly_meta"),
        ("risk",     "RiskProfilingAgent", 3, _run_risk,     "risk_meta"),
    ]
    if run_report:
        agent_stages.append(("report", "ReportAgent", 4, _run_report, "report"))

    # ── UI containers ────────────────────────────────────────────────────────
    stage_errors: dict[str, str]  = {}
    stage_results: dict[str, dict] = {}
    started_at   = time.perf_counter()
    aborted      = False

    with st.status("Running multi-agent pipeline…", expanded=True) as pipeline_status:
        for stage_name, agent_name, step_idx, agent_fn, meta_key in agent_stages:
            st.write(f"▶ **{agent_name}** running…")
            t0 = time.perf_counter()

            try:
                result = agent_fn(state)
                state = result
                meta    = state.get(meta_key)
                err     = meta.get("error") if isinstance(meta, dict) else None
                elapsed = (
                    meta.get("elapsed_s", round(time.perf_counter() - t0, 2))
                    if isinstance(meta, dict) else round(time.perf_counter() - t0, 2)
                )
                stage_results[stage_name] = {"ok": err is None, "error": err, "elapsed_s": elapsed}

                if err:
                    stage_errors[stage_name] = err
                    st.write(f"❌ **{agent_name}** failed — {err}")
                    if not continue_on_error:
                        aborted = True
                        break
                else:
                    st.write(f"✅ **{agent_name}** — {elapsed:.2f}s")

            except Exception as exc:
                elapsed = round(time.perf_counter() - t0, 2)
                stage_results[stage_name] = {"ok": False, "error": str(exc), "elapsed_s": elapsed}
                stage_errors[stage_name]  = str(exc)
                st.write(f"❌ **{agent_name}** exception — {exc}")
                if not continue_on_error:
                    aborted = True
                    break

        if aborted:
            pipeline_status.update(label="Pipeline stopped (error)", state="error")
        else:
            pipeline_status.update(label="Pipeline completed ✓", state="complete")

    # Show final agent graph (embedded via React + Babel CDN — iframe-safe)
    n_ok = len([v for v in stage_results.values() if v["ok"]])
    _components.html(
        _render_agent_graph_html(n_ok, stage_errors),
        height=1120,
        scrolling=True,
    )

    summary = {
        "perimeter":        perimeter,
        "report_path":      state.get("report_path"),
        "stages":           stage_results,
        "step_errors":      stage_errors,
        "completed_stages": [k for k, v in stage_results.items() if v["ok"]],
        "failed_stages":    [k for k, v in stage_results.items() if not v["ok"]],
        "run_config": {
            "run_report": run_report, "use_llm": use_llm,
            "dry_run":    dry_run,    "continue_on_error": continue_on_error,
            "save_outputs": save_outputs,
        },
        "runtime_s": round(time.perf_counter() - started_at, 3),
    }
    return state, summary


# ─────────────────────────── Route map helpers ───────────────────────────────

_RISK_COLORS  = {"ALTA": "#ef4444", "MEDIA": "#f59e0b", "NORMALE": "#22c55e"}
_RISK_WIDTHS  = {"ALTA": 2.5,       "MEDIA": 1.5,       "NORMALE": 0.5}
_RISK_OPACITY = {"ALTA": 0.90,      "MEDIA": 0.65,      "NORMALE": 0.18}


def _make_route_map_figure(
    df: pd.DataFrame,
    findings_by_rotta: dict | None = None,
) -> tuple[go.Figure, list[str]]:
    """
    Builds a Plotly Scattergeo figure.

    Returns (fig, clickable_routes) where clickable_routes[i] is the ROTTA
    corresponding to trace i — used to map Plotly click events back to routes.
    ALTA + MEDIA: one trace per route (hover + click).
    NORMALE: single batched trace (performance).
    """
    findings_by_rotta = findings_by_rotta or {}
    risk_col = "risk_label" if "risk_label" in df.columns else "anomaly_label"

    traces: list            = []
    clickable_routes: list[str] = []

    # Sort so ALTA renders on top of NORMALE
    df_work = df.copy()
    _ro_map  = {"ALTA": 2, "MEDIA": 1, "NORMALE": 0}
    df_work["_ro"] = df_work.get(risk_col, pd.Series("NORMALE", index=df.index)).map(_ro_map).fillna(0)
    df_work = df_work.sort_values("_ro")

    norm_lats: list = []
    norm_lons: list = []

    for _, row in df_work.iterrows():
        rotta = str(row.get("ROTTA", ""))
        parts = rotta.split("-")
        if len(parts) != 2:
            continue
        dep, arr = parts
        if dep not in IATA_COORDS or arr not in IATA_COORDS:
            continue

        lat_dep, lon_dep = IATA_COORDS[dep]
        lat_arr, lon_arr = IATA_COORDS[arr]
        risk  = str(row.get(risk_col, "NORMALE"))
        score = float(row.get("ensemble_score") or row.get("anomaly_score") or 0)

        if risk == "NORMALE":
            norm_lats.extend([lat_dep, lat_arr, None])
            norm_lons.extend([lon_dep, lon_arr, None])
        else:
            raw_exp     = findings_by_rotta.get(rotta, {}).get("explanation", "")
            short_exp   = (raw_exp[:160] + "…") if len(raw_exp) > 160 else raw_exp
            hover_extra = (f"<br><br><i>{short_exp}</i>" if short_exp and
                           not short_exp.startswith(("LLM explanation skipped", "LLM explanation unavailable")) else "")
            hover = (
                f"<b>{rotta}</b><br>"
                f"Risk: <b style='color:{_RISK_COLORS[risk]}'>{risk}</b><br>"
                f"Score: {score:.3f}"
                f"{hover_extra}"
            )
            traces.append(go.Scattergeo(
                lat=[lat_dep, lat_arr],
                lon=[lon_dep, lon_arr],
                mode="lines+markers",
                line=dict(width=_RISK_WIDTHS[risk], color=_RISK_COLORS[risk]),
                # size=[12, 0] → visible clickable dot only at departure (index 0)
                marker=dict(
                    size=[12, 0],
                    color=_RISK_COLORS[risk],
                    opacity=0.9,
                    symbol="circle",
                    line=dict(width=1.5, color="white"),
                ),
                opacity=_RISK_OPACITY[risk],
                hovertemplate=hover + "<extra></extra>",
                name=rotta,
                customdata=[rotta, rotta],
                showlegend=False,
            ))
            clickable_routes.append(rotta)

    # NORMALE batch
    if norm_lats:
        traces.append(go.Scattergeo(
            lat=norm_lats, lon=norm_lons,
            mode="lines",
            line=dict(width=0.5, color="#22c55e"),
            opacity=0.13,
            hoverinfo="skip",
            name="NORMALE",
            showlegend=False,
        ))

    # Italian airport dots
    arr_apts = {
        str(r).split("-")[1]
        for r in df["ROTTA"].dropna()
        if len(str(r).split("-")) == 2 and str(r).split("-")[1] in IATA_COORDS
    }
    if arr_apts:
        traces.append(go.Scattergeo(
            lat=[IATA_COORDS[a][0] for a in arr_apts],
            lon=[IATA_COORDS[a][1] for a in arr_apts],
            mode="markers+text",
            marker=dict(size=14, color="#22d3ee", symbol="diamond",
                        line=dict(width=1.5, color="#0e7490")),
            text=list(arr_apts),
            textposition="top right",
            textfont=dict(size=9, color="#67e8f9"),
            hovertext=[f"🇮🇹 {a}" for a in arr_apts],
            hoverinfo="text",
            name="Aeroporti IT",
            showlegend=False,
        ))

    # Legend dummy traces
    for label in ["ALTA", "MEDIA", "NORMALE"]:
        traces.append(go.Scattergeo(
            lat=[None], lon=[None],
            mode="lines",
            line=dict(width=3, color=_RISK_COLORS[label]),
            name=label,
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        height=580,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        legend=dict(
            orientation="h", x=0.5, xanchor="center",
            y=0.01, yanchor="bottom",
            bgcolor="rgba(15,23,42,0.8)",
            bordercolor="#334155", borderwidth=1,
            font=dict(color="#94a3b8", size=12),
        ),
        geo=dict(
            projection_type="natural earth",
            showland=True,     landcolor="#1e293b",
            showocean=True,    oceancolor="#020617",
            showcountries=True, countrycolor="#0f172a",
            showframe=False,
            coastlinecolor="#334155", coastlinewidth=0.5,
            bgcolor="#0f172a",
            center=dict(lat=28, lon=18),
            projection_scale=1.2,
        ),
    )
    return fig, clickable_routes


def _show_route_map_tab(df_anom: pd.DataFrame | None, report_obj: dict | None) -> None:
    st.markdown("### Risk Route Map")
    st.caption(
        "Arcs coloured by risk level. "
        "**Hover** over a route for a quick summary — "
        "**click** or use the selector below for the full LLM analysis."
    )

    if df_anom is None or df_anom.empty:
        st.info("Run the pipeline to visualise the route map.")
        return

    findings_by_rotta: dict = {}
    if isinstance(report_obj, dict):
        for f in report_obj.get("findings", []):
            rotta = f.get("ROTTA", "")
            if rotta:
                findings_by_rotta[rotta] = f

    risk_col = "risk_label" if "risk_label" in df_anom.columns else "anomaly_label"

    # KPI header
    if risk_col in df_anom.columns:
        counts = df_anom[risk_col].value_counts()
        total_mapped = sum(
            1 for r in df_anom["ROTTA"].dropna()
            if (parts := str(r).split("-")) and len(parts) == 2
            and parts[0] in IATA_COORDS and parts[1] in IATA_COORDS
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ALTA",    int(counts.get("ALTA",    0)))
        c2.metric("MEDIA",   int(counts.get("MEDIA",   0)))
        c3.metric("NORMALE", int(counts.get("NORMALE", 0)))
        c4.metric("Mapped routes", total_mapped)

    fig, clickable_routes = _make_route_map_figure(df_anom, findings_by_rotta)

    # Render map — use on_select if Streamlit >= 1.35, else fall back
    selected_route: str | None = None
    try:
        event = st.plotly_chart(
            fig, use_container_width=True, on_select="rerun", key="risk_map"
        )
        if event and hasattr(event, "selection") and event.selection.points:
            pt = event.selection.points[0]
            idx = pt.get("curve_number", -1)
            if 0 <= idx < len(clickable_routes):
                selected_route = clickable_routes[idx]
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)

    # Route detail panel
    if risk_col in df_anom.columns:
        high_risk_routes = (
            df_anom[df_anom[risk_col].isin(["ALTA", "MEDIA"])]
            .sort_values("ensemble_score" if "ensemble_score" in df_anom.columns
                         else df_anom.columns[0], ascending=False)["ROTTA"]
            .dropna().tolist()
        )
        if not high_risk_routes:
            st.info("No HIGH/MEDIUM risk routes in the current dataset.")
            return

        st.markdown("---")
        st.markdown("#### Route Detail — HIGH / MEDIUM risk")

        default_idx = 0
        if selected_route and selected_route in high_risk_routes:
            default_idx = high_risk_routes.index(selected_route)

        sel = st.selectbox(
            "Select route",
            options=high_risk_routes,
            index=default_idx,
            key="route_detail_selector",
        )

        if sel:
            row = df_anom[df_anom["ROTTA"] == sel].iloc[0]
            risk  = row.get(risk_col, "N/A")
            score_val = (
                row.get("ensemble_score") or row.get("anomaly_score") or 0
            )
            paese = row.get("PAESE_PART", "N/A")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Route",   sel)
            col2.metric("Country", paese)
            col3.metric("Risk",    risk)
            col4.metric("Score",   f"{float(score_val):.3f}")

            # LLM explanation
            _PLACEHOLDER_PREFIXES = ("LLM explanation skipped", "LLM explanation unavailable")
            if sel in findings_by_rotta:
                exp = findings_by_rotta[sel].get("explanation", "")
                if exp and not exp.startswith(_PLACEHOLDER_PREFIXES):
                    st.markdown("**LLM Analysis:**")
                    st.info(exp)
                else:
                    st.caption(
                        "LLM explanation not available for this route. "
                        "Enable the ReportAgent with **Enable LLM Report** and re-run."
                    )
            else:
                st.caption(
                    "No LLM explanation available — run the pipeline with "
                    "**Enable LLM Report** checked."
                )

            # Key metrics
            metric_cols = [c for c in [
                "baseline_score", "ensemble_score", "score_composito",
                "tasso_fermati", "tasso_rilevanza", "tasso_allarme_medio",
                "pct_interpol", "pct_sdi", "pct_nsis", "tasso_respinti",
            ] if c in df_anom.columns]
            if metric_cols:
                with st.expander("Detailed metrics"):
                    metrics_df = pd.DataFrame({
                        "Feature": metric_cols,
                        "Value":   [round(float(row.get(c) or 0), 4) for c in metric_cols],
                    })
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)


# ──────────────────────────── Misc helpers ────────────────────────────────────

def _build_perimeter(
    anno: int | None,
    paese_partenza: str,
    aeroporto_partenza: str,
    aeroporto_arrivo: str,
    zona: int | None,
) -> dict:
    perimeter: dict = {}
    if anno is not None:
        perimeter["anno"] = anno
    if paese_partenza.strip():
        perimeter["paese_partenza"] = paese_partenza.strip()
    if aeroporto_partenza.strip():
        perimeter["aeroporto_partenza"] = aeroporto_partenza.strip().upper()
    if aeroporto_arrivo.strip():
        perimeter["aeroporto_arrivo"] = aeroporto_arrivo.strip().upper()
    if zona is not None:
        perimeter["zona"] = zona
    return perimeter


def _render_stage_badges(summary: dict) -> None:
    stages = summary.get("stages", {})
    if not stages:
        st.info("No stages executed.")
        return
    html = []
    for stage, details in stages.items():
        css   = "ok" if details.get("ok") else "err"
        label = f"{stage}: {'OK' if details.get('ok') else 'ERROR'}"
        html.append(f"<span class='chip {css}'>{label}</span>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _safe_read_report(path: str | None, in_memory: dict | None) -> dict | None:
    if in_memory and not in_memory.get("error"):
        return in_memory
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _load_filter_options() -> dict:
    merged_path = PROJECT_ROOT / "data/processed/dataset_merged.csv"
    if not merged_path.exists():
        return {"anni": [2024], "paesi": [], "apt_dep": [], "apt_arr": [], "zone": list(range(1, 10))}
    try:
        df   = pd.read_csv(merged_path)
        anni = sorted([int(x) for x in df.get("ANNO_PARTENZA", pd.Series(dtype="float")).dropna().unique()])
        paesi   = sorted(df.get("PAESE_PART", pd.Series(dtype="object")).dropna().astype(str).unique().tolist())
        apt_dep = sorted(df.get("AREOPORTO_PARTENZA", pd.Series(dtype="object")).dropna().astype(str).unique().tolist())
        apt_arr = sorted(df.get("AREOPORTO_ARRIVO",   pd.Series(dtype="object")).dropna().astype(str).unique().tolist())
        zone    = sorted([int(x) for x in df.get("ZONA", pd.Series(dtype="float")).dropna().unique()])
        return {
            "anni":    anni    or [2024],
            "paesi":   paesi,
            "apt_dep": apt_dep,
            "apt_arr": apt_arr,
            "zone":    zone    or list(range(1, 10)),
        }
    except Exception:
        return {"anni": [2024], "paesi": [], "apt_dep": [], "apt_arr": [], "zone": list(range(1, 10))}


def _stage_table(summary: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {"stage": s, "status": "OK" if d.get("ok") else "ERROR",
         "elapsed_s": d.get("elapsed_s", ""), "error": d.get("error") or ""}
        for s, d in summary.get("stages", {}).items()
    ])


@st.cache_data(show_spinner=False)
def _load_classical_report() -> pd.DataFrame | None:
    """Load the classical pipeline report (pre-computed).
    Tries final_report.csv first, then anomaly_results.csv."""
    for name in ("final_report.csv", "anomaly_results.csv"):
        cl_path = PROJECT_ROOT / "data" / "processed" / name
        if cl_path.exists():
            try:
                return pd.read_csv(cl_path)
            except Exception:
                continue
    return None


# ──────────────────────────────── Main ───────────────────────────────────────

def main() -> None:
    _inject_style()

    for key, default in [("last_run", None), ("run_history", [])]:
        if key not in st.session_state:
            st.session_state[key] = default

    options = _load_filter_options()

    st.markdown("""
<div style="
  background: linear-gradient(135deg, rgba(14,165,233,0.07) 0%, rgba(6,13,26,0.97) 100%);
  border: 1px solid rgba(14,165,233,0.15);
  border-radius: 18px;
  padding: 26px 32px 22px;
  margin-bottom: 22px;
  position: relative;
  overflow: hidden;
">
  <div style="
    position:absolute; top:-40px; right:-60px; width:260px; height:260px; border-radius:50%;
    background: radial-gradient(circle, rgba(14,165,233,0.10) 0%, transparent 70%);
    pointer-events:none;
  "></div>
  <div style="position:relative; display:flex; align-items:center; gap:16px;">
    <div style="
      width:52px; height:52px; border-radius:14px; flex-shrink:0;
      background: linear-gradient(135deg, rgba(14,165,233,0.18), rgba(2,132,199,0.1));
      border: 1px solid rgba(14,165,233,0.3);
      display:flex; align-items:center; justify-content:center;
      font-size:26px;
    ">✈️</div>
    <div>
      <div style="
        font-size:22px; font-weight:900; letter-spacing:-0.4px;
        background: linear-gradient(135deg, #f1f5f9 30%, #7dd3fc 100%);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        font-family:'Inter',system-ui,sans-serif;
      ">Airport Risk Intelligence</div>
      <div style="margin-top:4px; font-size:12.5px; color:#475569; font-family:'Inter',system-ui,sans-serif;">
        Multi-agent anomaly detection &nbsp;·&nbsp; LangGraph orchestration &nbsp;·&nbsp; Reply × LUISS 2026
      </div>
    </div>
    <div style="margin-left:auto; display:flex; gap:8px; flex-shrink:0;">
      <span style="
        padding:4px 11px; border-radius:999px; font-size:10.5px; font-weight:700;
        letter-spacing:0.5px; font-family:'Inter',system-ui,sans-serif;
        background:rgba(34,197,94,0.1); border:1px solid rgba(34,197,94,0.25); color:#4ade80;
      ">LIVE</span>
      <span style="
        padding:4px 11px; border-radius:999px; font-size:10.5px; font-weight:700;
        letter-spacing:0.5px; font-family:'Inter',system-ui,sans-serif;
        background:rgba(168,85,247,0.1); border:1px solid rgba(168,85,247,0.25); color:#c084fc;
      ">LLM-POWERED</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
<div style="
  padding: 12px 0 8px;
  font-size:11px; font-weight:700; letter-spacing:1.4px;
  text-transform:uppercase; color:#475569;
  font-family:'Inter',system-ui,sans-serif;
  border-bottom:1px solid rgba(30,41,59,0.6);
  margin-bottom:14px;
">⚙️ &nbsp;Configuration</div>""", unsafe_allow_html=True)

        use_anno = st.checkbox("Filter by year", value=True)
        anno     = st.selectbox("Year", options["anni"], index=0, disabled=not use_anno)

        paese   = st.selectbox("Departure country",  ["(all)"] + options["paesi"],   index=0)
        apt_dep = st.selectbox("Departure airport",  ["(all)"] + options["apt_dep"], index=0)
        apt_arr = st.selectbox("Arrival airport",    ["(all)"] + options["apt_arr"], index=0)

        use_zona = st.checkbox("Filter by zone", value=False)
        zona     = st.selectbox("Zone", options["zone"], index=0, disabled=not use_zona)

        st.divider()
        has_api_key = bool(get_anthropic_api_key())
        run_report  = st.checkbox(
            "Enable LLM Report (Anthropic)",
            value=has_api_key,
            help="Requires ANTHROPIC_API_KEY environment variable.",
        )
        dry_run = st.checkbox(
            "Dry run report (no LLM calls)",
            value=not has_api_key,
            help="Generate report without consuming API credits.",
        )
        save_outputs      = st.checkbox("Save outputs to disk",        value=True)
        continue_on_error = st.checkbox("Continue if a stage fails",   value=False)

        st.divider()
        run = st.button("Run pipeline", use_container_width=True, type="primary")

    # ── Pipeline execution ────────────────────────────────────────────────────
    if run:
        perimeter = _build_perimeter(
            anno=int(anno) if use_anno else None,
            paese_partenza=""     if paese   == "(all)" else paese,
            aeroporto_partenza="" if apt_dep == "(all)" else apt_dep,
            aeroporto_arrivo=""   if apt_arr == "(all)" else apt_arr,
            zona=int(zona) if use_zona else None,
        )
        if run_report and not get_anthropic_api_key():
            st.warning("`ANTHROPIC_API_KEY` not set — LLM report automatically disabled.")
            run_report = False

        st.markdown("""
<div style="display:flex;align-items:center;gap:10px;margin:8px 0 14px;">
  <div style="width:3px;height:22px;border-radius:2px;background:linear-gradient(180deg,#0ea5e9,#0284c7);flex-shrink:0;"></div>
  <span style="font-size:17px;font-weight:700;color:#e2e8f0;font-family:'Inter',system-ui,sans-serif;letter-spacing:-0.2px;">
    Pipeline Execution
  </span>
</div>""", unsafe_allow_html=True)
        state, summary = _run_pipeline_with_live_ui(
            perimeter=perimeter,
            run_report=run_report,
            use_llm=run_report and not dry_run,
            dry_run=dry_run,
            continue_on_error=continue_on_error,
            save_outputs=save_outputs,
        )
        elapsed_s = summary["runtime_s"]

        st.markdown("""
<div style="display:flex;align-items:center;gap:10px;margin:18px 0 10px;">
  <div style="width:3px;height:22px;border-radius:2px;background:linear-gradient(180deg,#22c55e,#16a34a);flex-shrink:0;"></div>
  <span style="font-size:17px;font-weight:700;color:#e2e8f0;font-family:'Inter',system-ui,sans-serif;letter-spacing:-0.2px;">
    Pipeline Status
  </span>
</div>""", unsafe_allow_html=True)
        _render_stage_badges(summary)

        completed = len(summary.get("completed_stages", []))
        failed    = len(summary.get("failed_stages",    []))
        df_anom   = state.get("df_anomalies")
        n_rotte   = int(len(df_anom)) if isinstance(df_anom, pd.DataFrame) else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Completed stages", completed)
        c2.metric("Failed stages",    failed)
        c3.metric("Routes analysed",  n_rotte, help=f"Runtime: {elapsed_s}s")

        st.markdown(
            f"<div class='section-card'><b>Runtime:</b> {elapsed_s}s &nbsp;|&nbsp; "
            f"<b>Perimeter:</b> {perimeter or 'no filter'}</div>",
            unsafe_allow_html=True,
        )

        st.session_state["last_run"] = {
            "state": state, "summary": summary,
            "elapsed_s": elapsed_s, "perimeter": perimeter,
        }
        st.session_state["run_history"].append({
            "runtime_s": elapsed_s, "completed": completed,
            "failed": failed,
            "perimeter": json.dumps(perimeter, ensure_ascii=False),
        })

    # ── Result tabs ───────────────────────────────────────────────────────────
    last_run = st.session_state.get("last_run")
    if last_run:
        state   = last_run["state"]
        summary = last_run["summary"]
        # Prefer the richer df_risk (RiskProfilingAgent output: br_*, br_score,
        # confidence, final_risk, risk_drivers) and fall back to df_anomalies
        # for older runs / configurations where the risk layer is missing.
        df_risk = state.get("df_risk")
        df_anom = state.get("df_anomalies")
        df_view = df_risk if isinstance(df_risk, pd.DataFrame) and not df_risk.empty else df_anom
        risk_meta = state.get("risk_meta") or {}

        report_obj = _safe_read_report(state.get("report_path"), state.get("report"))

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🗺️ Route Map",
            "⚠️ Anomalies",
            "⚖️ Classical vs Multi-Agent",
            "📋 Report",
            "🔧 Stage Detail",
            "🐛 Debug JSON",
        ])

        # ── Tab 1: Route Map ──────────────────────────────────────────────────
        with tab1:
            _show_route_map_tab(
                df_view if isinstance(df_view, pd.DataFrame) else None,
                report_obj,
            )

        # ── Tab 2: Anomalies ──────────────────────────────────────────────────
        with tab2:
            if isinstance(df_view, pd.DataFrame) and not df_view.empty:
                using_risk = "final_risk" in df_view.columns

                # ML risk distribution (ALTA/MEDIA/NORMALE) — always available
                st.markdown("### ML risk distribution (OutlierAgent)")
                risk_col = "risk_label" if "risk_label" in df_view.columns else "anomaly_label"
                if risk_col in df_view.columns:
                    counts = (
                        df_view[risk_col]
                        .value_counts()
                        .reindex(["ALTA", "MEDIA", "NORMALE"], fill_value=0)
                    )
                    st.bar_chart(counts)

                # Final-risk distribution (CRITICO/ALTO/MEDIO/BASSO) from
                # RiskProfilingAgent — only if df_risk is present.
                if using_risk:
                    st.markdown("### Final risk classification (RiskProfilingAgent)")
                    final_counts = (
                        df_view["final_risk"]
                        .value_counts()
                        .reindex(["CRITICO", "ALTO", "MEDIO", "BASSO"], fill_value=0)
                    )
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("CRITICO", int(final_counts.get("CRITICO", 0)))
                    c2.metric("ALTO",    int(final_counts.get("ALTO", 0)))
                    c3.metric("MEDIO",   int(final_counts.get("MEDIO", 0)))
                    c4.metric("BASSO",   int(final_counts.get("BASSO", 0)))

                    # Business-rule hit counts (RiskProfilingAgent meta)
                    rh = risk_meta.get("rule_hits") or {}
                    if rh:
                        st.markdown("### Business rules — hit counts")
                        rh_df = pd.DataFrame(
                            [(k.replace("br_", ""), int(v)) for k, v in rh.items()],
                            columns=["business_rule", "n_routes"],
                        )
                        st.dataframe(rh_df, use_container_width=True, hide_index=True)

                # Top routes table — show the richer risk-layer columns when available
                st.markdown("### Top routes")
                preferred_cols = [
                    "ROTTA", "PAESE_PART", "ZONA",
                    "risk_label", "final_risk", "confidence",
                    "ensemble_score", "br_score", "baseline_score",
                    "risk_drivers",
                ]
                visible_cols = [c for c in preferred_cols if c in df_view.columns]
                sort_col = (
                    "confidence" if "confidence" in df_view.columns
                    else "ensemble_score" if "ensemble_score" in df_view.columns
                    else df_view.columns[0]
                )
                show_df = df_view.sort_values(sort_col, ascending=False)[visible_cols].head(50).copy()
                # risk_drivers is a list — flatten for the table
                if "risk_drivers" in show_df.columns:
                    show_df["risk_drivers"] = show_df["risk_drivers"].apply(
                        lambda v: " | ".join(v) if isinstance(v, list) else (v or "")
                    )
                st.dataframe(show_df, use_container_width=True)
                st.download_button(
                    "Download anomalies (CSV)",
                    data=show_df.to_csv(index=False).encode("utf-8"),
                    file_name="top_routes_anomalies.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.info("No anomaly data available.")

        # ── Tab 3: Classical vs Multi-Agent ───────────────────────────────────
        with tab3:
            st.markdown("### Classical vs Multi-Agent Comparison")
            cl = _load_classical_report()

            if cl is None:
                st.warning(
                    "Classical `final_report.csv` not found in `data/processed/`. "
                    "Run the classical pipeline notebooks first (01→06)."
                )
            elif not isinstance(df_view, pd.DataFrame) or df_view.empty:
                st.info("Run the multi-agent pipeline to enable the comparison.")
            else:
                cl_cols      = ["ROTTA", "anomaly_score", "anomaly_label"]
                missing_cl   = [c for c in cl_cols if c not in cl.columns]
                if missing_cl:
                    st.error(f"Missing columns in classical report: {missing_cl}")
                else:
                    df_cmp = cl[cl_cols].merge(
                        df_view[["ROTTA", "ensemble_score", "risk_label"]],
                        on="ROTTA", how="inner",
                    )
                    if df_cmp.empty:
                        st.warning("No routes in common between the two pipelines.")
                    else:
                        df_cmp["label_concorde"] = (
                            df_cmp["anomaly_label"] == df_cmp["risk_label"]
                        )
                        df_cmp["delta_score"] = (
                            df_cmp["ensemble_score"] - df_cmp["anomaly_score"]
                        ).round(4)

                        n_cmp        = len(df_cmp)
                        scope_label  = (
                            "full dataset"
                            if not last_run.get("perimeter")
                            else f"perimeter: {last_run['perimeter']}"
                        )
                        st.caption(f"Comparison on **{n_cmp}** routes ({scope_label})")

                        pr, _  = pearsonr( df_cmp["anomaly_score"], df_cmp["ensemble_score"])
                        sr, _  = spearmanr(df_cmp["anomaly_score"], df_cmp["ensemble_score"])
                        agree  = df_cmp["label_concorde"].mean()
                        top_n  = min(20, n_cmp)
                        top_cl = set(df_cmp.nlargest(top_n, "anomaly_score")["ROTTA"])
                        top_ma = set(df_cmp.nlargest(top_n, "ensemble_score")["ROTTA"])

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Pearson r",       f"{pr:.3f}",  help="Linear score correlation")
                        c2.metric("Spearman r",      f"{sr:.3f}",  help="Rank correlation")
                        c3.metric("Label agreement", f"{agree:.1%}")
                        c4.metric(f"Top-{top_n} overlap",
                                  f"{len(top_cl & top_ma)}/{top_n}")

                        chart = (
                            alt.Chart(df_cmp)
                            .mark_circle(size=55, opacity=0.7)
                            .encode(
                                x=alt.X("anomaly_score:Q",  title="Classical score"),
                                y=alt.Y("ensemble_score:Q", title="Multi-Agent score"),
                                color=alt.Color(
                                    "anomaly_label:N",
                                    scale=alt.Scale(
                                        domain=["ALTA", "MEDIA", "NORMALE"],
                                        range=["#e05252", "#e0a852", "#5285e0"],
                                    ),
                                    legend=alt.Legend(title="Classical label"),
                                ),
                                tooltip=["ROTTA", "anomaly_label", "risk_label",
                                         "anomaly_score", "ensemble_score", "delta_score"],
                            )
                            .properties(
                                title=f"Score correlation (Pearson r={pr:.3f} | Spearman r={sr:.3f})",
                                width=600, height=400,
                            )
                        )
                        max_val   = max(float(df_cmp["anomaly_score"].max()),
                                        float(df_cmp["ensemble_score"].max()), 0.6)
                        diagonal  = (
                            alt.Chart(pd.DataFrame({"x": [0, max_val], "y": [0, max_val]}))
                            .mark_line(color="gray", strokeDash=[4, 4], opacity=0.5)
                            .encode(x="x:Q", y="y:Q")
                        )
                        st.altair_chart(chart + diagonal, use_container_width=True)

                        gold = df_cmp[
                            (df_cmp["anomaly_label"] == "ALTA") &
                            (df_cmp["risk_label"]    == "ALTA")
                        ]
                        st.markdown(
                            f"### ALTA routes agreed by both pipelines ({len(gold)} routes)"
                        )
                        if not gold.empty:
                            st.dataframe(
                                gold[["ROTTA", "anomaly_score", "ensemble_score", "delta_score"]]
                                .sort_values("anomaly_score", ascending=False),
                                use_container_width=True, hide_index=True,
                            )
                        else:
                            st.info("No routes classified ALTA by both pipelines.")

                        with st.expander(f"Full comparison table ({n_cmp} routes)"):
                            st.dataframe(
                                df_cmp.sort_values("anomaly_score", ascending=False),
                                use_container_width=True, hide_index=True,
                            )
                            st.download_button(
                                "Download comparison (CSV)",
                                data=df_cmp.to_csv(index=False).encode("utf-8"),
                                file_name="comparison_classical_vs_multiagent.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )

        # ── Tab 4: Report ─────────────────────────────────────────────────────
        with tab4:
            raw_report  = state.get("report") or {}
            report_error = raw_report.get("error") if isinstance(raw_report, dict) else None
            if report_obj:
                st.markdown("### Summary")
                st.write(report_obj.get("summary", "N/A"))

                # RiskProfilingAgent recap up front so the user sees the
                # business-rule layer next to the LLM narratives.
                if risk_meta and not risk_meta.get("error"):
                    st.markdown("### Final risk classification (RiskProfilingAgent)")
                    cc = st.columns(4)
                    cc[0].metric("CRITICO", int(risk_meta.get("n_critico", 0)))
                    cc[1].metric("ALTO",    int(risk_meta.get("n_alto", 0)))
                    cc[2].metric("MEDIO",   int(risk_meta.get("n_medio", 0)))
                    cc[3].metric("BASSO",   int(risk_meta.get("n_basso", 0)))

                findings = report_obj.get("findings", [])
                if findings:
                    st.markdown("### Findings")
                    findings_df = pd.DataFrame(findings)
                    if "risk_drivers" in findings_df.columns:
                        findings_df["risk_drivers"] = findings_df["risk_drivers"].apply(
                            lambda v: " | ".join(v) if isinstance(v, list) else (v or "")
                        )
                    st.dataframe(findings_df, use_container_width=True)
                else:
                    st.caption("No HIGH/MEDIUM risk routes to explain.")
                st.download_button(
                    "Download report (JSON)",
                    data=json.dumps(report_obj, indent=2, ensure_ascii=False, default=str),
                    file_name="multiagent_report.json",
                    mime="application/json",
                    use_container_width=True,
                )
            else:
                stages      = (summary or {}).get("stages", {})
                report_stage = stages.get("report")
                if report_error:
                    st.error(f"ReportAgent error: {report_error}")
                elif report_stage is None:
                    st.info(
                        "Report not executed. Check 'Enable LLM Report' and run again."
                    )
                else:
                    st.info("Report not available for this run.")

        # ── Tab 5: Stage Detail ────────────────────────────────────────────────
        with tab5:
            st.markdown("### Stage Results")
            st_df = _stage_table(summary)
            st.dataframe(st_df, use_container_width=True, hide_index=True)
            if not st_df.empty and (st_df["status"] == "ERROR").any():
                first_err = st_df[st_df["status"] == "ERROR"].iloc[0]["error"]
                st.error(first_err or "Stage failed with no error detail.")

            hist = st.session_state.get("run_history", [])
            if hist:
                st.markdown("### Run History (current session)")
                st.dataframe(pd.DataFrame(hist).tail(10), use_container_width=True, hide_index=True)

        # ── Tab 6: Debug JSON ─────────────────────────────────────────────────
        with tab6:
            st.markdown("### Summary orchestrator")
            st.json(summary)
            st.markdown("### Meta")
            st.json({
                "data_meta":     state.get("data_meta"),
                "feature_meta":  state.get("feature_meta"),
                "baseline_meta": state.get("baseline_meta"),
                "anomaly_meta":  state.get("anomaly_meta"),
                "risk_meta":     state.get("risk_meta"),
                "report_path":   state.get("report_path"),
            })
    else:
        st.markdown("""
<div style="
  margin-top:32px;
  background: linear-gradient(135deg, rgba(14,165,233,0.05), rgba(15,23,42,0.7));
  border: 1px dashed rgba(14,165,233,0.2);
  border-radius: 16px;
  padding: 48px 32px;
  text-align: center;
">
  <div style="font-size:40px;margin-bottom:16px;">🛫</div>
  <div style="font-size:18px;font-weight:700;color:#e2e8f0;
              font-family:'Inter',system-ui,sans-serif;margin-bottom:8px;">
    Ready to analyse
  </div>
  <div style="font-size:13px;color:#475569;font-family:'Inter',system-ui,sans-serif;max-width:380px;margin:0 auto;">
    Configure filters in the sidebar, then click <b style="color:#38bdf8">Run pipeline</b> to start the multi-agent anomaly detection.
  </div>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
