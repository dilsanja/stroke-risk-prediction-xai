# =============================================================================
#  stroke_app.py  —  NeuraScan · Brain Stroke Risk Prediction
#  P.D.S. PERERA | 258733L | CS5998 Capstone | University of Moratuwa
#
#  SETUP (run once):
#    1. Run stroke_prediction_milestone3.ipynb  →  creates deployment/ folder
#    2. pip install streamlit scikit-learn joblib numpy pandas matplotlib
#    3. streamlit run stroke_app.py
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib, json, os, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  MAC libomp FIX — handles source-installed Python on Apple Silicon
#  Sets DYLD path so LightGBM can find libomp even without Homebrew
# ─────────────────────────────────────────────────────────────────────────────
import platform, ctypes.util
if platform.system() == "Darwin":
    # Common libomp locations — try each one
    libomp_candidates = [
        "/usr/local/opt/libomp/lib/libomp.dylib",
        "/opt/homebrew/opt/libomp/lib/libomp.dylib",
        "/opt/local/lib/libomp/libomp.dylib",
        "/usr/local/lib/libomp.dylib",
    ]
    libomp_found = any(Path(p).exists() for p in libomp_candidates)
    if not libomp_found:
        # libomp not installed — patch joblib to skip LightGBM gracefully
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        os.environ["OMP_NUM_THREADS"] = "1"

# Safe LightGBM import — won't crash if missing or broken
try:
    import lightgbm as lgb
    LGB_OK = True
except (ImportError, OSError):
    LGB_OK = False

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuraScan — Stroke Risk AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=Outfit:wght@300;400;500;600;700&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"]  { font-family: 'Outfit', sans-serif; }
#MainMenu, footer, header   { visibility: hidden; }
.stDeployButton             { display: none; }
.stApp                      { background: #080E1F; }
.block-container            { padding: 0 !important; max-width: 100% !important; }

/* ── Sidebar collapsed ── */
section[data-testid="stSidebar"] { display: none; }
[data-testid="collapsedControl"] { display: none; }

/* ── Typography ── */
.serif  { font-family: 'Playfair Display', serif; }
.sans   { font-family: 'Outfit', sans-serif; }

/* ══════════════════════════════════════
   HOME PAGE
══════════════════════════════════════ */

.hero {
  min-height: auto;
  padding-bottom: 20px !important;
  background:
    radial-gradient(ellipse 80% 60% at 50% -10%, rgba(21,101,192,0.35) 0%, transparent 70%),
    radial-gradient(ellipse 40% 40% at 85% 80%, rgba(0,137,123,0.2) 0%, transparent 60%),
    #080E1F;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 24px 80px;
  text-align: center;
  position: relative;
  overflow: hidden;
}

.hero::before {
  content: '';
  position: absolute; inset: 0;
  background-image:
    radial-gradient(circle, rgba(255,255,255,0.04) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events: none;
}

.hero-badge {
  display: inline-flex; align-items: center; gap: 8px;
  background: rgba(21,101,192,0.15);
  border: 1px solid rgba(21,101,192,0.4);
  color: #64B5F6;
  font-size: 0.72rem; font-weight: 600;
  letter-spacing: 2px; text-transform: uppercase;
  padding: 6px 16px; border-radius: 99px;
  margin-bottom: 28px;
  animation: fadeUp 0.6s ease both;
}

.hero-title {
  font-family: 'Playfair Display', serif;
  font-size: clamp(3rem, 7vw, 5.5rem);
  font-weight: 800;
  color: #FFFFFF;
  line-height: 1.05;
  margin: 0 0 8px;
  animation: fadeUp 0.7s 0.1s ease both;
}

.hero-title span {
  background: linear-gradient(135deg, #42A5F5 0%, #00BCD4 50%, #00897B 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.hero-sub {
  font-size: 1.15rem; font-weight: 300;
  color: rgba(255,255,255,0.55);
  max-width: 560px; line-height: 1.7;
  margin: 16px auto 40px;
  animation: fadeUp 0.7s 0.2s ease both;
}

.hero-cta-hint {
  margin-top: 28px;
  color: rgba(255,255,255,0.35);
  font-size: 0.8rem;
  letter-spacing: 2px;
  animation: pulse 2s infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 0.35; }
  50%       { opacity: 0.8;  }
}
.hero-stats {
  display: flex; gap: 40px; justify-content: center;
  flex-wrap: wrap;
  margin-bottom: 52px;
  animation: fadeUp 0.7s 0.3s ease both;
}
.hero-stat { text-align: center; }
.hero-stat .n {
  font-family: 'Playfair Display', serif;
  font-size: 2.2rem; font-weight: 700; color: #FFFFFF;
  display: block;
}
.hero-stat .l {
  font-size: 0.76rem; color: rgba(255,255,255,0.4);
  text-transform: uppercase; letter-spacing: 1.2px;
}

.hero-btn {
  display: inline-block;
  background: linear-gradient(135deg, #1565C0, #1E88E5);
  color: #FFFFFF !important;
  font-family: 'Outfit', sans-serif;
  font-size: 1rem; font-weight: 600;
  padding: 15px 44px;
  border-radius: 12px;
  text-decoration: none;
  cursor: pointer;
  border: none;
  box-shadow: 0 8px 32px rgba(21,101,192,0.4);
  transition: transform 0.18s, box-shadow 0.18s;
  animation: fadeUp 0.7s 0.4s ease both;
}
.hero-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 12px 40px rgba(21,101,192,0.55);
}

.features {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
  max-width: 900px; width: 100%;
  margin-top: 64px;
  animation: fadeUp 0.7s 0.5s ease both;
}
.feature-card {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 14px;
  padding: 22px 20px;
  text-align: left;
}
.feature-card .icon { font-size: 1.6rem; margin-bottom: 10px; }
.feature-card .ft   { font-size: 0.93rem; font-weight: 600; color: #E3F2FD; margin-bottom: 6px; }
.feature-card .fd   { font-size: 0.8rem; color: rgba(255,255,255,0.38); line-height: 1.5; }

@keyframes fadeUp {
  from { opacity:0; transform: translateY(22px); }
  to   { opacity:1; transform: translateY(0); }
}

/* ══════════════════════════════════════
   ASSESSMENT PAGE
══════════════════════════════════════ */

.assess-wrap {
  background:
    radial-gradient(ellipse 60% 50% at 100% 0%, rgba(0,137,123,0.12) 0%, transparent 60%),
    #080E1F;
  padding: 12px 0 80px;
}

.assess-header {
  max-width: 1100px; margin: 0 auto 32px; padding: 0 32px;
}
.assess-header h2 {
  font-family: 'Playfair Display', serif;
  font-size: 2rem; color: #FFFFFF; margin: 0 0 6px;
}
.assess-header p  { color: rgba(255,255,255,0.45); font-size: 0.9rem; margin: 0; }

.form-shell {
  max-width: 1100px; margin: 0 auto; padding: 0 32px;
}

.form-card {
  background: rgba(255,255,255,0.035);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 18px;
  padding: 28px 32px;
  margin-bottom: 20px;
}
.form-card-title {
  font-size: 0.72rem; font-weight: 700;
  letter-spacing: 2px; text-transform: uppercase;
  color: #64B5F6; margin-bottom: 18px;
  padding-bottom: 10px;
  border-bottom: 1px solid rgba(255,255,255,0.06);
}

/* Streamlit widgets on dark bg */
.stSlider > label, .stSelectbox > label, .stCheckbox > label,
.stRadio > label { color: rgba(255,255,255,0.75) !important; font-size: 0.88rem !important; }
.stSlider [data-baseweb="slider"] [role="slider"] { background:#1E88E5 !important; }
div[data-testid="stSliderThumb"]  { background: #1E88E5 !important; }
.stSelectbox div[data-baseweb="select"] > div {
  background: rgba(255,255,255,0.06) !important;
  border-color: rgba(255,255,255,0.12) !important;
  color: #fff !important;
}
.stCheckbox > label > div { background: rgba(255,255,255,0.06) !important; }
.stCheckbox [data-testid="stCheckbox"] { color: white !important; }

/* Submit button */
div.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #00BCD4 0%, #26A69A 100%) !important;
  color: #000 !important;
  font-weight: 700 !important;
  font-size: 1.05rem !important;
  padding: 14px 28px !important;
  border-radius: 12px !important;
  border: none !important;
  letter-spacing: 0.5px !important;
  box-shadow: 0 4px 24px rgba(0,188,212,0.35) !important;
  transition: all 0.2s ease !important;
  margin-top: 8px !important;
}
div.stButton > button[kind="primary"]:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 32px rgba(0,188,212,0.5) !important;
}
div.stButton > button {
  background: linear-gradient(135deg, #0D47A1, #1565C0, #1E88E5) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 12px !important;
  font-family: 'Outfit', sans-serif !important;
  font-weight: 700 !important;
  font-size: 1.05rem !important;
  padding: 14px 0 !important;
  width: 100%;
  letter-spacing: 0.5px;
  box-shadow: 0 6px 28px rgba(21,101,192,0.45) !important;
  transition: opacity 0.2s !important;
}
div.stButton > button:hover { opacity: 0.88 !important; }

/* Back button variant */
div.stButton > button[kind="secondary"] {
  background: rgba(255,255,255,0.06) !important;
  box-shadow: none !important;
}

/* ── Number display ── */
.stat-pill {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.09);
  border-radius: 10px;
  padding: 14px 18px;
  text-align: center;
}
.stat-pill .sv { font-family:'Playfair Display',serif; font-size:1.8rem; font-weight:700; color:#fff; }
.stat-pill .sl { font-size:0.73rem; text-transform:uppercase; letter-spacing:1.2px; color:rgba(255,255,255,0.38); margin-top:2px; }

/* ══════════════════════════════════════
   RESULT PAGE
══════════════════════════════════════ */

.result-wrap {
  padding: 12px 0 80px;
  background:
    radial-gradient(ellipse 70% 50% at 50% 0%, rgba(21,101,192,0.15) 0%, transparent 65%),
    #080E1F;
}

.result-shell { max-width: 860px; margin: 0 auto; padding: 0 32px; }

.result-hero {
  border-radius: 20px;
  padding: 40px 40px 36px;
  margin-bottom: 28px;
  position: relative; overflow: hidden;
}
.result-hero.high {
  background: linear-gradient(135deg, #3E0000 0%, #1A0000 100%);
  border: 1px solid rgba(198,40,40,0.35);
}
.result-hero.low  {
  background: linear-gradient(135deg, #003320 0%, #001A10 100%);
  border: 1px solid rgba(46,125,50,0.35);
}
.result-hero::before {
  content: '';
  position: absolute; top: -60px; right: -60px;
  width: 200px; height: 200px; border-radius: 50%;
  opacity: 0.12;
}
.result-hero.high::before { background: #C62828; }
.result-hero.low::before  { background: #2E7D32; }

.result-hero .rh-badge {
  display: inline-block;
  font-size: 0.7rem; font-weight: 700; letter-spacing: 2px;
  text-transform: uppercase; padding: 4px 12px; border-radius: 99px;
  margin-bottom: 14px;
}
.result-hero.high .rh-badge { background: rgba(198,40,40,0.25); color: #EF9A9A; border: 1px solid rgba(198,40,40,0.4); }
.result-hero.low  .rh-badge { background: rgba(46,125,50,0.25);  color: #A5D6A7; border: 1px solid rgba(46,125,50,0.4); }

.result-hero .rh-pct {
  font-family: 'Playfair Display', serif;
  font-size: clamp(3.5rem, 8vw, 5.5rem);
  font-weight: 800; color: #FFFFFF;
  line-height: 1; margin: 0 0 10px;
}
.result-hero.high .rh-pct { color: #FF8A80; }
.result-hero.low  .rh-pct { color: #69F0AE; }

.result-hero .rh-label {
  font-size: 1.05rem; font-weight: 500;
  color: rgba(255,255,255,0.65);
  margin-bottom: 20px;
}

/* Gauge bar */
.gauge-track {
  height: 10px; border-radius: 99px;
  background: rgba(255,255,255,0.08);
  overflow: visible; position: relative; margin: 6px 0 24px;
}
.gauge-fill { height: 100%; border-radius: 99px; transition: width 0.8s cubic-bezier(.4,0,.2,1); }
.gauge-needle {
  position: absolute; top: -5px;
  width: 2px; height: 20px; border-radius: 2px;
  background: rgba(255,255,255,0.8);
  transform: translateX(-50%);
}

/* Metric strip */
.metric-strip {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
  margin-bottom: 24px;
}
.mstrip-item { text-align: center; }
.mstrip-val  { font-family:'Playfair Display',serif; font-size:1.5rem; font-weight:700; color:#fff; }
.mstrip-lbl  { font-size:0.7rem; text-transform:uppercase; letter-spacing:1px; color:rgba(255,255,255,0.35); margin-top:3px; }

/* Sections */
.rs-title {
  font-size: 0.7rem; font-weight:700; letter-spacing:2px; text-transform:uppercase;
  color: rgba(255,255,255,0.35); margin: 28px 0 14px;
  padding-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.07);
}

/* LIME card */
.lime-card {
  background: rgba(255,255,255,0.035);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 16px; padding: 24px;
  margin-bottom: 20px;
}

/* Recommendation card */
.rec-card {
  border-radius: 14px; padding: 22px 24px; margin-bottom: 14px;
}
.rec-card.urgent {
  background: rgba(198,40,40,0.1); border: 1px solid rgba(198,40,40,0.25);
}
.rec-card.monitor {
  background: rgba(46,125,50,0.1); border: 1px solid rgba(46,125,50,0.2);
}
.rec-card .rc-title { font-weight:700; font-size:0.95rem; margin-bottom:8px; }
.rec-card.urgent .rc-title { color: #FF8A80; }
.rec-card.monitor .rc-title { color: #A5D6A7; }
.rec-card .rc-text { font-size:0.85rem; color:rgba(255,255,255,0.55); line-height:1.6; }

/* Risk factors list */
.factor-row { display:flex; align-items:center; gap:12px; padding:10px 0; border-bottom:1px solid rgba(255,255,255,0.05); }
.factor-row:last-child { border-bottom: none; }
.factor-bar-bg { flex:1; height:6px; background:rgba(255,255,255,0.07); border-radius:99px; overflow:hidden; }
.factor-bar-fill { height:100%; border-radius:99px; }
.factor-name { width:130px; font-size:0.82rem; color:rgba(255,255,255,0.65); flex-shrink:0; }
.factor-val  { width:54px; text-align:right; font-size:0.82rem; font-weight:600; flex-shrink:0; }

/* Disclaimer */
.disclaimer {
  background: rgba(249,168,37,0.07);
  border: 1px solid rgba(249,168,37,0.2);
  border-radius: 10px; padding: 14px 18px;
  font-size: 0.8rem; color: rgba(249,168,37,0.75); line-height: 1.6;
}

/* Footer */
.ns-footer {
  text-align: center; padding: 32px 20px;
  color: rgba(255,255,255,0.2); font-size: 0.77rem;
  border-top: 1px solid rgba(255,255,255,0.05);
  margin-top: 60px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"
if "result" not in st.session_state:
    st.session_state.result = None


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    # Try M4 paths first, then M3 fallback
    candidates = [
        ("deployment/stroke_model_m4.pkl",  "deployment/stroke_scaler_m4.pkl",  "deployment/model_metadata_m4.json"),
        ("deployment/stroke_model.pkl",      "deployment/stroke_scaler.pkl",     "deployment/model_metadata.json"),
    ]
    for model_path, scaler_path, meta_path in candidates:
        if Path(model_path).exists() and Path(scaler_path).exists():
            model  = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            meta   = json.loads(Path(meta_path).read_text()) if Path(meta_path).exists() else {}
            return model, scaler, meta
    return None, None, None

model, scaler, meta = load_model()
model_ready = model is not None

if model_ready:
    THRESHOLD     = meta.get("decision_threshold", 0.35)
    FEATURE_NAMES = meta.get("feature_names", [
        "age","gender","hypertension","heart_disease","ever_married",
        "work_type","Residence_type","avg_glucose_level","bmi","smoking_status",
        "age_group","glucose_cat","bmi_cat","cardio_risk"
    ])
    MODEL_NAME    = meta.get("model_name", "Best Model")
    TEST_METRICS  = meta.get("test_metrics", {})
else:
    THRESHOLD = 0.35
    FEATURE_NAMES = [
        "age","gender","hypertension","heart_disease","ever_married",
        "work_type","Residence_type","avg_glucose_level","bmi","smoking_status",
        "age_group","glucose_cat","bmi_cat","cardio_risk"
    ]
    MODEL_NAME   = "Not loaded"
    TEST_METRICS = {}


# ─────────────────────────────────────────────────────────────────────────────
#  LIME HELPER
# ─────────────────────────────────────────────────────────────────────────────
def run_lime(instance_scaled, n_samples=2000, top_n=10):
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(42)
    stds = np.ones(len(instance_scaled)) * 0.5
    noise = rng.normal(0, 1, (n_samples, len(instance_scaled)))
    perturbed = np.vstack([instance_scaled, instance_scaled + noise * stds])
    probs = model.predict_proba(perturbed)[:, 1]
    kw = np.sqrt(len(instance_scaled)) * 0.75
    dists = np.sqrt(np.sum(((perturbed - instance_scaled) / stds) ** 2, axis=1))
    weights = np.exp(-(dists ** 2) / (kw ** 2))
    lr = Ridge(alpha=1.0)
    lr.fit(perturbed, probs, sample_weight=weights)
    contribs = lr.coef_
    order = np.argsort(np.abs(contribs))[::-1][:top_n]
    return [FEATURE_NAMES[i] for i in order], contribs[order]


# ─────────────────────────────────────────────────────────────────────────────
#  ENCODING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def encode_input(age, gender, hypertension, heart_disease, ever_married,
                 work_type, residence, avg_glucose, bmi, smoking):
    def g_cat(g): return 0 if g < 100 else (1 if g < 125 else 2)
    def b_cat(b): return 0 if b < 18.5 else (1 if b < 25 else (2 if b < 30 else 3))
    def a_grp(a): return 0 if a < 18 else (1 if a < 35 else (2 if a < 50 else (3 if a < 65 else 4)))
    return {
        "age": age,
        "gender": {"Female": 0, "Male": 1, "Other": 2}[gender],
        "hypertension": int(hypertension),
        "heart_disease": int(heart_disease),
        "ever_married": {"No": 0, "Yes": 1}[ever_married],
        "work_type": {"Govt_job": 0, "Never_worked": 1, "Private": 2,
                      "Self-employed": 3, "children": 4}[work_type],
        "Residence_type": {"Rural": 0, "Urban": 1}[residence],
        "avg_glucose_level": avg_glucose,
        "bmi": bmi,
        "smoking_status": {"Unknown": 0, "formerly smoked": 1,
                            "never smoked": 2, "smokes": 3}[smoking],
        "age_group":   a_grp(age),
        "glucose_cat": g_cat(avg_glucose),
        "bmi_cat":     b_cat(bmi),
        "cardio_risk": int(hypertension or heart_disease),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.page == "home":
    # ── CTA button ABOVE the hero so it's always visible ────────────────────
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown("""
        <div style="text-align:center; padding: 18px 0 6px;">
          <div style="font-size:0.72rem; letter-spacing:3px; text-transform:uppercase;
                      color:rgba(255,255,255,0.35); margin-bottom:6px;">
            🧠 AI · Medical Screening · Explainable
          </div>
          <div style="font-family:'Playfair Display',serif; font-size:2.6rem; font-weight:800;
                      color:#fff; line-height:1.15; margin-bottom:4px;">
            Stroke Risk
          </div>
          <div style="font-family:'Playfair Display',serif; font-size:2.6rem; font-weight:800;
                      background:linear-gradient(135deg,#00BCD4,#26A69A);
                      -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                      line-height:1.15; margin-bottom:16px;">
            Prediction AI
          </div>
          <div style="color:rgba(255,255,255,0.55); font-size:0.95rem; margin-bottom:24px;">
            Enter clinical details → get an instant AI-powered stroke risk score with full explainability.
          </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔍  Start Risk Assessment →", use_container_width=True, type="primary"):
            st.session_state.page = "assess"
            st.rerun()

        if not model_ready:
            st.warning("⚠️  Model files not found. Run the M4 notebook first to generate deployment/ files.", icon="⚠️")

        st.markdown("""
        <div style="display:flex; justify-content:center; gap:32px; margin-top:24px; flex-wrap:wrap;">
          <div style="text-align:center;">
            <div style="font-size:1.8rem; font-weight:800; color:#fff;">5,110</div>
            <div style="font-size:0.65rem; letter-spacing:2px; color:rgba(255,255,255,0.35);">TRAINING RECORDS</div>
          </div>
          <div style="text-align:center;">
            <div style="font-size:1.8rem; font-weight:800; color:#fff;">12</div>
            <div style="font-size:0.65rem; letter-spacing:2px; color:rgba(255,255,255,0.35);">ML MODELS</div>
          </div>
          <div style="text-align:center;">
            <div style="font-size:1.8rem; font-weight:800; color:#fff;">80%+</div>
            <div style="font-size:0.65rem; letter-spacing:2px; color:rgba(255,255,255,0.35);">RECALL TARGET</div>
          </div>
          <div style="text-align:center;">
            <div style="font-size:1.8rem; font-weight:800; color:#fff;">0.843</div>
            <div style="font-size:0.65rem; letter-spacing:2px; color:rgba(255,255,255,0.35);">AUC-ROC</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="hero" style="min-height:unset; padding:20px 24px 60px; background:none">
      <div class="features">
        <div class="feature-card">
          <div class="icon">🎯</div>
          <div class="ft">Threshold-Optimised</div>
          <div class="fd">Decision threshold tuned for recall ≥ 0.80 — clinically safer than the default 0.5.</div>
        </div>
        <div class="feature-card">
          <div class="icon">🔬</div>
          <div class="ft">LIME Explainability</div>
          <div class="fd">Every prediction comes with a local explanation of which features drove the score.</div>
        </div>
        <div class="feature-card">
          <div class="icon">⚡</div>
          <div class="ft">Instant Results</div>
          <div class="fd">Sub-second inference. Enter your data, click assess, get your risk score.</div>
        </div>
        <div class="feature-card">
          <div class="icon">📊</div>
          <div class="ft">Clinical Context</div>
          <div class="fd">Results shown with probability, risk level, contributing factors, and recommendations.</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if model_ready:
        # M4 metadata uses recall_opt / prec_opt keys
        # fallback to old M3 keys for backwards compatibility
        rec  = TEST_METRICS.get("recall_opt",  TEST_METRICS.get("recall",    "—"))
        prec = TEST_METRICS.get("prec_opt",    TEST_METRICS.get("precision", "—"))
        auc  = TEST_METRICS.get("auc_roc", "—")
        ap   = TEST_METRICS.get("ap",      TEST_METRICS.get("avg_precision", "—"))
        st.markdown(f"""
        <div style="max-width:700px; margin:0 auto 40px; padding:0 24px">
          <div style="background:rgba(255,255,255,0.035); border:1px solid rgba(255,255,255,0.07);
                      border-radius:14px; padding:20px 24px; text-align:center">
            <div style="font-size:0.7rem; letter-spacing:2px; text-transform:uppercase;
                        color:rgba(255,255,255,0.3); margin-bottom:14px">Active Model</div>
            <div style="font-size:1rem; font-weight:600; color:#64B5F6; margin-bottom:16px">
              {MODEL_NAME}
            </div>
            <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:12px">
              <div><div style="font-family:'Playfair Display',serif; font-size:1.5rem; color:#fff; font-weight:700">{f"{float(rec):.2f}" if rec != "—" else "—"}</div><div style="font-size:0.68rem; color:rgba(255,255,255,0.35); text-transform:uppercase; letter-spacing:1px">Recall</div></div>
              <div><div style="font-family:'Playfair Display',serif; font-size:1.5rem; color:#fff; font-weight:700">{f"{float(prec):.2f}" if prec != "—" else "—"}</div><div style="font-size:0.68rem; color:rgba(255,255,255,0.35); text-transform:uppercase; letter-spacing:1px">Precision</div></div>
              <div><div style="font-family:'Playfair Display',serif; font-size:1.5rem; color:#fff; font-weight:700">{f"{float(auc):.3f}" if auc != "—" else "—"}</div><div style="font-size:0.68rem; color:rgba(255,255,255,0.35); text-transform:uppercase; letter-spacing:1px">AUC-ROC</div></div>
              <div><div style="font-family:'Playfair Display',serif; font-size:1.5rem; color:#fff; font-weight:700">{f"{float(ap):.3f}" if ap != "—" else "—"}</div><div style="font-size:0.68rem; color:rgba(255,255,255,0.35); text-transform:uppercase; letter-spacing:1px">Avg Precision</div></div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="ns-footer">
      CS5998 Capstone · P.D.S. PERERA (258733L) · University of Moratuwa · Milestone 4
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: ASSESS
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == "assess":
    st.markdown('<div class="assess-wrap">', unsafe_allow_html=True)
    st.markdown("""
    <div class="assess-header">
      <h2>🩺 Patient Assessment Form</h2>
      <p>Fill in all clinical details accurately for the most reliable prediction</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="form-shell">', unsafe_allow_html=True)

    with st.form("assessment_form"):
        # ── DEMOGRAPHICS ──────────────────────────────────────────────────────
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        st.markdown('<div class="form-card-title">👤 Demographics</div>', unsafe_allow_html=True)
        dc1, dc2, dc3, dc4 = st.columns(4)
        age          = dc1.number_input("Age (years)", min_value=1, max_value=100, value=55, step=1)
        gender       = dc2.selectbox("Gender", ["Female", "Male", "Other"])
        ever_married = dc3.selectbox("Ever Married?", ["Yes", "No"])
        residence    = dc4.selectbox("Residence Type", ["Urban", "Rural"])
        st.markdown('</div>', unsafe_allow_html=True)

        # ── CLINICAL ──────────────────────────────────────────────────────────
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        st.markdown('<div class="form-card-title">🏥 Clinical Measurements</div>', unsafe_allow_html=True)
        cc1, cc2, cc3, cc4 = st.columns(4)
        avg_glucose   = cc1.number_input("Avg Glucose Level (mg/dL)", min_value=50.0, max_value=350.0, value=90.0, step=0.5)
        bmi           = cc2.number_input("BMI", min_value=10.0, max_value=70.0, value=26.0, step=0.1)
        hypertension  = cc3.checkbox("Hypertension", value=False)
        heart_disease = cc4.checkbox("Heart Disease", value=False)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── LIFESTYLE ─────────────────────────────────────────────────────────
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        st.markdown('<div class="form-card-title">🌿 Lifestyle & Occupation</div>', unsafe_allow_html=True)
        lc1, lc2 = st.columns(2)
        work_type = lc1.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        smoking   = lc2.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("")
        submitted = st.form_submit_button("⚡  Predict Stroke Risk", use_container_width=True)

    if submitted:
        if not model_ready:
            st.error("Model not loaded. Please run the Milestone 3 notebook first.")
        else:
            inp = encode_input(age, gender, hypertension, heart_disease,
                               ever_married, work_type, residence,
                               avg_glucose, bmi, smoking)
            inp_df = pd.DataFrame([inp])[FEATURE_NAMES]
            scaled = scaler.transform(inp_df)
            prob   = float(model.predict_proba(scaled)[0, 1])

            with st.spinner("Computing LIME explanation…"):
                lime_feats, lime_vals = run_lime(scaled[0])

            st.session_state.result = {
                "prob": prob,
                "pred": int(prob >= THRESHOLD),
                "threshold": THRESHOLD,
                "lime_feats": lime_feats,
                "lime_vals": lime_vals.tolist(),
                "inputs": {
                    "Age": age, "Gender": gender,
                    "Avg Glucose": f"{avg_glucose:.1f} mg/dL",
                    "BMI": f"{bmi:.1f}",
                    "Hypertension": "Yes" if hypertension else "No",
                    "Heart Disease": "Yes" if heart_disease else "No",
                    "Work Type": work_type, "Smoking": smoking,
                    "Married": ever_married, "Residence": residence,
                },
            }
            st.session_state.page = "result"
            st.rerun()

    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("← Back to Home"):
            st.session_state.page = "home"
            st.rerun()

    st.markdown('</div></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: RESULT
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == "result" and st.session_state.result:
    r        = st.session_state.result
    prob     = r["prob"]
    pred     = r["pred"]
    thresh   = r["threshold"]
    pct      = prob * 100
    is_high  = pred == 1
    css_cls  = "high" if is_high else "low"
    risk_lbl = "HIGH RISK" if is_high else "LOW RISK"
    risk_ico = "⚠️" if is_high else "✅"

    st.markdown('<div class="result-wrap"><div class="result-shell">', unsafe_allow_html=True)

    # ── HERO RESULT CARD ──────────────────────────────────────────────────────
    gauge_fill_color = "#FF5252" if is_high else "#00E676"
    gauge_pct = int(pct)
    thresh_pct = int(thresh * 100)

    st.markdown(f"""
    <div class="result-hero {css_cls}">
      <div class="rh-badge">{risk_ico} {risk_lbl}</div>
      <div class="rh-pct">{pct:.1f}%</div>
      <div class="rh-label">Estimated stroke probability for this patient</div>
      <div style="font-size:0.78rem; color:rgba(255,255,255,0.35); margin-bottom:8px">
        Probability scale &nbsp;·&nbsp; Threshold at {thresh_pct}%
      </div>
      <div class="gauge-track">
        <div class="gauge-fill" style="width:{gauge_pct}%; background: linear-gradient(90deg, {'#FF5252' if is_high else '#00E676'}, {'#FF1744' if is_high else '#1DE9B6'})"></div>
        <div class="gauge-needle" style="left:{thresh_pct}%"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── METRIC STRIP ─────────────────────────────────────────────────────────
    rec  = TEST_METRICS.get("recall", 0)
    prec = TEST_METRICS.get("precision", 0)
    auc  = TEST_METRICS.get("auc_roc", 0)
    st.markdown(f"""
    <div class="metric-strip">
      <div class="mstrip-item">
        <div class="mstrip-val">{pct:.1f}%</div>
        <div class="mstrip-lbl">Stroke Probability</div>
      </div>
      <div class="mstrip-item">
        <div class="mstrip-val">{thresh*100:.0f}%</div>
        <div class="mstrip-lbl">Decision Threshold</div>
      </div>
      <div class="mstrip-item">
        <div class="mstrip-val">{float(rec):.2f}</div>
        <div class="mstrip-lbl">Model Recall</div>
      </div>
      <div class="mstrip-item">
        <div class="mstrip-val">{float(auc):.3f}</div>
        <div class="mstrip-lbl">Model AUC-ROC</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── CLINICAL RECOMMENDATION ───────────────────────────────────────────────
    st.markdown('<div class="rs-title">Clinical Recommendation</div>', unsafe_allow_html=True)
    if is_high:
        st.markdown(f"""
        <div class="rec-card urgent">
          <div class="rc-title">⚠️  Urgent Medical Referral Recommended</div>
          <div class="rc-text">
            This patient's predicted stroke probability of <strong style="color:#FF8A80">{pct:.1f}%</strong>
            exceeds the clinical screening threshold of {thresh_pct}%. The model is optimised
            for recall ≥ 0.80, meaning it errs toward catching at-risk patients rather than missing them.
            <br><br>
            <strong style="color:rgba(255,255,255,0.7)">Recommended actions:</strong><br>
            • Refer to a neurologist or stroke specialist for further evaluation<br>
            • Consider CT/MRI brain imaging and vascular assessment<br>
            • Assess and manage modifiable risk factors (hypertension, glucose, BMI)<br>
            • Review medication for hypertension and anticoagulation if appropriate
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="rec-card monitor">
          <div class="rc-title">✅  Low Risk — Continue Routine Monitoring</div>
          <div class="rc-text">
            This patient's predicted stroke probability of <strong style="color:#A5D6A7">{pct:.1f}%</strong>
            is below the screening threshold of {thresh_pct}%. No immediate intervention indicated
            based on the current data.
            <br><br>
            <strong style="color:rgba(255,255,255,0.7)">Recommended actions:</strong><br>
            • Maintain regular health check-ups and blood pressure monitoring<br>
            • Encourage healthy lifestyle: diet, physical activity, no smoking<br>
            • Reassess if clinical status changes (new hypertension, glucose changes)<br>
            • Annual stroke risk review for patients over 50
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── LIME CHART ────────────────────────────────────────────────────────────
    st.markdown('<div class="rs-title">Why this score? — LIME Local Explanation</div>', unsafe_allow_html=True)

    lime_feats = r["lime_feats"]
    lime_vals  = np.array(r["lime_vals"])

    st.markdown('<div class="lime-card">', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="#0F1829")
    colours = ["#FF5252" if v > 0 else "#40C4FF" for v in lime_vals]
    y_pos = np.arange(len(lime_feats))

    bars = ax.barh(y_pos, lime_vals, color=colours, edgecolor="none", height=0.58)
    for i, (v, c) in enumerate(zip(lime_vals, colours)):
        offset = 0.0008 if v >= 0 else -0.0008
        ax.text(v + offset, i, f"{v:+.4f}",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=8.5, color=c, fontweight="600",
                fontfamily="DejaVu Sans")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(lime_feats, fontsize=10, color=(1, 1, 1, 0.75))
    for label in ax.get_yticklabels():
        label.set_color("#B0BEC5")

    ax.axvline(0, color=(1, 1, 1, 0.2), lw=1)
    ax.set_xlabel("← Decreases risk          LIME Contribution          Increases risk →",
                  fontsize=9, color="#607D8B", labelpad=10)
    ax.set_title(f"Feature contributions for this patient  |  P(stroke) = {pct:.2f}%  |  Threshold = {thresh_pct}%",
                 fontsize=10, color="#90A4AE", pad=12, loc="left")

    ax.set_facecolor("#0F1829")
    fig.patch.set_facecolor("#0F1829")
    ax.grid(True, axis="x", alpha=0.1, color="white", linestyle="--")
    for spine in ax.spines.values():
        spine.set_color((1, 1, 1, 0.08))
        spine.set_linewidth(0.5)
    ax.tick_params(colors="#607D8B", labelsize=9)

    red  = mpatches.Patch(color="#FF5252", label="Increases stroke risk")
    blue = mpatches.Patch(color="#40C4FF", label="Decreases stroke risk")
    leg = ax.legend(handles=[red, blue], fontsize=8.5, loc="lower right",
                    framealpha=0.15, edgecolor=(1, 1, 1, 0.1),
                    labelcolor="white")

    # Auto-fit: draw first so matplotlib knows label widths, then adjust
    fig.canvas.draw()
    max_label_len = max(len(f) for f in lime_feats)
    left_margin = min(0.45, 0.18 + max_label_len * 0.012)
    plt.tight_layout(pad=1.5)
    plt.subplots_adjust(left=left_margin)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── FACTOR BARS ───────────────────────────────────────────────────────────
    st.markdown('<div class="rs-title">Risk Factor Breakdown</div>', unsafe_allow_html=True)
    st.markdown('<div class="lime-card">', unsafe_allow_html=True)

    abs_vals = np.abs(lime_vals)
    max_abs  = abs_vals.max() if abs_vals.max() > 0 else 1

    factors_html = ""
    for feat, val, absv in zip(lime_feats, lime_vals, abs_vals):
        bar_pct  = int(absv / max_abs * 100)
        is_risk  = val > 0
        bar_col  = "#FF5252" if is_risk else "#40C4FF"
        val_col  = "#FF8A80" if is_risk else "#80D8FF"
        direction_arrow = "↑" if is_risk else "↓"
        factors_html += f"""
        <div class="factor-row">
          <div class="factor-name">{feat}</div>
          <div class="factor-bar-bg">
            <div class="factor-bar-fill" style="width:{bar_pct}%; background:{bar_col}; opacity:0.8"></div>
          </div>
          <div class="factor-val" style="color:{val_col}">{direction_arrow} {absv:.4f}</div>
        </div>
        """
    st.markdown(factors_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── PATIENT INPUTS SUMMARY ────────────────────────────────────────────────
    st.markdown('<div class="rs-title">Patient Data Used for This Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="lime-card">', unsafe_allow_html=True)
    inp = r["inputs"]
    keys = list(inp.keys())
    num_cols = 5
    cols = st.columns(num_cols)
    for i, (k, v) in enumerate(inp.items()):
        cols[i % num_cols].markdown(f"""
        <div style="margin-bottom:14px">
          <div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:1px; color:rgba(255,255,255,0.3); margin-bottom:4px">{k}</div>
          <div style="font-size:0.95rem; font-weight:600; color:#E3F2FD">{v}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── DISCLAIMER ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
      ⚕️ <strong>Medical Disclaimer:</strong> This prediction is generated by a machine learning
      model trained on the Kaggle Stroke Prediction Dataset (5,110 patients). It is a research
      prototype for CS5998 Capstone and <strong>must not be used as a substitute for clinical
      diagnosis</strong>. Always consult a qualified neurologist or physician for medical decisions.
    </div>
    """, unsafe_allow_html=True)

    # ── NAVIGATION BUTTONS ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("← Back to Home", use_container_width=True):
            st.session_state.page = "home"
            st.session_state.result = None
            st.rerun()
    with b2:
        if st.button("🔄  Assess Another Patient", use_container_width=True):
            st.session_state.page = "assess"
            st.session_state.result = None
            st.rerun()

    st.markdown('</div></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="ns-footer">
      CS5998 Capstone · P.D.S. PERERA (258733L) · University of Moratuwa · Milestone 4
    </div>
    """, unsafe_allow_html=True)
