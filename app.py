import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
import qrcode
import base64
from io import BytesIO
import joblib
from pathlib import Path
import time
warnings.filterwarnings("ignore")

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Respira Melhor - UBS Vila Curuçá", layout="wide")

st.markdown("""
<style>
#MainMenu, footer, header, [data-testid="stDecoration"],
[data-testid="stToolbar"], [data-testid="stStatusWidget"],
[data-testid="stHeader"], .stDeployButton { display: none !important; }
.block-container { padding: 0 !important; margin: 0 !important; }
iframe{ height: 100vh !important; width: 100% !important; border: none !important; }            
.st-emotion-cache-tn0cau{gap:0 !important;}
</style>
""", unsafe_allow_html=True)

# ================================
# CONSTANTES
# ================================
BASE_URL = "https://arcgis.cetesb.sp.gov.br/server/rest/services/QUALAR_views/Qualidade_Ar_QUALAR/MapServer"
ESTACAO_ALVO = "Itaim Paulista"
LAYERS = {0: "CO", 1: "MP10", 2: "MP2.5", 3: "NO2", 4: "O3", 5: "SO2"}

LIMITES_IQA = {
    "MP2.5": [(0,0,15,40), (15,40,50,80), (50,80,75,120), (75,120,125,200), (125,200,300,400)],
    "MP10": [(0,0,45,40), (45,40,100,80), (100,80,150,120), (150,120,250,200), (250,200,600,400)],
    "O3": [(0,0,100,40), (100,40,130,80), (130,80,160,120), (160,120,200,200), (200,200,800,400)],
    "NO2": [(0,0,200,40), (200,40,240,80), (240,80,320,120), (320,120,1130,200), (1130,200,3750,400)],
    "CO": [(0,0,9,40), (9,40,11,80), (11,80,13,120), (13,120,15,200), (15,200,50,400)],
    "SO2": [(0,0,40,40), (40,40,50,80), (50,80,125,120), (125,120,800,200), (800,200,2620,400)],
}
CLASSES_IQA = [
    (0, 40, "Boa", "#27AE60", "#E8F8EF"),
    (41, 80, "Moderada", "#F39C12", "#FEF9E7"),
    (81, 120, "Ruim", "#E67E22", "#FEF0E7"),
    (121, 200, "Muito Ruim", "#E74C3C", "#FDEDEC"),
    (201, 999, "Péssima", "#8E44AD", "#F5EEF8"),
]
# Recomendações baseadas em OMS, EPA e CETESB
# Efeitos à saúde por faixa — base científica, não normativa
RECOMENDACOES = {
    "Boa":        "Ar em boa condição. A rotina e as atividades ao ar livre podem seguir normalmente.",
    "Moderada":   "Pessoas mais sensíveis devem pegar leve em exercícios ao ar livre se houver incômodo.",
    "Ruim": "O ar pode causar desconforto, principalmente em pessoas sensíveis. Evite esforço forte na rua.",
    "Muito Ruim": "O ar pode piorar tosse, cansaço e falta de ar. Evite esforço ao ar livre e reduza a exposição.",
    "Péssima":    "O risco à saúde está alto. Evite exposição prolongada e procure a UBS se os sintomas piorarem.",
}
NOMES_POL = {
    "MP2.5": "Partículas Finas", "MP10": "Partículas Inaláveis",
    "O3": "Ozônio", "NO2": "Dióxido de Nitrogênio",
    "CO": "Monóxido de Carbono", "SO2": "Dióxido de Enxofre",
}
POLUENTES = ["MP2.5", "MP10", "O3", "NO2", "CO", "SO2"]
SLIDE_NAMES = ["📊 Visão Geral", "🔮 Previsão (MP2.5)", "📋 Qualidade do Ar", "ℹ️ Sobre"]
INTERVALO_SEGUNDOS = 20
APP_URL = "https://respiramelhor.onrender.com/"

FEATURES_MODELO = [
    "MP25", "MP10", "O3", "NO2", "hora_sin", "hora_cos", "dia_semana",
    "lag_1h", "lag_2h", "lag_3h", "lag_6h", "lag_12h", "lag_24h",
    "mp10_lag_1h", "mp10_lag_3h", "mp10_lag_24h",
    "o3_lag_1h", "o3_lag_3h", "o3_lag_24h",
    "no2_lag_1h", "no2_lag_3h", "no2_lag_24h",
    "media_3h", "media_6h", "diff_1h",
]
RENAME_PARA_MODELO = {"MP2.5": "MP25"}

# ================================
# FUNÇÕES
# ================================
def calcular_iqa(valor, poluente):
    if valor is None or pd.isna(valor) or valor < 0:
        return None
    for cl, il, ch, ih in LIMITES_IQA.get(poluente, []):
        if cl <= valor <= ch:
            if ch == cl:
                return float(il)
            return ((ih - il) / (ch - cl)) * (valor - cl) + il
    return 400.0

def classificar(iqa):
    for lo, hi, nome, cor, bg in CLASSES_IQA:
        if lo <= iqa <= hi:
            return nome, cor, bg
    return "Péssima", "#8E44AD", "#F5EEF8"

def gerar_qr_base64(url: str) -> str:
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=6, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#1a1a2e", back_color="white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

@st.cache_data(ttl=600)
def coletar_dados():
    dados = []
    for layer, pol in LAYERS.items():
        try:
            r = requests.get(f"{BASE_URL}/{layer}/query",
                           params={"where": "1=1", "outFields": "*", "f": "json"},
                           timeout=10)
            r.raise_for_status()
            for f in r.json().get("features", []):
                if ESTACAO_ALVO.lower() not in f["attributes"].get("STATNM", "").lower():
                    continue
                for i in range(1, 49):
                    v = f["attributes"].get(f"M{i}")
                    t = f["attributes"].get(f"TM{i}")
                    if v is not None and t is not None:
                        dados.append({"poluente": pol, "valor": float(v),
                                      "datahora": pd.to_datetime(t, unit="ms")})
        except Exception as e:
            st.warning(f"Erro em {pol}: {e}")
    return pd.DataFrame(dados) if dados else pd.DataFrame()

def preparar_pivot(df):
    if df.empty:
        return pd.DataFrame()
    p = df.pivot_table(index="datahora", columns="poluente", values="valor", aggfunc="mean")
    p.columns.name = None
    return p.resample("1h").mean().ffill()

@st.cache_resource
def carregar_modelo():
    base = Path(__file__).parent
    mp = base / "modelo_xgboost.pkl"
    fp = base / "features.pkl"
    if not mp.exists():
        return None, None
    try:
        return joblib.load(mp), joblib.load(fp) if fp.exists() else FEATURES_MODELO
    except:
        return None, None

def preparar_features(hm, hmp10, ho3, hno2, ts, features):
    def sg(s, lag):
        return float(s.iloc[-lag]) if len(s) >= lag else (float(s.iloc[-1]) if len(s) > 0 else 0.0)
    ang = 2 * np.pi * ts.hour / 24
    v = hm.values
    u = float(v[-1]) if len(v) > 0 else 0.0
    row = {
        "MP25": u, "MP10": sg(hmp10, 24), "O3": sg(ho3, 24), "NO2": sg(hno2, 24),
        "hora_sin": np.sin(ang), "hora_cos": np.cos(ang), "dia_semana": ts.dayofweek,
        "lag_1h": sg(hm, 1), "lag_2h": sg(hm, 2), "lag_3h": sg(hm, 3),
        "lag_6h": sg(hm, 6), "lag_12h": sg(hm, 12), "lag_24h": sg(hm, 24),
        "mp10_lag_1h": sg(hmp10, 1), "mp10_lag_3h": sg(hmp10, 3), "mp10_lag_24h": sg(hmp10, 24),
        "o3_lag_1h": sg(ho3, 1), "o3_lag_3h": sg(ho3, 3), "o3_lag_24h": sg(ho3, 24),
        "no2_lag_1h": sg(hno2, 1), "no2_lag_3h": sg(hno2, 3), "no2_lag_24h": sg(hno2, 24),
        "media_3h": float(np.mean(v[-3:])) if len(v) >= 3 else u,
        "media_6h": float(np.mean(v[-6:])) if len(v) >= 6 else u,
        "diff_1h": float(v[-1] - v[-2]) if len(v) >= 2 else 0.0
    }
    return pd.DataFrame([row]).reindex(columns=features, fill_value=0.0).values

def gerar_forecast(pivot, model, features, horas=6):
    """Gera previsão XGBoost — retorna listas vazias em caso de erro."""
    try:
        h = pivot.rename(columns=RENAME_PARA_MODELO).copy().ffill().bfill()
        for p in ["MP25", "MP10", "O3", "NO2"]:
            if p not in h.columns:
                h[p] = 0.0
        mp25, mp10, o3, no2 = h["MP25"].copy(), h["MP10"].copy(), h["O3"].copy(), h["NO2"].copy()
        if len(mp25) == 0:
            return [], [], []
        labels, concs, iqas = [], [], []
        for _ in range(horas):
            prx = mp25.index[-1] + timedelta(hours=1)
            X = preparar_features(mp25, mp10, o3, no2, prx, features)
            pred = float(max(model.predict(X)[0], 0.0))
            _n = lambda s: float(s.iloc[-24]) if len(s) >= 24 else float(s.iloc[-1])
            mp25 = pd.concat([mp25, pd.Series([pred], index=[prx])])
            mp10 = pd.concat([mp10, pd.Series([_n(mp10)], index=[prx])])
            o3 = pd.concat([o3, pd.Series([_n(o3)], index=[prx])])
            no2 = pd.concat([no2, pd.Series([_n(no2)], index=[prx])])
            labels.append(prx.strftime("%Hh"))
            concs.append(round(pred, 1))
            iqa_v = calcular_iqa(pred, "MP2.5")
            iqas.append(round(iqa_v, 1) if iqa_v else 0.0)
        return labels, concs, iqas
    except Exception:
        return [], [], []

# ================================
# COLETA DE DADOS
# ================================
with st.spinner("Carregando dados da CETESB Itaim Paulista..."):
    df = coletar_dados()

if df.empty:
    st.error("Não foi possível carregar os dados da CETESB. Tente novamente em alguns minutos.")
    st.stop()

pivot = preparar_pivot(df)

iqas, valores = {}, {}
for pol in POLUENTES:
    if pol in pivot.columns:
        serie = pivot[pol].dropna()
        if not serie.empty:
            v = serie.iloc[-1]
            iqa = calcular_iqa(v, pol)
            if iqa is not None:
                iqas[pol] = iqa
                valores[pol] = v

if not iqas:
    st.error("Não foi possível calcular o IQAr. Dados insuficientes.")
    st.stop()

iqa_geral = max(iqas.values())
pol_critico = max(iqas, key=iqas.get)
nc, cc, bg = classificar(iqa_geral)
coleta_ts = datetime.now()
qr_b64 = gerar_qr_base64(APP_URL)

# Modelo + previsão
model_xgb, features_xgb = carregar_modelo()
fc_labels, fc_conc, fc_iqa = [], [], []
if model_xgb is not None:
    fc_labels, fc_conc, fc_iqa = gerar_forecast(pivot, model_xgb, features_xgb or FEATURES_MODELO)

# Histórico 48h
hist_labels, hist_mp25, hist_mp10, hist_o3, hist_no2 = [], [], [], [], []
cols_h = [p for p in ["MP2.5", "MP10", "O3", "NO2"] if p in pivot.columns]
if cols_h:
    hist_df = pivot[cols_h].tail(48).dropna(how="all")
    for ts_, row in hist_df.iterrows():
        hist_labels.append(ts_.strftime("%d/%b %Hh"))
        hist_mp25.append(round(row["MP2.5"], 1) if "MP2.5" in row and pd.notna(row["MP2.5"]) else None)
        hist_mp10.append(round(row["MP10"], 1)  if "MP10"  in row and pd.notna(row["MP10"])  else None)
        hist_o3.append(round(row["O3"], 1)      if "O3"    in row and pd.notna(row["O3"])    else None)
        hist_no2.append(round(row["NO2"], 1)    if "NO2"   in row and pd.notna(row["NO2"])   else None)

# Histórico 24h para slide de previsão
hist24_labels, hist24_vals = [], []
if "MP2.5" in pivot.columns:
    h24 = pivot["MP2.5"].tail(24)
    for ts_, v_ in h24.items():
        hist24_labels.append(ts_.strftime("%Hh"))
        hist24_vals.append(round(float(v_), 1) if not pd.isna(v_) else None)

# Cards HTML
cards_html = ""
for pol in POLUENTES:
    nome = NOMES_POL.get(pol, pol)
    if pol in iqas:
        iqa_v = iqas[pol]
        v = valores[pol]
        pnome, pcor, _ = classificar(iqa_v)
        un = "ppm" if pol == "CO" else "µg/m³"
        cards_html += f"""
        <div class="pol-card">
            <div class="pol-card-name">{nome}</div>
            <div class="pol-card-iqa" style="color:{pcor}">{iqa_v:.0f}</div>
            <div class="pol-card-value">{v:.1f} {un}</div>
            <div class="pol-card-status" style="color:{pcor}">{pnome}</div>
        </div>
        """
    else:
        cards_html += f"""
        <div class="pol-card pol-card-empty">
            <div class="pol-card-name">{nome}</div>
            <div class="pol-card-iqa pol-card-iqa-empty">--</div>
            <div class="pol-card-empty-text">Não monitorado</div>
        </div>
        """

icon_nc = "✅" if nc == "Boa" else ("⚠️" if nc == "Moderada" else "🚨")
tem_previsao = len(fc_labels) > 0
fc_labels_js = json.dumps(fc_labels)
fc_conc_js   = json.dumps(fc_conc)
fc_iqa_js    = json.dumps(fc_iqa)
all_labels_js= json.dumps(hist24_labels + fc_labels)
hist_data_js = json.dumps(hist24_vals + [None]*len(fc_labels))
fc_data_js   = json.dumps([None]*len(hist24_labels) + fc_conc)

hero_class_map = {
    "Boa": "hero-boa",
    "Moderada": "hero-moderada",
    "Ruim": "hero-ruim",
    "Muito Ruim": "hero-muito-ruim",
    "Péssima": "hero-pessima",
}
hero_class = hero_class_map.get(nc, "hero-pessima")

# ================================
# HTML COMPLETO
# ================================
# language=html
HTML_COMPLETO =  f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/toastify-js/src/toastify.min.css">
<script src="https://cdn.jsdelivr.net/npm/toastify-js"></script>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0,viewport-fit=cover">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
<style>
/* ================================
   1. RESET & GERAL
================================ */
* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}

html, body {{
    min-height: 100vh;
    width: 100%;
}}

body {{
    /* !important adicionado para barrar a fonte serifada do Streamlit */
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
    padding: 12px;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
    transition: background .25s ease, color .25s ease;
}}

body, button, input, textarea, select, table, th, td, div, span, p, a {{
    font-family: inherit;
    color: inherit;
}}

iframe, canvas {{
    color-scheme: inherit;
}}

a {{
    color: var(--primary);
    text-decoration: none;
}}

a:hover {{
    color: var(--primary-hover);
}}

/* ================================
   2. VARIÁVEIS DE TEMA
================================ */
:root {{
    color-scheme: light;
    --bg:#F0F2F5; --surface:#FFFFFF; --surface-2:#F8F9FA; --surface-3:#EEF2F6;
    --border:#E8ECF0; --border-soft:#DCE3EA;
    --text:#1A1A2E; --text-strong:#131322; --text-muted:#5A6575; --text-faint:#8A94A6;
    
    --primary:#185FA5; --primary-hover:#144C84; --primary-strong:#0F4A84; --primary-soft:#EBF3FD;
    
    /* Nova paleta Danger (Vermelho) - Tema Claro */
    --danger:#D32F2F; --danger-hover:#C62828; --danger-strong:#B71C1C; --danger-soft:#FFEBEE;
    
    --info-bg:#EBF3FD; --info-border:#185FA5;
    --table-head:#2C3E50; --table-head-text:#FFFFFF;
    --dot:#D0D5DD; --track:#E8ECF0;
    --good-bg:#DDF6E8; --moderate-bg:#FCEFCB; --bad-bg:#F9E1CC; --very-bad-bg:#F7D6D2; --terrible-bg:#E8DCF5;
    --shadow:0 8px 24px rgba(16,24,40,.06);
}}

html[data-theme="dark"] {{
    color-scheme: dark;
    --bg:#1E1D1C; --surface:#2B2A29; --surface-2:#323130; --surface-3:#3A3938;
    --border:#454341; --border-soft:#3A3836;
    --text:#F1ECE6; --text-strong:#FBF7F2; --text-muted:#C8C1B8; --text-faint:#A59D93;
    
    /* Paleta Primary ajustada para melhor contraste e vibração no escuro */
    --primary:#3B82F6; --primary-hover:#60A5FA; --primary-strong:#93C5FD; --primary-soft:#1E293B;
    
    /* Nova paleta Danger (Vermelho) - Tema Escuro */
    --danger:#EF5350; --danger-hover:#F44336; --danger-strong:#E53935; --danger-soft:#3B1818;
    
    /* Info border atualizada para combinar com o novo azul primário */
    --info-bg:#2C353D; --info-border:#3B82F6; 
    --table-head:#353B42; --table-head-text:#F6F7F8;
    --dot:#6A655F; --track:#454341;
    --good-bg:#22362B; --moderate-bg:#3D3424; --bad-bg:#433126; --very-bad-bg:#472B2A; --terrible-bg:#3B3143;
    --shadow:0 8px 24px rgba(0,0,0,.24);
}}

html[data-theme="light"] {{
    color-scheme: light;
}}

/* ================================
   3. ESTRUTURA E LAYOUT
================================ */
.content-wrap {{ padding-bottom:60px; }}
.cards-layout {{ display:flex; gap:16px; align-items:flex-start; margin-bottom:12px; }}
.card-left, .card-right {{ flex:1 1 0; min-width:0; box-sizing:border-box; }}
.cards-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:12px; margin-bottom:15px; }}
.slide-top-grid {{ margin-bottom: 12px; }}

/* ================================
   4. COMPONENTES COMUNS
================================ */
.card {{
    background:var(--surface); border-radius:16px; padding:16px;
    border:1px solid var(--border); box-shadow:var(--shadow);
}}
.badge {{
    display:inline-flex; align-items:center; gap:6px; padding:5px 14px;
    border-radius:30px; font-size:14px; font-weight:600;
    background:var(--surface-2); color:var(--text); border:1px solid var(--border);
}}
.title-text {{ font-size:16px; font-weight:700; margin-bottom:12px; color:var(--text-strong); }}

.section-card, .chart-card, .qr-card, .hero-banner, .aqr-switcher, .info-box {{ margin-bottom:12px; }}
.chart-wrap {{ height:260px; }}
.chart-wrap-lg {{ height:300px; }}

/* ================================
   5. HEADER
================================ */
.header-bar {{
    display:flex; justify-content:space-between; align-items:center; gap:12px;
    background:var(--surface); border-radius:12px; padding:12px 16px;
    margin-bottom:12px; border:1px solid var(--border); box-shadow:var(--shadow);
}}
.header-meta {{ display:flex; flex-direction:column; gap:2px; min-width:0; }}
.header-title {{ font-size:18px; font-weight:700; color:var(--text-strong); }}
.header-subtitle {{ font-size:11px; color:var(--text-faint); white-space:nowrap; }}
.header-actions {{ display:flex; align-items:center; gap:8px; flex-shrink:0; }}

.theme-toggle {{
    display:inline-flex; align-items:center; justify-content:center;
    min-width:40px; height:40px; padding:0 12px; border:none; border-radius:999px;
    background:var(--surface-2); color:var(--text); cursor:pointer;
    border:1px solid var(--border); transition:background .2s ease, transform .2s ease, border-color .2s ease;
}}
.theme-toggle:hover {{ background:var(--surface-3); border-color:var(--primary); }}
.theme-toggle:active {{ transform:translateY(1px); }}
.theme-toggle:focus-visible {{ outline:2px solid var(--primary); outline-offset:2px; }}

/* ================================
   6. CARDS DE DESTAQUE (HERO / SCORE)
================================ */
.hero-banner {{
    border-radius:14px; padding:16px; display:flex; align-items:center; 
    gap:16px; flex-wrap:wrap; border:1px solid var(--border); color:var(--text);
}}
.hero-banner-center {{ justify-content: center; text-align: center; }}

.hero-boa {{ background:var(--good-bg); border-color:color-mix(in srgb, #27AE60 38%, var(--border)); }}
.hero-moderada {{ background:var(--moderate-bg); border-color:color-mix(in srgb, #F39C12 38%, var(--border)); }}
.hero-ruim {{ background:var(--bad-bg); border-color:color-mix(in srgb, #E67E22 38%, var(--border)); }}
.hero-muito-ruim {{ background:var(--very-bad-bg); border-color:color-mix(in srgb, #E74C3C 38%, var(--border)); }}
.hero-pessima {{ background:var(--terrible-bg); border-color:color-mix(in srgb, #8E44AD 38%, var(--border)); }}

.hero-banner-score {{ font-size:48px; font-weight:700; color:var(--status-color); line-height:1; }}
.hero-banner-content {{ display:flex; flex-direction:column; gap:4px; }}
.hero-banner-title {{ font-size:18px; font-weight:700; color:var(--status-color); margin-bottom:6px; }}
.hero-banner-text {{ font-size:14px; color:var(--text-muted); line-height:1.5; }}

.card-left-top {{ display: grid; grid-template-columns: minmax(120px, 160px) 1fr; gap: 12px; margin-bottom: 15px; align-items: stretch; }}
.status-surface {{ border-radius: 14px; padding: 14px; margin-bottom: 0; border: 1px solid color-mix(in srgb, var(--accent-color) 38%, var(--border)); }}
.iqar-summary-card {{ display: flex; align-items: center; justify-content: center; text-align: center; }}
.iqar-summary-inner {{ display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 6px; width: 100%; }}
.iqar-eyebrow {{ font-size: 13px; font-weight: 600; line-height: 1.2; }}
.iqar-score {{ font-size: 52px; font-weight: 700; line-height: 1; color: var(--text-strong); }}
.status-accent-text {{ color: var(--accent-color) !important; }}

.status-badge {{
  margin-top: 2px;
  background: color-mix(in srgb, var(--accent-color) 14%, transparent);
  color: var(--accent-color) !important;
  border: 1px solid color-mix(in srgb, var(--accent-color) 28%, var(--border));
}}
.primary-pollutant-card {{ min-width: 0; }}
.slide-meta-text {{ margin-top: 8px; font-size: 13px; color: var(--text-faint); }}
.slide-eyebrow {{ font-size:14px; font-weight:600; margin-bottom:4px; }}
.slide-eyebrow-muted {{ color:var(--text-muted); }}

/* ================================
   7. CARDS DE POLUENTES
================================ */
.pol-card {{
    background:var(--surface); border-radius:14px; padding:14px;
    text-align:center; border:1px solid var(--border);
}}
.pol-card-empty {{ background:var(--surface-2); }}
.pol-card-name {{ font-size:14px; font-weight:600; color:var(--text-faint); }}
.pol-card-iqa {{ font-size:34px; font-weight:700; margin-top:4px; line-height:1.1; }}
.pol-card-iqa-empty {{ color:var(--text-faint); }}
.pol-card-value {{ font-size:13px; color:var(--text-muted); margin:4px 0; }}
.pol-card-status {{ font-size:13px; font-weight:600; }}
.pol-card-empty-text {{ font-size:12px; color:var(--text-faint); margin-top:8px; }}

/* ================================
   8. TABELAS
================================ */
.table-wrap {{ overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; background:var(--surface); border-radius:12px; overflow:hidden; }}
th, td {{ padding:10px 8px; text-align:center; font-size:13px; border-bottom:1px solid var(--border-soft); }}
th {{ background:var(--table-head); color:var(--table-head-text); font-weight:600; }}
td {{ color:var(--text); }}
td:first-child {{ text-align:left; font-weight:600; color:var(--text-strong); }}

.table-row-good {{ background:var(--good-bg); }}
.table-row-moderate {{ background:var(--moderate-bg); }}
.table-row-bad {{ background:var(--bad-bg); }}
.table-row-very-bad {{ background:var(--very-bad-bg); }}
.table-row-terrible {{ background:var(--terrible-bg); }}

/* ================================
   9. INFORMAÇÕES E BOTÕES
================================ */
.info-box {{
    background:var(--info-bg); border-left:4px solid var(--info-border); color:var(--text);
    border-radius:10px; padding:14px; font-size:14px; line-height:1.6;
}}
.info-box-title {{ font-size: 15px; color: var(--text-strong); margin-bottom: 6px; display: inline-block; }}

.note-box {{
    margin-top:10px; padding:10px 14px; background:var(--surface-2);
    border-radius:10px; border-left:3px solid var(--text-faint);
}}
.note-box span {{ font-size:12px; color:var(--text-faint); }}

.qr-card {{ display:flex; align-items:center; gap:20px; flex-wrap:wrap; }}
.qr-image {{ width:110px; height:110px; border-radius:12px; border:2px solid var(--border); flex-shrink:0; background:var(--surface-2); }}
.qr-content {{ display:flex; flex-direction:column; gap:4px; }}
.qr-title {{ font-size:16px; font-weight:700; color:var(--text-strong); }}
.qr-link {{ font-size:14px; color:var(--primary); }}
.qr-link:hover {{ color:var(--primary-hover); }}
.qr-help {{ font-size:13px; color:var(--text-faint); margin-top:4px; }}

.about-text {{ font-size:14px; color:var(--text-muted); line-height:1.6; }}
.about-text strong {{ color:var(--text-strong); }}

.aqr-switcher-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:10px; }}
.aqr-switcher-card {{ background:var(--surface); border-radius:12px; padding:12px; border:1px solid var(--border); }}
.aqr-switcher-title {{ font-size:14px; font-weight:700; margin-bottom:4px; }}
.aqr-switcher-text {{ font-size:12px; line-height:1.4; color:var(--text-muted); }}

.aqr-switcher-notice {{
    margin-top:12px; padding:12px 14px; background:var(--surface-2); border-left:3px solid var(--primary);
    border-radius:10px; font-size:12px; line-height:1.6; color:var(--text-muted);
}}
.aqr-switcher-btn {{
    margin-top:10px; width:100%; border:none; border-radius:12px; background:var(--primary);
    color:#fff; font-size:14px; font-weight:600; padding:12px 14px; cursor:pointer;
    transition:background .2s ease, transform .2s ease;
}}
.aqr-switcher-btn:hover {{ background:var(--primary-hover); }}
.aqr-switcher-btn:active {{ transform:translateY(1px); }}

.logoff-btn {{
    margin-top:10px; width:100%; border:none; border-radius:12px; background:var(--danger);
    color:#fff; font-size:14px; font-weight:600; padding:12px 14px; cursor:pointer;
    transition:background .2s ease, transform .2s ease;
}}
.logoff-btn:hover {{ background:var(--danger-hover); }}
.logoff-btn:active {{ transform:translateY(1px); }}

/* ================================
   10. MEDIA QUERIES
================================ */
@media (max-width:768px), (orientation:portrait) {{
    .cards-layout {{ flex-direction:column; }}
    .card-left, .card-right {{ width:100%; }}
    .table-wrap {{ overflow-x:auto; }}
    .header-title {{ font-size:14px!important; }}
    .header-bar {{ align-items:flex-start; }}
    .header-subtitle {{ white-space:normal; }}
    .theme-toggle {{ min-width:38px; height:38px; padding:0 10px; }}
    
    /* Mantém os cards lado a lado, mas dá uma margem menor pro card do score */
    .card-left-top {{ grid-template-columns: minmax(100px, 130px) 1fr; }}
    .iqar-summary-card {{ min-height: auto; }}
    .iqar-score {{ font-size: 48px; }}
    
    .cards-grid {{ grid-template-columns:repeat(3,1fr); gap:8px; }}
}}

@media(max-width:480px) {{
    .cards-grid {{ grid-template-columns:repeat(2,1fr); }}
}}
</style>
</head>
<body>
<div class="content-wrap">

<!-- HEADER -->
<div class="header-bar">
    <div class="header-meta">
        <span class="header-title">Respira Melhor • UBS Vila Curuçá</span>
        <span class="header-subtitle">CETESB • Itaim Paulista • {coleta_ts.strftime("%d/%m/%Y %H:%M")}</span>
    </div>

    <div class="header-actions">
        <span class="header-subtitle">Tema</span>
        <button
            id="themeToggleBtn"
            class="theme-toggle"
            type="button"
            aria-label="Alternar tema"
            title="Alternar tema"
        >🌙</button>
    </div>
</div>

<div class="cards-layout" id="cardsLayout1">
 <div id="card-left" class="card-left">
<div class="card-left-top">
    <div class="iqar-summary-card status-surface {hero_class}" style="--accent-color:{cc};">
      <div class="iqar-summary-inner">
        <div class="iqar-eyebrow status-accent-text">IQAr</div>
        <div class="iqar-score">{iqa_geral:.0f}</div>
        <div class="badge status-badge">{nc}</div>
      </div>
    </div>

    <div class="primary-pollutant-card status-surface {hero_class}" style="--accent-color:{cc};">
      <div class="hero-banner-title status-accent-text">{icon_nc} {nc}</div>
      <div class="hero-banner-text">
        {RECOMENDACOES.get(nc, "")}
      </div>
      <div class="slide-meta-text">
        Principal poluente: <b>{NOMES_POL.get(pol_critico, pol_critico)}</b>
      </div>
    </div>
  </div>


    <div class="card section-card">
        <div class="title-text">Tabela CONAMA 506/2024 vigente desde 01/01/2026</div>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Qualidade</th>
                        <th>IQAr</th>
                        <th>MP2.5</th>
                        <th>MP10</th>
                        <th>O₃</th>
                        <th>NO₂</th>
                        <th>CO</th>
                        <th>SO₂</th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="table-row-good"><td>🟢 N1 — Boa</td><td>0–40</td><td>0–15</td><td>0–45</td><td>0–100</td><td>0–200</td><td>0–9</td><td>0–40</td></tr>
                    <tr class="table-row-moderate"><td>🟡 N2 — Moderada</td><td>41–80</td><td>15–50</td><td>45–100</td><td>100–130</td><td>200–240</td><td>9–11</td><td>40–50</td></tr>
                    <tr class="table-row-bad"><td>🟠 N3 — Ruim</td><td>81–120</td><td>50–75</td><td>100–150</td><td>130–160</td><td>240–320</td><td>11–13</td><td>50–125</td></tr>
                    <tr class="table-row-very-bad"><td>🔴 N4 — Muito Ruim</td><td>121–200</td><td>75–125</td><td>150–250</td><td>160–200</td><td>320–1130</td><td>13–15</td><td>125–800</td></tr>
                    <tr class="table-row-terrible"><td>⚫ N5 — Péssima</td><td>201–400</td><td>125–300</td><td>250–600</td><td>200–800</td><td>1130–3750</td><td>15–50</td><td>800–2620</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <div id="aqr-switcher" class="aqr-switcher">
        <div class="aqr-switcher-grid" id="aqrCards"></div>
        <div class="aqr-switcher-notice" id="aqrNotice">
            Deslocamentos necessários, inclusive para atendimento de saúde, não devem ser adiados. Sempre que possível, reduza a exposição e observe o aparecimento de sintomas.
        </div>
        
    </div>
</div>

  <div id="card-right" class="card-right">
    <div class="info-box">
        <b>O que é o IQAr?</b><br>
        O Índice de Qualidade do Ar vai de <b>0 a 400</b>; quanto menor, melhor o ar.
        Sensores medem os poluentes a cada hora e uma fórmula linear os converte numa nota única, sempre a do poluente mais preocupante.
        Tabela vigente desde <b>01/01/2026</b> conforme <b>CONAMA 506/2024</b>.
    </div>

    <div class="card qr-card">
        <img src="{qr_b64}" class="qr-image" alt="QR Code para acessar o painel">
        <div class="qr-content">
            <div class="qr-title">Acesse pelo celular</div>
            <a href="{APP_URL}" target="_blank" class="qr-link">{APP_URL}</a>
            <div class="qr-help">Aponte a câmera para o QR Code</div>
        </div>
    </div>

    <div class="card">
        <div class="title-text">Sobre este painel</div>
        <div class="about-text">
            Serviço informativo da <strong>UBS Vila Curuçá</strong>. Baseado no Guia MMA/CETESB jan/2025 e CONAMA 506/2024.
            <br><br>
            <strong>Não substitui orientação médica.</strong><br>
            Situação muito ruim ou péssima e com sintomas graves: procure a <strong>UBS</strong>.
            <br><br>
            <strong>Grupos sensíveis:</strong>
            crianças, idosos, gestantes, asmáticos, pessoas com doenças respiratórias (DPOC, bronquite) ou cardiovasculares.
            <br><br>
            <span class="header-subtitle">
                Recomendações baseadas em diretrizes técnico-científicas da OMS, EPA e CETESB.
                Episódio Crítico: responsabilidade CETESB / CONAMA 491/2018, Art. 10–11.
            </span>
        </div>

        <div class="note-box">
            <span>
                Os dados são atualizados a cada hora e refletem a qualidade do ar nas últimas horas, não o instante exato da consulta.
            </span>
        </div>
        <button id="loginBtn" class="aqr-switcher-btn" type="button" aria-label="Acessar área logada">
            Acessar área logada
        </button>
    </div>
</div>
</div>

<div class="cards-layout" id="cardsLayout2" style="display:none;">
<div id="card-left" class="card-left">
  <div class="card-left-top">
    <div class="iqar-summary-card status-surface {hero_class}" style="--accent-color:{cc};">
      <div class="iqar-summary-inner">
        <div class="iqar-eyebrow status-accent-text">IQAr</div>
        <div class="iqar-score">{iqa_geral:.0f}</div>
        <div class="badge status-badge">{nc}</div>
      </div>
    </div>

    <div class="primary-pollutant-card status-surface {hero_class}" style="--accent-color:{cc};">
      <div class="hero-banner-title status-accent-text">{icon_nc} {nc}</div>
      <div class="hero-banner-text">
        {RECOMENDACOES.get(nc, "")}
      </div>
      <div class="slide-meta-text">
        Principal poluente: <b>{NOMES_POL.get(pol_critico, pol_critico)}</b>
      </div>
    </div>
  </div>

  <div class="cards-grid">{cards_html}</div>

  <div class="card chart-card">
    <div class="title-text">📈 Evolução dos poluentes — últimas 48h</div>
    <div class="chart-wrap">
      <canvas id="chart0"></canvas>
    </div>
  </div>
</div>

<div id="card-right" class="card-right">
  <div class="slide-top-grid">
    

    <div class="info-box">
      <b class="info-box-title">Previsão com IA (XGBoost)</b><br>
      {'Modelo treinado com dados históricos da CETESB. Prevê concentração de MP2.5 para as próximas 6 horas usando 25 features.' if tem_previsao else 'Arquivo modelo_xgboost.pkl não encontrado. Exibindo dados históricos apenas.'}
    </div>
  </div>

  <div class="card chart-card">
    <div class="title-text">MP2.5 histórico 24h + previsão 6h</div>
    <div class="chart-wrap">
      <canvas id="chart1"></canvas>
    </div>
  </div>

  <div class="card chart-card">
    <div class="title-text">IQAr previsto (MP2.5)</div>
    <div class="chart-wrap">
      <canvas id="chart2"></canvas>
    </div>
    <button id="logoffBtn" class="logoff-btn" type="button" aria-label="Acessar área logada">
            Sair área logada
        </button>
  </div>
</div>

</div>

<script>

(function () {{
  function getCssVar(name) {{
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }}

  const root = document.documentElement;
  const themeBtn = document.getElementById("themeToggleBtn");

  let currentTheme = "dark";
  root.setAttribute("data-theme", currentTheme);

  function syncThemeButton() {{
    if (!themeBtn) return;
    themeBtn.textContent = currentTheme === "dark" ? "☀️" : "🌙";
    themeBtn.setAttribute(
      "aria-label",
      currentTheme === "dark" ? "Ativar tema claro" : "Ativar tema escuro"
    );
    themeBtn.setAttribute(
      "title",
      currentTheme === "dark" ? "Ativar tema claro" : "Ativar tema escuro"
    );
  }}

  function applyChartTheme() {{
    Chart.defaults.font.family = "-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif";
    Chart.defaults.font.size = 11;
    Chart.defaults.color = getCssVar("--text-faint");
    Chart.defaults.borderColor = getCssVar("--border-soft");
  }}

  function refreshModelColors() {{
    modelos[0].cards[0].bg = getCssVar("--good-bg");
    modelos[0].cards[1].bg = getCssVar("--moderate-bg");
    modelos[0].cards[2].bg = getCssVar("--bad-bg");
    modelos[0].cards[3].bg = getCssVar("--very-bad-bg");
    modelos[0].cards[4].bg = getCssVar("--terrible-bg");

    modelos[1].cards[0].bg = getCssVar("--good-bg");
    modelos[1].cards[1].bg = getCssVar("--moderate-bg");
    modelos[1].cards[2].bg = getCssVar("--bad-bg");
    modelos[1].cards[3].bg = getCssVar("--very-bad-bg");
    modelos[1].cards[4].bg = getCssVar("--terrible-bg");

    modelos[2].cards[0].bg = getCssVar("--good-bg");
    modelos[2].cards[1].bg = getCssVar("--moderate-bg");
    modelos[2].cards[2].bg = getCssVar("--bad-bg");
    modelos[2].cards[3].bg = getCssVar("--very-bad-bg");
    modelos[2].cards[4].bg = getCssVar("--terrible-bg");
  }}

  function toggleTheme() {{
    currentTheme = currentTheme === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", currentTheme);
    syncThemeButton();
    applyChartTheme();
    renderModelo();
  }}

  if (themeBtn) {{
    themeBtn.addEventListener("click", toggleTheme);
  }}

  syncThemeButton();
  applyChartTheme();

  const modelos = [
    {{
      id: "sus",
      nome: "Institucional SUS",
      aviso: "Deslocamentos necessários, inclusive para atendimento de saúde, não devem ser adiados. Sempre que possível, reduza a exposição e observe o aparecimento de sintomas.",
      cards: [
        {{ bg:getCssVar("--good-bg"), color:"#27AE60", titulo:"🟢 Boa", texto:"Qualidade do ar satisfatória. Não há restrições para atividades habituais ao ar livre para a população em geral." }},
        {{ bg:getCssVar("--moderate-bg"), color:"#F39C12", titulo:"🟡 Moderada", texto:"Pessoas de grupos mais sensíveis, como crianças, idosos, gestantes e pessoas com doenças respiratórias ou cardíacas, devem reduzir esforço físico intenso ao ar livre se apresentarem desconforto." }},
        {{ bg:getCssVar("--bad-bg"), color:"#E67E22", titulo:"🟠 Ruim", texto:"A qualidade do ar pode causar sintomas em grupos sensíveis e desconforto em parte da população. Recomenda-se reduzir atividades físicas intensas ao ar livre e priorizar ambientes mais protegidos sempre que possível." }},
        {{ bg:getCssVar("--very-bad-bg"), color:"#E74C3C", titulo:"🔴 Muito Ruim", texto:"A exposição ao ar poluído pode agravar sintomas respiratórios e cardiovasculares. Reduza o tempo de permanência ao ar livre e evite esforço físico intenso, especialmente se houver sintomas." }},
        {{ bg:getCssVar("--terrible-bg"), color:"#8E44AD", titulo:"⚫ Péssima", texto:"Situação de maior risco à saúde. Evite exposição prolongada ao ar livre e atividades externas não essenciais; em caso de falta de ar, dor no peito, tontura ou piora importante de sintomas, procure uma UBS, UPA ou outro serviço de saúde." }}
      ]
    }},
    {{
      id: "popular",
      nome: "App popular",
      aviso: "Se você precisar sair para trabalhar, comprar alimentos ou ir à UBS, UPA ou hospital, não adie. O mais importante é reduzir a exposição e buscar ajuda se os sintomas piorarem.",
      cards: [
        {{ bg:getCssVar("--good-bg"), color:"#27AE60", titulo:"🟢 Boa", texto:"Ar em boa condição. A rotina e as atividades ao ar livre podem seguir normalmente." }},
        {{ bg:getCssVar("--moderate-bg"), color:"#F39C12", titulo:"🟡 Moderada", texto:"Pessoas mais sensíveis devem pegar leve em exercícios ao ar livre se houver incômodo." }},
        {{ bg:getCssVar("--bad-bg"), color:"#E67E22", titulo:"🟠 Ruim", texto:"O ar pode causar desconforto, principalmente em pessoas sensíveis. Evite esforço forte na rua." }},
        {{ bg:getCssVar("--very-bad-bg"), color:"#E74C3C", titulo:"🔴 Muito Ruim", texto:"O ar pode piorar tosse, cansaço e falta de ar. Evite esforço ao ar livre e reduza a exposição." }},
        {{ bg:getCssVar("--terrible-bg"), color:"#8E44AD", titulo:"⚫ Péssima", texto:"O risco à saúde está alto. Evite exposição prolongada e procure a UBS se os sintomas piorarem." }}
      ]
    }},
    {{
      id: "tecnico",
      nome: "Técnica jurídica",
      aviso: "Este aviso tem caráter informativo e preventivo. Deslocamentos indispensáveis, inclusive para atendimento em UBS, UPA, hospital ou outros serviços de saúde, não devem ser postergados; na ocorrência de falta de ar, dor torácica, tontura ou piora importante de sintomas, recomenda-se avaliação por serviço de saúde.",
      cards: [
        {{ bg:getCssVar("--good-bg"), color:"#27AE60", titulo:"🟢 Boa", texto:"Qualidade do ar classificada como satisfatória, sem indicação de restrições adicionais para atividades habituais ao ar livre na população em geral." }},
        {{ bg:getCssVar("--moderate-bg"), color:"#F39C12", titulo:"🟡 Moderada", texto:"Indivíduos de maior susceptibilidade, incluindo crianças, idosos, gestantes e pessoas com doenças respiratórias ou cardiovasculares, devem considerar a redução de esforço físico intenso ao ar livre na presença de sintomas ou desconforto." }},
        {{ bg:getCssVar("--bad-bg"), color:"#E67E22", titulo:"🟠 Ruim", texto:"A condição do ar pode produzir efeitos adversos em grupos sensíveis e desconforto em parte da população. Recomenda-se reduzir atividades físicas intensas em ambiente externo e priorizar, sempre que viável, ambientes com menor exposição." }},
        {{ bg:getCssVar("--very-bad-bg"), color:"#E74C3C", titulo:"🔴 Muito Ruim", texto:"A exposição ambiental pode favorecer agravamento de manifestações respiratórias e cardiovasculares. Recomenda-se redução do tempo de exposição ao ar livre e evitar esforço físico intenso, especialmente entre pessoas sintomáticas ou mais vulneráveis." }},
        {{ bg:getCssVar("--terrible-bg"), color:"#8E44AD", titulo:"⚫ Péssima", texto:"Condição associada a maior probabilidade de efeitos adversos à saúde. Recomenda-se evitar exposição prolongada ao ar livre e atividades externas não essenciais, sem prejuízo de deslocamentos necessários, inclusive para acesso a serviços de saúde quando houver sinais de agravamento clínico." }}
      ]
    }}
  ];

  let indiceAtual = 1;

  const cardsEl = document.getElementById("aqrCards");
  const noticeEl = document.getElementById("aqrNotice");
  const btnEl = document.getElementById("aqrToggleBtn");
  const btLogin = document.getElementById("loginBtn");
  const btnLogoff = document.getElementById("logoffBtn");
  const cardLayout1 = document.getElementById("cardsLayout1");
  const cardLayout2 = document.getElementById("cardsLayout2");
  

    if (btLogin) {{
    btLogin.addEventListener("click", login);
  }}
    if (btnLogoff) {{
    btnLogoff.addEventListener("click", logoff);
  }}

  function login() {{ 
     let senha = prompt("Área logada. Digite a senha de acesso:");
      if (senha === "vila123") {{
        cardLayout1.style.display = "none";
        cardLayout2.style.display = "flex";
        window.scrollTo(0, 0);
      }}
      else{{
        alert("Senha incorreta. Acesso negado.");
      }}
  }}

  function logoff() {{
    cardLayout1.style.display = "flex";
    cardLayout2.style.display = "none";
    window.scrollTo(0, 0);
  }}

  function renderModelo() {{
    refreshModelColors();
    const modelo = modelos[indiceAtual];

    cardsEl.innerHTML = modelo.cards.map(card => `
      <div class="aqr-switcher-card" style="background:${{card.bg}}">
        <div class="aqr-switcher-title" style="color:${{card.color}}">${{card.titulo}}</div>
        <div class="aqr-switcher-text">${{card.texto}}</div>
      </div>
    `).join("");

    noticeEl.textContent = modelo.aviso;
  }}

  function mostrarToast() {{
    const modelo = modelos[indiceAtual];
    Toastify({{
      text: `Linguagem ativa: ${{modelo.nome}}`,
      duration: 2400,
      gravity: "top",
      position: "right",
      close: true,
      stopOnFocus: true,
      style: {{
        background: "#1f2937",
        color: "#ffffff",
        borderRadius: "12px"
      }}
    }}).showToast();
  }}

  function trocarModelo() {{
    indiceAtual = (indiceAtual + 1) % modelos.length;
    renderModelo();
    mostrarToast();
  }}

  if (btnEl) {{
    btnEl.addEventListener("click", trocarModelo);
  }}

  if (noticeEl) {{
    noticeEl.addEventListener("click", trocarModelo);
  }}

  renderModelo();

  const baseOpts = {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      tooltip: {{ mode:"index", intersect:false }}
    }},
    scales: {{
      x: {{
        grid: {{ display:false }},
        ticks: {{ maxRotation:30, maxTicksLimit:8, color:getCssVar("--text-faint") }}
      }},
      y: {{
        grid: {{ color:getCssVar("--border-soft") }},
        ticks: {{ color:getCssVar("--text-faint") }}
      }}
    }}
  }};

  new Chart(document.getElementById("chart0"), {{
    type: "line",
    data: {{
      labels: {json.dumps(hist_labels)},
      datasets: [
        {{ label:"MP2.5", data:{json.dumps(hist_mp25)}, borderColor:"#2980B9", borderWidth:2.5, pointRadius:0, tension:0.4, fill:false }},
        {{ label:"MP10",  data:{json.dumps(hist_mp10)}, borderColor:"#27AE60", borderWidth:2.5, pointRadius:0, tension:0.4, fill:false }},
        {{ label:"O₃",    data:{json.dumps(hist_o3)},   borderColor:"#E74C3C", borderWidth:2.5, pointRadius:0, tension:0.4, fill:false }},
        {{ label:"NO₂",   data:{json.dumps(hist_no2)},  borderColor:"#F39C12", borderWidth:2.5, pointRadius:0, tension:0.4, fill:false }}
      ]
    }},
    options: {{
      ...baseOpts,
      plugins: {{
        ...baseOpts.plugins,
        legend: {{
          position:"bottom",
          labels: {{
            font: {{ size:11 }},
            color:getCssVar("--text-faint")
          }}
        }}
      }}
    }}
  }});

  const allLabels = {all_labels_js};
  const histData  = {hist_data_js};
  const fcData    = {fc_data_js};
  const fcLabels  = {fc_labels_js};
  const fcIqa     = {fc_iqa_js};

  if (allLabels.length > 0) {{
    new Chart(document.getElementById("chart1"), {{
      type: "line",
      data: {{
        labels: allLabels,
        datasets: [
          {{ label:"Medido (24h)", data:histData, borderColor:"#2980B9", borderWidth:2.5, pointRadius:0, tension:0.4, fill:false }},
          {{ label:"Previsão (6h)", data:fcData, borderColor:"#E67E22", borderWidth:3, borderDash:[6,4], pointRadius:4, pointBackgroundColor:"#E67E22", tension:0.4, fill:false }}
        ]
      }},
      options: {{
        ...baseOpts,
        plugins: {{
          ...baseOpts.plugins,
          legend: {{
            position:"bottom",
            labels: {{
              font: {{ size:11 }},
              color:getCssVar("--text-faint")
            }}
          }}
        }},
        scales: {{
          ...baseOpts.scales,
          y: {{
            ...baseOpts.scales.y,
            title: {{
              display:true,
              text:"µg/m³",
              font: {{ size:11 }},
              color:getCssVar("--text-faint")
            }}
          }}
        }}
      }}
    }});
  }}

  if (fcLabels.length > 0) {{
    const iqa_max = Math.max(...fcIqa.filter(v => v !== null));
    const yMax = Math.max(90, iqa_max + 15);

    new Chart(document.getElementById("chart2"), {{
      type: "line",
      data: {{
        labels: fcLabels,
        datasets: [
          {{
            label:"_boa",
            data: fcLabels.map(() => 40),
            borderWidth:0,
            pointRadius:0,
            backgroundColor:"rgba(39,174,96,0.12)",
            fill:{{ target:"origin", above:"rgba(39,174,96,0.12)" }},
            tension:0,
            order:5
          }},
          {{
            label:"_mod",
            data: fcLabels.map(() => 80),
            borderWidth:0,
            pointRadius:0,
            backgroundColor:"rgba(243,156,18,0.12)",
            fill:{{ target:{{ value:40 }}, above:"rgba(243,156,18,0.12)" }},
            tension:0,
            order:4
          }},
          {{
            label:"_ruim",
            data: fcLabels.map(() => 120),
            borderWidth:0,
            pointRadius:0,
            backgroundColor:"rgba(230,126,34,0.12)",
            fill:{{ target:{{ value:80 }}, above:"rgba(230,126,34,0.12)" }},
            tension:0,
            order:3
          }},
          {{
            label:"_mruim",
            data: fcLabels.map(() => 200),
            borderWidth:0,
            pointRadius:0,
            backgroundColor:"rgba(231,76,60,0.12)",
            fill:{{ target:{{ value:120 }}, above:"rgba(231,76,60,0.12)" }},
            tension:0,
            order:2
          }},
          {{
            label:"IQAr previsto",
            data:fcIqa,
            borderColor:"#8E44AD",
            borderWidth:3,
            pointRadius:5,
            pointBackgroundColor:"#8E44AD",
            tension:0.3,
            fill:false,
            order:1
          }}
        ]
      }},
      options: {{
        responsive:true,
        maintainAspectRatio:false,
        plugins: {{
          legend: {{ display:false }},
          tooltip: {{
            mode:"index",
            intersect:false,
            filter: item => item.dataset.label === "IQAr previsto"
          }},
          annotation: {{
            annotations: {{
              lBoa: {{
                type:"line",
                yMin:40,
                yMax:40,
                borderColor:"rgba(39,174,96,0.5)",
                borderWidth:1,
                borderDash:[4,4],
                label:{{ content:"Boa", display:true, position:"end", color:"#27AE60", font:{{size:10}}, padding:2 }}
              }},
              lMod: {{
                type:"line",
                yMin:80,
                yMax:80,
                borderColor:"rgba(243,156,18,0.5)",
                borderWidth:1,
                borderDash:[4,4],
                label:{{ content:"Moderada", display:true, position:"end", color:"#F39C12", font:{{size:10}}, padding:2 }}
              }},
              lRuim: {{
                type:"line",
                yMin:120,
                yMax:120,
                borderColor:"rgba(230,126,34,0.5)",
                borderWidth:1,
                borderDash:[4,4],
                label:{{ content:"Ruim", display:true, position:"end", color:"#E67E22", font:{{size:10}}, padding:2 }}
              }}
            }}
          }}
        }},
        scales: {{
          x: {{
            grid:{{ display:false }},
            ticks:{{ maxRotation:0, font:{{ size:11 }}, color:getCssVar("--text-faint") }}
          }},
          y: {{
            grid: {{ color:getCssVar("--border-soft") }},
            min: 0,
            max: yMax,
            title: {{ display:true, text:"IQAr MP2.5", font:{{ size:11 }}, color:getCssVar("--text-faint") }},
            ticks: {{ color:getCssVar("--text-faint") }}
          }}
        }}
      }}
    }});
  }}

  goTo(0);
}})();
</script>
</body></html>"""

# ================================
# EXIBIR
# ================================
st.iframe(HTML_COMPLETO, height="stretch", width="stretch")
# Recarregar dados a cada 10 minutos — carrossel roda em JS, sem piscar
time.sleep(600)
st.rerun()