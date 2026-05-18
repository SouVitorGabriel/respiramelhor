from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from cachetools import cached, TTLCache
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

app = FastAPI(title="API Respira Melhor - UBS Vila Curuçá")

# Permite que o frontend (qualquer origem) consuma esta API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

RECOMENDACOES = {
    "Boa": "Ar em boa condição. A rotina e as atividades ao ar livre podem seguir normalmente.",
    "Moderada": "Pessoas mais sensíveis devem pegar leve em exercícios ao ar livre se houver incômodo.",
    "Ruim": "O ar pode causar desconforto, principalmente em pessoas sensíveis. Evite esforço forte na rua.",
    "Muito Ruim": "O ar pode piorar tosse, cansaço e falta de ar. Evite esforço ao ar livre e reduza a exposição.",
    "Péssima": "O risco à saúde está alto. Evite exposição prolongada e procure a UBS se os sintomas piorarem.",
}

NOMES_POL = {
    "MP2.5": "Partículas Finas", "MP10": "Partículas Inaláveis",
    "O3": "Ozônio", "NO2": "Dióxido de Nitrogênio",
    "CO": "Monóxido de Carbono", "SO2": "Dióxido de Enxofre",
}
POLUENTES = ["MP2.5", "MP10", "O3", "NO2", "CO", "SO2"]

FEATURES_MODELO = [
    "MP25", "MP10", "O3", "NO2", "hora_sin", "hora_cos", "dia_semana",
    "lag_1h", "lag_2h", "lag_3h", "lag_6h", "lag_12h", "lag_24h",
    "mp10_lag_1h", "mp10_lag_3h", "mp10_lag_24h",
    "o3_lag_1h", "o3_lag_3h", "o3_lag_24h",
    "no2_lag_1h", "no2_lag_3h", "no2_lag_24h",
    "media_3h", "media_6h", "diff_1h",
]

# Carrega o modelo de ML na memória ao iniciar a API
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

MODEL_XGB, FEATURES_XGB = carregar_modelo()

# ================================
# FUNÇÕES DE NEGÓCIO E DADOS
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

# Cache TTL de 600 segundos (10 minutos)
@cached(cache=TTLCache(maxsize=1, ttl=600))
def coletar_dados():
    dados = []
    for layer, pol in LAYERS.items():
        try:
            r = requests.get(f"{BASE_URL}/{layer}/query",
                           params={"where": "1=1", "outFields": "*", "f": "json"},
                           timeout=10)
            if r.status_code == 200:
                for f in r.json().get("features", []):
                    if ESTACAO_ALVO.lower() not in f["attributes"].get("STATNM", "").lower():
                        continue
                    for i in range(1, 49):
                        v = f["attributes"].get(f"M{i}")
                        t = f["attributes"].get(f"TM{i}")
                        if v is not None and t is not None:
                            dados.append({"poluente": pol, "valor": float(v),
                                          "datahora": pd.to_datetime(t, unit="ms")})
        except Exception:
            pass
    return pd.DataFrame(dados) if dados else pd.DataFrame()

def preparar_pivot(df):
    if df.empty:
        return pd.DataFrame()
    p = df.pivot_table(index="datahora", columns="poluente", values="valor", aggfunc="mean")
    p.columns.name = None
    return p.resample("1h").mean().ffill()

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
    try:
        h = pivot.rename(columns={"MP2.5": "MP25"}).copy().ffill().bfill()
        for p in ["MP25", "MP10", "O3", "NO2"]:
            if p not in h.columns: h[p] = 0.0
        mp25, mp10, o3, no2 = h["MP25"].copy(), h["MP10"].copy(), h["O3"].copy(), h["NO2"].copy()
        if len(mp25) == 0: return [], [], []
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
# ROTA PRINCIPAL DA API
# ================================
@app.get("/api/qualidade")
def get_qualidade_ar():
    df = coletar_dados()
    if df.empty:
        raise HTTPException(status_code=503, detail="Dados da CETESB indisponíveis no momento.")

    pivot = preparar_pivot(df)
    
    iqas = {}
    valores = {}
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
        raise HTTPException(status_code=503, detail="Não foi possível calcular o IQAr.")

    iqa_geral = max(iqas.values())
    pol_critico = max(iqas, key=iqas.get)
    nc, cc, bg = classificar(iqa_geral)
    
    # Prepara lista de poluentes para renderização dos cards
    lista_poluentes = []
    for pol in POLUENTES:
        nome = NOMES_POL.get(pol, pol)
        if pol in iqas:
            pnome, pcor, _ = classificar(iqas[pol])
            lista_poluentes.append({
                "nome": nome,
                "iqa": int(iqas[pol]),
                "valor": round(valores[pol], 1),
                "unidade": "ppm" if pol == "CO" else "µg/m³",
                "status": pnome,
                "cor_hex": pcor
            })
        else:
            lista_poluentes.append({"nome": nome, "iqa": None})

    # Dados para Gráficos
    fc_labels, fc_conc, fc_iqa = [], [], []
    if MODEL_XGB is not None:
        fc_labels, fc_conc, fc_iqa = gerar_forecast(pivot, MODEL_XGB, FEATURES_XGB)

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

    hist24_labels, hist24_vals = [], []
    if "MP2.5" in pivot.columns:
        h24 = pivot["MP2.5"].tail(24)
        for ts_, v_ in h24.items():
            hist24_labels.append(ts_.strftime("%Hh"))
            hist24_vals.append(round(float(v_), 1) if not pd.isna(v_) else None)

    return {
        "data_coleta": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "iqa_geral": int(iqa_geral),
        "classificacao": nc,
        "poluente_critico": NOMES_POL.get(pol_critico, pol_critico),
        "icone": "✅" if nc == "Boa" else ("⚠️" if nc == "Moderada" else "🚨"),
        "recomendacao_padrao": RECOMENDACOES.get(nc, ""),
        "cor_hex": cc,
        "cor_bg_fundo": bg,
        "poluentes": lista_poluentes,
        "graficos": {
            "historico_48h": {
                "labels": hist_labels,
                "mp25": hist_mp25,
                "mp10": hist_mp10,
                "o3": hist_o3,
                "no2": hist_no2
            },
            "previsao_mp25": {
                "hist24_labels": hist24_labels,
                "hist24_vals": hist24_vals,
                "fc_labels": fc_labels,
                "fc_conc": fc_conc,
                "fc_iqa": fc_iqa
            }
        }
    }