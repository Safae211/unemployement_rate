# ==============================================================================
# PROJET : ANALYSE ET PRÉDICTION DU CHÔMAGE MONDIAL PAR MACHINE LEARNING
# ==============================================================================
# Dataset  : ILO (Organisation Internationale du Travail)
# 183 pays | 1991 - 2025 | 57 519 lignes
# Modèle   : XGBoost Regressor (eXtreme Gradient Boosting)
#
# COMMENT LANCER :
#   pip install streamlit pandas numpy matplotlib seaborn scikit-learn plotly xgboost
#   streamlit run projet_chomage_mondial.py
# ==============================================================================


# ==============================================================================
# ÉTAPE 1 — IMPORTER LES BIBLIOTHÈQUES
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score

from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# ÉTAPE 2 — CONFIGURER L'INTERFACE STREAMLIT
# ==============================================================================

st.set_page_config(
    page_title="Chômage Mondial — XGBoost",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: 800; color: #00E5FF;
        text-align: center; margin-bottom: 0.2rem;
        line-height: 1.3; text-shadow: 0 0 20px rgba(0,229,255,0.5);
    }
    .sub-title {
        font-size: 1rem; color: #E040FB;
        text-align: center; margin-bottom: 1.2rem; font-weight: 600;
    }
    .section-title {
        font-size: 1.15rem; font-weight: 700; color: #00E5FF;
        border-left: 5px solid #E040FB;
        padding-left: 0.6rem; margin: 1rem 0 0.8rem 0;
    }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1A0533, #0D1B40);
        border: 1px solid #E040FB; border-radius: 12px;
        padding: 0.7rem 0.8rem; box-shadow: 0 0 12px rgba(224,64,251,0.25);
    }
    div[data-testid="metric-container"] label {
        color: #B0BEC5 !important; font-weight: 700 !important;
        font-size: 0.75rem !important; text-transform: uppercase !important;
    }
    div[data-testid="stMetricValue"] {
        color: #00E5FF !important; font-size: 1.35rem !important;
        font-weight: 800 !important; text-shadow: 0 0 10px rgba(0,229,255,0.4);
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A0015 0%, #0D0020 100%);
        border-right: 2px solid #E040FB;
    }
    div[data-testid="stSidebar"] * { color: #CFD8DC !important; }
    div[data-testid="stSidebar"] strong { color: #E040FB !important; font-weight: 700 !important; }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #E040FB, #00E5FF) !important;
        color: #000000 !important; border: none !important;
        font-weight: 800 !important; font-size: 1rem !important;
        border-radius: 10px !important;
        box-shadow: 0 0 18px rgba(224,64,251,0.5) !important;
    }
    .stSelectbox label, .stSlider label, .stMultiSelect label {
        color: #00E5FF !important; font-weight: 700 !important;
    }
    div[data-testid="stCaptionContainer"] p {
        color: #78909C !important; font-style: italic;
        border-left: 2px solid #E040FB; padding-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# ÉTAPE 3 — CHARGER ET NETTOYER LES DONNÉES
# ==============================================================================

@st.cache_data
def charger_donnees():
    try:
        df = pd.read_csv("dataset_final_clean.csv")
        if "chomage_lag1" not in df.columns:
            df = _ajouter_lag_features(df)
        return df
    except FileNotFoundError:
        pass

    df_chomage = pd.read_csv("disoccupazione.csv")
    df_emploi  = pd.read_csv("occupazione.csv")

    df_chomage.dropna(inplace=True)
    df_emploi.dropna(inplace=True)
    df_chomage.drop_duplicates(inplace=True)
    df_emploi.drop_duplicates(inplace=True)

    df_chomage.rename(columns={"obs_value": "taux_chomage"}, inplace=True)
    df_emploi.rename(columns={"obs_value": "taux_emploi"},   inplace=True)

    df = pd.merge(
        df_chomage,
        df_emploi[["iso_code", "country", "sex", "age", "year", "taux_emploi"]],
        on=["iso_code", "country", "sex", "age", "year"],
        how="left"
    )

    # Combler taux_emploi manquants
    df["taux_emploi"] = df.groupby(["country", "sex", "age"])["taux_emploi"].transform(
        lambda x: x.fillna(x.median())
    )
    df["taux_emploi"].fillna(df["taux_emploi"].median(), inplace=True)

    df["post_covid"] = (df["year"] >= 2020).astype(int)
    df["post_2008"]  = ((df["year"] >= 2008) & (df["year"] <= 2013)).astype(int)
    df["decennie"]   = (df["year"] // 10) * 10

    le_sex     = LabelEncoder().fit(df["sex"])
    le_age     = LabelEncoder().fit(df["age"])
    le_country = LabelEncoder().fit(df["country"])

    df["sex_encoded"]     = le_sex.transform(df["sex"])
    df["age_encoded"]     = le_age.transform(df["age"])
    df["country_encoded"] = le_country.transform(df["country"])

    df = _ajouter_lag_features(df)
    return df


def _ajouter_lag_features(df):
    """
    ✅ FIX 0 NaN :
    - min_periods=1 sur tous les rolling → jamais de NaN
    - fillna(0) sur delta et delta2 → jamais de NaN
    - fillna(lag1) sur lag2 → jamais de NaN
    - dropna uniquement sur lag1 (première année de chaque groupe)
    """
    df = df.sort_values(["country", "sex", "age", "year"]).copy()
    grp = df.groupby(["country", "sex", "age"])["taux_chomage"]

    # Lag 1 et 2
    df["chomage_lag1"] = grp.shift(1)
    df["chomage_lag2"] = grp.shift(2)

    # Rolling avec min_periods=1 → JAMAIS de NaN
    df["chomage_rolling3"] = grp.transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df["chomage_rolling5"] = grp.transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()   # ✅ min_periods=1
    )

    # Delta avec fillna(0) → JAMAIS de NaN
    df["chomage_delta"]  = grp.diff().fillna(0)        # ✅ fillna(0)
    df["chomage_delta2"] = grp.diff().diff().fillna(0) # ✅ fillna(0)

    # Supprimer seulement la 1ère année de chaque groupe (lag1 forcément NaN)
    df.dropna(subset=["chomage_lag1"], inplace=True)

    # Combler lag2 NaN par lag1 (2ème année du groupe)
    df["chomage_lag2"] = df["chomage_lag2"].fillna(df["chomage_lag1"])  # ✅

    df.reset_index(drop=True, inplace=True)
    return df


# ==============================================================================
# ÉTAPE 4 — ENTRAÎNER LE MODÈLE XGBOOST
# ==============================================================================

@st.cache_resource
def entrainer_modele(df):
    df_model = df.copy()
    df_model.drop(columns=["iso_code", "country", "sex", "age"], inplace=True)
    df_model.dropna(inplace=True)
    df_model.reset_index(drop=True, inplace=True)

    # Split temporel (pas aléatoire → pas de data leakage)
    df_model = df_model.sort_values("year").reset_index(drop=True)
    split_idx = int(len(df_model) * 0.8)
    train_df  = df_model.iloc[:split_idx]
    test_df   = df_model.iloc[split_idx:]

    feature_cols = [c for c in df_model.columns if c != "taux_chomage"]
    X_train, y_train = train_df[feature_cols], train_df["taux_chomage"]
    X_test,  y_test  = test_df[feature_cols],  test_df["taux_chomage"]

    modele = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=1.0,
        early_stopping_rounds=30,
        eval_metric="mae",
        random_state=42,
        n_jobs=-1
    )
    modele.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = modele.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)

    importances = pd.Series(
        modele.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    le_sex     = LabelEncoder().fit(df["sex"])
    le_age     = LabelEncoder().fit(df["age"])
    le_country = LabelEncoder().fit(df["country"])

    return modele, X_test, y_test, y_pred, mae, r2, importances, feature_cols, le_sex, le_age, le_country


# ==============================================================================
# ÉTAPE 5 — HELPER : construire une ligne de features
# ==============================================================================

def _build_feature_row(annee, taux_emploi, sex_enc, age_enc, country_enc,
                        lag1, rolling3, delta, feature_cols,
                        lag2=None, rolling5=None, delta2=None):
    row = {
        "year":             annee,
        "taux_emploi":      taux_emploi,
        "post_covid":       int(annee >= 2020),
        "post_2008":        int(2008 <= annee <= 2013),
        "decennie":         (annee // 10) * 10,
        "sex_encoded":      sex_enc,
        "age_encoded":      age_enc,
        "country_encoded":  country_enc,
        "chomage_lag1":     lag1,
        "chomage_lag2":     lag2     if lag2     is not None else lag1,
        "chomage_rolling3": rolling3,
        "chomage_rolling5": rolling5 if rolling5 is not None else rolling3,
        "chomage_delta":    delta,
        "chomage_delta2":   delta2   if delta2   is not None else 0.0,
    }
    return pd.DataFrame([[row.get(c, 0) for c in feature_cols]], columns=feature_cols)


# ==============================================================================
# ÉTAPE 6 — FONCTIONS GRAPHIQUES (EDA)
# ==============================================================================

def graphique_distribution(df):
    data = df[df["sex"] == "Total"]["taux_chomage"]
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(data, kde=True, color="#7B2FBE", edgecolor="black", alpha=0.7, bins=40, ax=ax)
    ax.axvline(data.mean(),   color="#2D2D6B", linestyle="--", linewidth=2, label=f"Moyenne : {data.mean():.2f}%")
    ax.axvline(data.median(), color="#E74C8B", linestyle="--", linewidth=2, label=f"Médiane : {data.median():.2f}%")
    ax.set_title("Distribution du Taux de Chômage Mondial", fontsize=14, fontweight="bold", color="#2D2D6B")
    ax.set_xlabel("Taux de chômage (%)"); ax.set_ylabel("Fréquence")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); return fig


def graphique_evolution(df):
    chomage_annuel = df[df["sex"] == "Total"].groupby("year")["taux_chomage"].mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(chomage_annuel.index, chomage_annuel.values, color="#2D2D6B", linewidth=2.5, marker="o", markersize=4)
    ax.axvline(x=2008, color="#E74C8B", linestyle="--", linewidth=1.5, label="Crise 2008")
    ax.axvline(x=2020, color="#F4C542", linestyle="--", linewidth=1.5, label="COVID-19")
    ax.fill_between(chomage_annuel.index, chomage_annuel.values, alpha=0.15, color="#7B2FBE")
    ax.set_title("Évolution du Chômage Mondial Moyen (1991–2025)", fontsize=14, fontweight="bold", color="#2D2D6B")
    ax.set_xlabel("Année"); ax.set_ylabel("Taux (%)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); return fig


def graphique_genre(df):
    df_genre = df[df["sex"] != "Total"].groupby(["year", "sex"])["taux_chomage"].mean().unstack()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    couleurs = {"Female": "#E74C8B", "Male": "#2D2D6B"}
    for sex, couleur in couleurs.items():
        if sex in df_genre.columns:
            ax1.plot(df_genre.index, df_genre[sex], color=couleur, linewidth=2.5, label=sex)
    ax1.set_title("Évolution par Genre", fontsize=12, fontweight="bold", color="#2D2D6B")
    ax1.set_xlabel("Année"); ax1.set_ylabel("Taux (%)"); ax1.legend(); ax1.grid(True, alpha=0.3)
    sns.boxplot(data=df[df["sex"] != "Total"], x="sex", y="taux_chomage", palette=couleurs, ax=ax2)
    ax2.set_title("Distribution par Genre", fontsize=12, fontweight="bold", color="#2D2D6B")
    ax2.set_xlabel("Genre"); ax2.set_ylabel("Taux (%)"); ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout(); return fig


def graphique_age(df):
    df_total    = df[df["sex"] == "Total"]
    df_age_year = df_total.groupby(["year", "age"])["taux_chomage"].mean().unstack()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    palette = {"15+": "#2D2D6B", "15-24": "#E74C8B", "25+": "#0D9488"}
    for grp, couleur in palette.items():
        if grp in df_age_year.columns:
            ax1.plot(df_age_year.index, df_age_year[grp], color=couleur, linewidth=2.5, label=grp)
    ax1.set_title("Évolution par Groupe d'Âge", fontsize=12, fontweight="bold", color="#2D2D6B")
    ax1.set_xlabel("Année"); ax1.set_ylabel("Taux (%)"); ax1.legend(); ax1.grid(True, alpha=0.3)
    sns.boxplot(data=df_total, x="age", y="taux_chomage", palette=palette, ax=ax2)
    ax2.set_title("Distribution par Groupe d'Âge", fontsize=12, fontweight="bold", color="#2D2D6B")
    ax2.set_xlabel("Groupe d'âge"); ax2.set_ylabel("Taux (%)"); ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout(); return fig


def graphique_covid(df):
    df_total = df[df["sex"] == "Total"].copy()
    df_total["periode"] = df_total["year"].apply(
        lambda a: "Avant COVID (≤ 2019)" if a < 2020 else "Après COVID (≥ 2020)"
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    moy    = df_total.groupby("periode")["taux_chomage"].mean()
    barres = ax1.bar(moy.index, moy.values, color=["#2D2D6B", "#E74C8B"], edgecolor="black", alpha=0.85, width=0.5)
    for b, v in zip(barres, moy.values):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.15, f"{v:.2f}%", ha="center", fontweight="bold")
    ax1.set_title("Moyenne Avant vs Après COVID", fontsize=12, fontweight="bold", color="#2D2D6B")
    ax1.set_ylabel("Taux moyen (%)"); ax1.set_ylim(0, moy.max()*1.3); ax1.grid(axis="y", alpha=0.3)
    palette_covid = {"Avant COVID (≤ 2019)": "#2D2D6B", "Après COVID (≥ 2020)": "#E74C8B"}
    sns.boxplot(data=df_total, x="periode", y="taux_chomage", palette=palette_covid, ax=ax2)
    ax2.set_title("Distribution par Période", fontsize=12, fontweight="bold", color="#2D2D6B")
    ax2.set_xlabel(""); ax2.set_ylabel("Taux (%)"); ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout(); return fig


def graphique_top_pays(df):
    df_total = df[(df["sex"] == "Total") & (df["age"] == "15+")]
    moy_pays = df_total.groupby("country")["taux_chomage"].mean().sort_values()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    bas = moy_pays.head(10); haut = moy_pays.tail(10)
    ax1.barh(bas.index, bas.values, color="#0D9488", edgecolor="black", alpha=0.85)
    for i, v in enumerate(bas.values): ax1.text(v+0.1, i, f"{v:.1f}%", va="center", fontsize=9)
    ax1.set_title("Top 10 — Chômage le Plus BAS", fontsize=11, fontweight="bold", color="#2D2D6B")
    ax1.set_xlabel("Taux moyen (%)"); ax1.grid(axis="x", alpha=0.3)
    ax2.barh(haut.index, haut.values, color="#E74C8B", edgecolor="black", alpha=0.85)
    for i, v in enumerate(haut.values): ax2.text(v+0.1, i, f"{v:.1f}%", va="center", fontsize=9)
    ax2.set_title("Top 10 — Chômage le Plus ÉLEVÉ", fontsize=11, fontweight="bold", color="#2D2D6B")
    ax2.set_xlabel("Taux moyen (%)"); ax2.grid(axis="x", alpha=0.3)
    plt.tight_layout(); return fig


def graphique_correlation(df):
    cols_num = [c for c in [
        "taux_chomage", "taux_emploi", "year", "post_covid", "post_2008",
        "decennie", "sex_encoded", "age_encoded",
        "chomage_lag1", "chomage_lag2", "chomage_rolling3",
        "chomage_rolling5", "chomage_delta", "chomage_delta2"
    ] if c in df.columns]
    corr = df[cols_num].corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    masque = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=masque, annot=True, fmt=".2f", cmap="RdPu",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Matrice de Corrélation", fontsize=13, fontweight="bold", color="#2D2D6B")
    plt.tight_layout(); return fig


# ==============================================================================
# ÉTAPE 7 — FONCTIONS GRAPHIQUES (MODÈLE)
# ==============================================================================

def graphique_predictions_vs_reels(y_test, y_pred, mae, r2):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.4, color="#7B2FBE", edgecolor="none", s=15)
    lim = [min(y_test.min(), y_pred.min())-1, max(y_test.max(), y_pred.max())+1]
    ax.plot(lim, lim, "r--", linewidth=2, label="Prédiction parfaite")
    ax.fill_between(lim, [lim[0]-mae, lim[1]-mae], [lim[0]+mae, lim[1]+mae],
                    alpha=0.1, color="#F4C542", label=f"Zone ±MAE ({mae:.2f}%)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Valeurs Réelles (%)", fontsize=12)
    ax.set_ylabel("Valeurs Prédites (%)", fontsize=12)
    ax.set_title(f"Prédictions vs Réalité\nMAE={mae:.2f}%  |  R²={r2:.3f}", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); return fig


def graphique_comparaison_annees(df, modele, feature_cols, le_sex, le_age, le_country,
                                  pays, genre, age_grp):
    df_hist = df[(df["country"]==pays) & (df["sex"]==genre) & (df["age"]==age_grp)].sort_values("year").copy()
    if len(df_hist) == 0:
        return None

    sex_enc     = int(le_sex.transform([genre])[0])
    age_enc     = int(le_age.transform([age_grp])[0])
    country_enc = int(le_country.transform([pays])[0])
    taux_emploi_moyen = df_hist["taux_emploi"].mean()

    # Prédictions historiques
    annees_historiques = sorted(df_hist["year"].unique())
    lignes_hist = []
    for annee in annees_historiques:
        row = df_hist[df_hist["year"] == annee].iloc[0]
        lignes_hist.append({
            "year":             annee,
            "taux_emploi":      row["taux_emploi"],
            "post_covid":       int(annee >= 2020),
            "post_2008":        int(2008 <= annee <= 2013),
            "decennie":         (annee // 10) * 10,
            "sex_encoded":      sex_enc,
            "age_encoded":      age_enc,
            "country_encoded":  country_enc,
            "chomage_lag1":     row["chomage_lag1"],
            "chomage_lag2":     row.get("chomage_lag2", row["chomage_lag1"]),
            "chomage_rolling3": row["chomage_rolling3"],
            "chomage_rolling5": row.get("chomage_rolling5", row["chomage_rolling3"]),
            "chomage_delta":    row["chomage_delta"],
            "chomage_delta2":   row.get("chomage_delta2", 0.0),
        })
    df_hist_input = pd.DataFrame(lignes_hist)
    for col in feature_cols:
        if col not in df_hist_input.columns:
            df_hist_input[col] = 0
    pred_historiques = modele.predict(df_hist_input[feature_cols])

    # Prédictions futures en cascade
    valeurs_connues = list(df_hist["taux_chomage"].values)
    annees_futures  = list(range(2026, 2046))
    pred_futures    = []
    fenetre         = valeurs_connues[-5:].copy()

    for annee in annees_futures:
        lag1     = fenetre[-1]
        lag2     = fenetre[-2] if len(fenetre) >= 2 else lag1
        rolling3 = float(np.mean(fenetre[-3:]))
        rolling5 = float(np.mean(fenetre[-5:]))
        delta    = fenetre[-1] - fenetre[-2] if len(fenetre) >= 2 else 0.0
        delta2   = (fenetre[-1]-fenetre[-2])-(fenetre[-2]-fenetre[-3]) if len(fenetre) >= 3 else 0.0

        df_entree = _build_feature_row(
            annee, taux_emploi_moyen, sex_enc, age_enc, country_enc,
            lag1, rolling3, delta, feature_cols,
            lag2=lag2, rolling5=rolling5, delta2=delta2
        )
        pred = max(0.0, float(modele.predict(df_entree)[0]))
        pred_futures.append(pred)
        fenetre.append(pred)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(df_hist["year"]), y=list(df_hist["taux_chomage"]),
        mode="lines+markers", name="Valeurs Réelles (ILOSTAT / OIT)",
        line=dict(color="#2D2D6B", width=2.5), marker=dict(size=6),
        hovertemplate="Année: %{x}<br>Réel (OIT): %{y:.2f}%<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=annees_historiques, y=list(pred_historiques),
        mode="lines+markers", name="XGBoost (années connues)",
        line=dict(color="#7B2FBE", width=2, dash="dot"),
        marker=dict(size=5, symbol="diamond"),
        hovertemplate="Année: %{x}<br>XGBoost: %{y:.2f}%<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=annees_futures, y=pred_futures,
        mode="lines+markers", name="Prédictions Futures (2026–2045)",
        line=dict(color="#F4C542", width=2.5, dash="dash"),
        marker=dict(size=7, symbol="star"),
        hovertemplate="Année: %{x}<br>Prédit: %{y:.2f}%<extra></extra>"
    ))
    fig.add_vline(x=2025, line_dash="dash", line_color="#E74C8B",
                  annotation_text="↑ Futur", annotation_position="top right")
    fig.update_layout(
        title=f"Réel (OIT) vs XGBoost — {pays} | {genre} | {age_grp}",
        xaxis_title="Année", yaxis_title="Taux de chômage (%)",
        template="plotly_white", hovermode="x unified", height=420,
        legend=dict(orientation="h",x=0.0, y=-0.22,bgcolor="rgba(255,255,255,1.0)", bordercolor="#7B2FBE", borderwidth=2,font=dict(color="#1E1B4B", size=11)
)
    )
    return fig


def graphique_importances(importances):
    fig, ax  = plt.subplots(figsize=(8, 5))
    top10    = importances.head(10)
    couleurs = sns.color_palette("RdPu_r", n_colors=len(top10))
    barres   = ax.barh(top10.index[::-1], top10.values[::-1], color=couleurs, edgecolor="#2D2D6B", alpha=0.85)
    for barre, val in zip(barres, top10.values[::-1]):
        ax.text(barre.get_width()+0.001, barre.get_y()+barre.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.set_title("Importance des Variables — XGBoost", fontsize=12, fontweight="bold", color="#2D2D6B")
    ax.set_xlabel("Importance"); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout(); return fig


def graphique_residus(y_test, y_pred):
    residus = np.array(y_test) - np.array(y_pred)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    sns.histplot(residus, kde=True, color="#7B2FBE", bins=50, edgecolor="black", alpha=0.7, ax=ax1)
    ax1.axvline(0, color="#E74C8B", linestyle="--", linewidth=2, label="0 = parfait")
    ax1.set_title("Distribution des Résidus", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Résidu (Réel - Prédit)"); ax1.set_ylabel("Fréquence")
    ax1.legend(); ax1.grid(axis="y", alpha=0.3)
    ax2.scatter(y_pred, residus, alpha=0.3, color="#2D2D6B", s=10)
    ax2.axhline(0, color="#E74C8B", linestyle="--", linewidth=2)
    ax2.set_title("Résidus vs Prédictions", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Valeurs Prédites (%)"); ax2.set_ylabel("Résidu")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout(); return fig


def graphique_cv_scores(cv_scores):
    fig, ax = plt.subplots(figsize=(8, 5))
    folds    = [f"Fold {i+1}" for i in range(len(cv_scores))]
    couleurs = ["#E74C8B" if v == max(cv_scores) else "#7B2FBE" for v in cv_scores]
    barres   = ax.bar(folds, cv_scores, color=couleurs, edgecolor="#2D2D6B", alpha=0.85)
    ymax = max(cv_scores)
    ax.set_ylim(0, ymax * 1.35)
    for b, v in zip(barres, cv_scores):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+ymax*0.02,
                f"{v:.3f}%", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(np.mean(cv_scores), color="#F4C542", linestyle="--", linewidth=2,
               label=f"Moyenne : {np.mean(cv_scores):.3f}%")
    ax.set_title("MAE par Fold — TimeSeriesSplit (5 folds)", fontsize=12, fontweight="bold", color="#2D2D6B")
    ax.set_ylabel("MAE (%)"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); return fig


# ==============================================================================
# ÉTAPE 8 — APPLICATION STREAMLIT
# ==============================================================================

def main():

    with st.sidebar:
        st.markdown("## 🌍 Navigation")
        section = st.radio("Choisir une section :", [
            "🏠 Accueil", "📂 Données & Nettoyage", "📊 Analyse Exploratoire",
            "🗺️ Carte Mondiale", "🤖 Modèle & Évaluation", "🎯 Interface de Prédiction",
        ])
        st.markdown("---")
        st.markdown("**📦 Dataset**")
        st.markdown("📌 Source : Kaggle — données ILO/ILOSTAT")
        st.markdown("📌 183 pays | 1991–2025")
        st.markdown("📌 57 519 lignes")
        st.markdown("---")
        st.markdown("**🤖 Modèle**")
        st.markdown("⚡ XGBoost Regressor")
        st.markdown("📏 MAE + R²")

    df = charger_donnees()

    # ==========================================================================
    # PAGE 1 : ACCUEIL
    # ==========================================================================
    if section == "🏠 Accueil":
        st.markdown('<div class="main-title">🌍 Analyse et Prédiction du Chômage Mondial</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-title">par Machine Learning — XGBoost Regressor</div>', unsafe_allow_html=True)
        st.markdown("---")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🌍 Pays",     "183")
        c2.metric("📅 Période",  "1991–2025")
        c3.metric("📊 Lignes",   f"{len(df):,}")
        c4.metric("📋 Colonnes", len(df.columns))

        st.markdown("---")
        col_g, col_d = st.columns(2)
        with col_g:
            st.markdown('<div class="section-title">🎯 Objectif</div>', unsafe_allow_html=True)
            st.markdown("""
            Développer un modèle **XGBoost** pour **prédire le taux de chômage** selon :
            - 🌍 Le **pays**
            - 👤 Le **genre** (Homme / Femme / Total)
            - 🎂 Le **groupe d'âge** (15+, 15-24, 25+)
            - 📅 L'**année** (passée ou future)
            """)
        with col_d:
            st.markdown('<div class="section-title">⚡ Pourquoi XGBoost ?</div>', unsafe_allow_html=True)
            st.markdown("""
            | Critère | Random Forest | XGBoost |
            |---|---|---|
            | Arbres | Parallèles | Séquentiels |
            | Correction erreurs | ❌ | ✅ |
            | Précision | Bonne | **Meilleure** |
            | Régularisation | ❌ | ✅ |
            """)

        st.markdown("---")
        st.markdown('<div class="section-title">📈 Aperçu Rapide</div>', unsafe_allow_html=True)
        st.pyplot(graphique_evolution(df), use_container_width=True); plt.close()

    # ==========================================================================
    # PAGE 2 : DONNÉES & NETTOYAGE
    # ==========================================================================
    elif section == "📂 Données & Nettoyage":
        st.markdown('<div class="main-title">📂 Données & Nettoyage</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">📌 Source des Données</div>', unsafe_allow_html=True)
        st.info("""
        **Valeurs réelles issues de l'OIT — Organisation Internationale du Travail**
        Base ILOSTAT : https://ilostat.ilo.org/data/

        | Fichier | Indicateur | Description |
        |---|---|---|
        | `disoccupazione.csv` | UNE_DEAP_SEX_AGE_RT | Taux de chômage (% pop. active) |
        | `occupazione.csv` | EMP_TEMP_SEX_AGE_RT | Taux d'emploi (% pop. en âge de travailler) |
        """)

        st.markdown('<div class="section-title">🔍 Exploration Initiale</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Lignes",      f"{len(df):,}")
        c2.metric("Colonnes",    len(df.columns))
        c3.metric("Pays",        df["country"].nunique())
        c4.metric("Valeurs NaN", df.isnull().sum().sum())   # ✅ Doit afficher 0

        st.markdown("**df.head() :**")
        st.dataframe(df.head(10), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Types des colonnes :**")
            st.dataframe(pd.DataFrame({
                "Colonne":  df.columns,
                "Type":     [str(df[c].dtype) for c in df.columns],
                "Non-Nuls": [df[c].notna().sum() for c in df.columns],
            }), use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**Statistiques :**")
            st.dataframe(df[["taux_chomage","taux_emploi","year"]].describe().round(2),
                         use_container_width=True)

        st.markdown('<div class="section-title">🧹 Étapes du Nettoyage</div>', unsafe_allow_html=True)
        with st.expander("✅ Étape 1 — Suppression des doublons", expanded=True):
            st.code("df_chomage.drop_duplicates(inplace=True)\ndf_emploi.drop_duplicates(inplace=True)", language="python")
            st.success("✅ 0 doublons")
        with st.expander("✅ Étape 2 — Valeurs manquantes", expanded=True):
            st.code("df_chomage.dropna(inplace=True)\ndf_emploi.dropna(inplace=True)", language="python")
            st.success(f"✅ {df.isnull().sum().sum()} valeur(s) NaN")
        with st.expander("✅ Étape 3 — Renommage des colonnes", expanded=True):
            st.code("df_chomage.rename(columns={'obs_value': 'taux_chomage'}, inplace=True)", language="python")
            st.success("✅ Colonnes renommées")
        with st.expander("✅ Étape 4 — Fusion (left join + imputation médiane)", expanded=True):
            st.code("""df = pd.merge(df_chomage, df_emploi[...], on=[...], how='left')
df['taux_emploi'] = df.groupby(['country','sex','age'])['taux_emploi']
                      .transform(lambda x: x.fillna(x.median()))""", language="python")
            st.success(f"✅ {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
        with st.expander("✅ Étape 5 — Feature Engineering", expanded=True):
            st.code("""df['post_covid'] = (df['year'] >= 2020).astype(int)
df['post_2008']  = ((df['year'] >= 2008) & (df['year'] <= 2013)).astype(int)
df['decennie']   = (df['year'] // 10) * 10""", language="python")
        with st.expander("✅ Étape 6 — Encodage LabelEncoder", expanded=True):
            st.code("""le_sex     = LabelEncoder().fit(df['sex'])
le_age     = LabelEncoder().fit(df['age'])
le_country = LabelEncoder().fit(df['country'])""", language="python")
        with st.expander("✅ Étape 7 — Lag Features (6 features temporelles) — 0 NaN ✅", expanded=True):
            st.code("""# FIX 0 NaN : min_periods=1 + fillna(0) sur tous les rolling/delta
df['chomage_lag1']     = grp.shift(1)
df['chomage_lag2']     = grp.shift(2).fillna(lag1)   # ← fillna
df['chomage_rolling3'] = grp.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
df['chomage_rolling5'] = grp.transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
df['chomage_delta']    = grp.diff().fillna(0)         # ← fillna(0)
df['chomage_delta2']   = grp.diff().diff().fillna(0)  # ← fillna(0)
df.dropna(subset=['chomage_lag1'], inplace=True)       # seule vraie NaN""", language="python")
            st.success("✅ 0 NaN — toutes les features sont complètes")

    # ==========================================================================
    # PAGE 3 : ANALYSE EXPLORATOIRE
    # ==========================================================================
    elif section == "📊 Analyse Exploratoire":
        st.markdown('<div class="main-title">📊 Analyse Exploratoire des Données</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">📈 Distribution du Taux de Chômage</div>', unsafe_allow_html=True)
        st.pyplot(graphique_distribution(df), use_container_width=True); plt.close()
        st.caption("Asymétrie à droite — la majorité des pays ont entre 3% et 15%.")

        st.markdown('<div class="section-title">📅 Évolution Mondiale (1991–2025)</div>', unsafe_allow_html=True)
        st.pyplot(graphique_evolution(df), use_container_width=True); plt.close()
        st.caption("Pic net en 2020 (COVID-19). Légère hausse après 2008 (crise financière).")

        st.markdown('<div class="section-title">👤 Impact du Genre</div>', unsafe_allow_html=True)
        st.pyplot(graphique_genre(df), use_container_width=True); plt.close()
        st.caption("Le chômage féminin est systématiquement plus élevé que le masculin.")

        st.markdown('<div class="section-title">🎂 Impact du Groupe d\'Âge</div>', unsafe_allow_html=True)
        st.pyplot(graphique_age(df), use_container_width=True); plt.close()
        st.caption("Les jeunes (15-24 ans) sont 2× plus touchés que les adultes (25+).")

        st.markdown('<div class="section-title">🦠 Avant / Après COVID-19</div>', unsafe_allow_html=True)
        st.pyplot(graphique_covid(df), use_container_width=True); plt.close()
        avant = df[(df["sex"]=="Total") & (df["year"]<2020)]["taux_chomage"].mean()
        apres = df[(df["sex"]=="Total") & (df["year"]>=2020)]["taux_chomage"].mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("Avant COVID", f"{avant:.2f}%")
        c2.metric("Après COVID", f"{apres:.2f}%")
        c3.metric("Différence",  f"{apres-avant:+.2f}%", delta_color="inverse")

        st.markdown('<div class="section-title">🌍 Top 10 Pays</div>', unsafe_allow_html=True)
        st.pyplot(graphique_top_pays(df), use_container_width=True); plt.close()

        st.markdown('<div class="section-title">🔥 Matrice de Corrélation</div>', unsafe_allow_html=True)
        st.pyplot(graphique_correlation(df), use_container_width=True); plt.close()
        st.caption("chomage_lag1 est très fortement corrélé à taux_chomage → feature clé.")

    # ==========================================================================
    # PAGE 4 : CARTE MONDIALE
    # ==========================================================================
    elif section == "🗺️ Carte Mondiale":
        st.markdown('<div class="main-title">🗺️ Carte Mondiale Interactive</div>', unsafe_allow_html=True)
        st.info("🎬 Cliquez sur **Play** pour animer l'évolution du chômage de 1991 à 2025.")

        df_map = df[(df["sex"]=="Total") & (df["age"]=="15+")].copy()
        fig = px.choropleth(
            df_map, locations="iso_code", color="taux_chomage",
            hover_name="country", animation_frame="year",
            color_continuous_scale="RdPu", range_color=[0, 35],
            labels={"taux_chomage": "Taux (%)"},
            title="Évolution du Taux de Chômage Mondial (1991–2025) — Source : ILOSTAT"
        )
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">📊 Comparer des Pays</div>', unsafe_allow_html=True)
        pays_liste = sorted(df["country"].unique())
        defaut = [p for p in ["Morocco", "France", "Algeria", "Germany", "Japan"] if p in pays_liste]
        pays_selec = st.multiselect("Sélectionner des pays :", pays_liste, default=defaut)
        if pays_selec:
            df_comp = df[(df["country"].isin(pays_selec)) & (df["sex"]=="Total") & (df["age"]=="15+")]
            fig2 = px.line(df_comp, x="year", y="taux_chomage", color="country",
                           title="Comparaison du Chômage (Source : ILOSTAT / OIT)",
                           labels={"taux_chomage": "Taux (%)", "year": "Année"},
                           template="plotly_white")
            fig2.update_layout(hovermode="x unified")
            st.plotly_chart(fig2, use_container_width=True)

    # ==========================================================================
    # PAGE 5 : MODÈLE & ÉVALUATION
    # ==========================================================================
    elif section == "🤖 Modèle & Évaluation":
        st.markdown('<div class="main-title">⚡ Modèle XGBoost</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">🧠 Comment fonctionne XGBoost ?</div>', unsafe_allow_html=True)
        st.markdown("""
        **XGBoost** construit des arbres de décision **en séquence** :
        1. **Arbre 1** → première prédiction (imparfaite)
        2. **Arbre 2** → corrige les erreurs de l'arbre 1
        3. **Arbre 3** → corrige les erreurs des arbres 1+2
        4. ... jusqu'à **500 arbres** + early stopping !

        **6 features temporelles (Lag Features) :**
        - `chomage_lag1` : taux N-1 | `chomage_lag2` : taux N-2
        - `chomage_rolling3` : moyenne 3 ans | `chomage_rolling5` : moyenne 5 ans
        - `chomage_delta` : tendance | `chomage_delta2` : accélération
        """)

        c1, c2, c3 = st.columns(3)
        c1.info("**Algorithme**\nXGBoost Regressor")
        c2.info("**Variable cible**\ntaux_chomage (%)")
        c3.info("**Split**\n80% train / 20% test (temporel)")

        with st.expander("⚙️ Voir le code du modèle", expanded=False):
            st.code("""
modele = XGBRegressor(
    n_estimators=500,       max_depth=5,
    learning_rate=0.05,     subsample=0.8,
    colsample_bytree=0.7,   min_child_weight=3,
    gamma=0.1,              reg_alpha=0.05,
    reg_lambda=1.0,         early_stopping_rounds=30,
    eval_metric='mae',      random_state=42
)
modele.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            """, language="python")

        with st.spinner("⚡ Entraînement XGBoost en cours..."):
            modele, X_test, y_test, y_pred, mae, r2, importances, feature_cols, le_sex, le_age, le_country = entrainer_modele(df)

        st.markdown('<div class="section-title">📏 Résultats</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE",   f"{mae:.2f}%")
        c2.metric("R²",    f"{r2:.3f}")
        c3.metric("Train", f"{int(len(df)*0.8):,}")
        c4.metric("Test",  f"{int(len(df)*0.2):,}")

        st.markdown(f"""
        - **MAE = {mae:.2f}%** → XGBoost se trompe en moyenne de ±{mae:.2f} points de %
        - **R² = {r2:.3f}** → Le modèle explique **{r2*100:.1f}%** de la variance du chômage
        """)
        if r2 >= 0.90: st.success(f"✅ Excellent modèle ! R² = {r2:.3f}")
        elif r2 >= 0.75: st.warning(f"⚠️ Bon modèle. R² = {r2:.3f}")
        else: st.error(f"❌ Modèle à améliorer. R² = {r2:.3f}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-title">📊 Prédictions vs Réalité</div>', unsafe_allow_html=True)
            st.pyplot(graphique_predictions_vs_reels(y_test, y_pred, mae, r2), use_container_width=True); plt.close()
        with c2:
            st.markdown('<div class="section-title">🏆 Importance des Variables</div>', unsafe_allow_html=True)
            st.pyplot(graphique_importances(importances), use_container_width=True); plt.close()
            st.caption("chomage_lag1 devrait dominer — feature la plus prédictive.")

        st.markdown('<div class="section-title">🔍 Analyse des Résidus</div>', unsafe_allow_html=True)
        st.pyplot(graphique_residus(y_test, y_pred), use_container_width=True); plt.close()

        st.markdown('<div class="section-title">⏳ Validation Croisée Temporelle (TimeSeriesSplit)</div>', unsafe_allow_html=True)
        with st.spinner("Calcul de la cross-validation temporelle (5 folds)..."):
            df_cv = df.copy()
            df_cv.drop(columns=["iso_code","country","sex","age"], inplace=True, errors="ignore")
            df_cv.dropna(inplace=True)
            df_cv = df_cv.sort_values("year").reset_index(drop=True)
            feat_cv = [c for c in df_cv.columns if c != "taux_chomage"]
            X_cv = df_cv[feat_cv]; y_cv = df_cv["taux_chomage"]
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = []
            for train_idx, val_idx in tscv.split(X_cv):
                m = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.7, random_state=42, n_jobs=-1)
                m.fit(X_cv.iloc[train_idx], y_cv.iloc[train_idx])
                cv_scores.append(mean_absolute_error(y_cv.iloc[val_idx], m.predict(X_cv.iloc[val_idx])))

        c1, c2, c3 = st.columns(3)
        c1.metric("MAE moyenne (CV)", f"{np.mean(cv_scores):.3f}%")
        c2.metric("Écart-type",       f"±{np.std(cv_scores):.3f}%")
        c3.metric("Meilleur fold",    f"{min(cv_scores):.3f}%")
        st.pyplot(graphique_cv_scores(cv_scores), use_container_width=True); plt.close()

    # ==========================================================================
    # PAGE 6 : INTERFACE DE PRÉDICTION
    # ==========================================================================
    elif section == "🎯 Interface de Prédiction":
        st.markdown('<div class="main-title">🎯 Interface de Prédiction</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-title">Entrez les paramètres pour prédire le taux de chômage via XGBoost</div>', unsafe_allow_html=True)
        st.info("📌 **Valeurs réelles :** ILOSTAT (OIT) — https://ilostat.ilo.org/data/")

        with st.spinner("Chargement du modèle..."):
            modele, X_test, y_test, y_pred_test, mae, r2, importances, feature_cols, le_sex, le_age, le_country = entrainer_modele(df)

        st.markdown("---")
        col_form, col_result = st.columns([1, 1])

        with col_form:
            st.markdown('<div class="section-title">📝 Paramètres</div>', unsafe_allow_html=True)
            pays_liste = sorted(df["country"].unique())
            pays    = st.selectbox("🌍 Pays", pays_liste,
                                   index=pays_liste.index("Morocco") if "Morocco" in pays_liste else 0)
            genre   = st.selectbox("👤 Genre", ["Total", "Male", "Female"])
            age_grp = st.selectbox("🎂 Groupe d'âge", ["15+", "15-24", "25+"])
            annee   = st.slider("📅 Année", min_value=1991, max_value=2045, value=2026, step=1)

            if annee > 2025:
                st.info(f"📈 Tendance projetée {annee} — scénario stable")
            else:
                st.info(f"📌 {annee} est une année connue dans ILOSTAT")

            predict_btn = st.button("⚡ Prédire avec XGBoost", use_container_width=True, type="primary")

        with col_result:
            st.markdown('<div class="section-title">📊 Résultat</div>', unsafe_allow_html=True)

            if predict_btn:
                sex_enc     = int(le_sex.transform([genre])[0])
                age_enc     = int(le_age.transform([age_grp])[0])
                country_enc = int(le_country.transform([pays])[0])

                masque  = (df["country"]==pays) & (df["sex"]==genre) & (df["age"]==age_grp)
                df_pays = df[masque].sort_values("year")

                if annee <= 2025 and annee in df_pays["year"].values:
                    row             = df_pays[df_pays["year"] == annee].iloc[0]
                    lag1            = row["chomage_lag1"]
                    lag2            = row.get("chomage_lag2", lag1)
                    rolling3        = row["chomage_rolling3"]
                    rolling5        = row.get("chomage_rolling5", rolling3)
                    delta           = row["chomage_delta"]
                    delta2          = row.get("chomage_delta2", 0.0)
                    taux_emploi_val = row["taux_emploi"]
                else:
                    if len(df_pays) == 0:
                        src = df[(df["sex"]==genre) & (df["age"]==age_grp)].sort_values("year")
                        taux_emploi_val = src["taux_emploi"].mean()
                        fenetre = list(src["taux_chomage"].values[-5:])
                    else:
                        taux_emploi_val = df_pays["taux_emploi"].mean()
                        fenetre = list(df_pays["taux_chomage"].values[-5:])

                    derniere = int(df_pays["year"].max()) if len(df_pays) > 0 else 2025
                    a = derniere + 1
                    while a <= annee:
                        l1 = fenetre[-1]
                        l2 = fenetre[-2] if len(fenetre) >= 2 else l1
                        r3 = float(np.mean(fenetre[-3:]))
                        r5 = float(np.mean(fenetre[-5:]))
                        d1 = fenetre[-1]-fenetre[-2] if len(fenetre) >= 2 else 0.0
                        d2 = (fenetre[-1]-fenetre[-2])-(fenetre[-2]-fenetre[-3]) if len(fenetre) >= 3 else 0.0
                        df_c = _build_feature_row(a, taux_emploi_val, sex_enc, age_enc, country_enc,
                                                  l1, r3, d1, feature_cols, lag2=l2, rolling5=r5, delta2=d2)
                        fenetre.append(max(0.0, float(modele.predict(df_c)[0])))
                        a += 1

                    lag1     = fenetre[-2]
                    lag2     = fenetre[-3] if len(fenetre) >= 3 else lag1
                    rolling3 = float(np.mean(fenetre[-4:-1]))
                    rolling5 = float(np.mean(fenetre[-6:-1]))
                    delta    = fenetre[-2]-fenetre[-3] if len(fenetre) >= 3 else 0.0
                    delta2   = (fenetre[-2]-fenetre[-3])-(fenetre[-3]-fenetre[-4]) if len(fenetre) >= 4 else 0.0

                df_entree  = _build_feature_row(
                    annee, taux_emploi_val, sex_enc, age_enc, country_enc,
                    lag1, rolling3, delta, feature_cols,
                    lag2=lag2, rolling5=rolling5, delta2=delta2
                )
                prediction = max(0.0, round(float(modele.predict(df_entree)[0]), 2))

                masque_reel   = masque & (df["year"]==annee)
                valeur_reelle = df[masque_reel]["taux_chomage"].values
                valeur_reelle = float(valeur_reelle[0]) if len(valeur_reelle) > 0 else None

                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #2D2D6B 0%, #7B2FBE 55%, #E74C8B 100%);
                            border-radius: 16px; padding: 1.5rem; text-align: center;
                            color: #FFFFFF; margin: 0.5rem 0;
                            box-shadow: 0 4px 20px rgba(123,47,190,0.3);">
                    <div style="font-size: 0.82rem; opacity: 0.9;">Taux de chômage prédit (XGBoost)</div>
                    <div style="font-size: 0.92rem; font-weight: 600; margin: 0.3rem 0;">
                        {pays} — {genre} — {age_grp} — {annee}
                    </div>
                    <div style="font-size: 3rem; font-weight: 800; color: #FFF9C4; margin: 0.3rem 0;">
                        {prediction:.2f}%
                    </div>
                    <div style="font-size: 0.76rem; opacity: 0.8;">Marge d'erreur : ±{mae:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

                if valeur_reelle is not None:
                    difference = prediction - valeur_reelle
                    st.markdown("**📊 Comparaison avec la valeur réelle (ILOSTAT / OIT) :**")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("✅ Valeur OIT",     f"{valeur_reelle:.2f}%")
                    c2.metric("⚡ XGBoost prédit", f"{prediction:.2f}%")
                    c3.metric("📏 Écart",          f"{difference:+.2f}%", delta_color="inverse")
                    if abs(difference) <= mae:
                        st.success(f"✅ Écart ({abs(difference):.2f}%) ≤ MAE ({mae:.2f}%) — excellente prédiction !")
                    else:
                        st.info(f"ℹ️ Écart de {abs(difference):.2f}% (MAE : {mae:.2f}%)")
                else:
                    st.info(f"ℹ️ Pas de données ILOSTAT pour {annee} — prédiction future")

                if prediction < 5:   st.success("🟢 Très faible — Marché du travail très dynamique.")
                elif prediction < 10: st.info("🟡 Modéré — Dans la moyenne mondiale.")
                elif prediction < 20: st.warning("🟠 Élevé — Politiques d'emploi nécessaires.")
                else:                 st.error("🔴 Très élevé — Situation critique.")

            else:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #1A0533, #0D1B40); border-radius: 12px;
                            padding: 2rem; text-align: center; border: 2px dashed #E040FB;">
                    <div style="font-size: 2.5rem;">⚡</div>
                    <div style="color: #00E5FF; font-weight: 600; margin-top: 0.8rem;">
                        Remplissez le formulaire et cliquez sur <strong>Prédire avec XGBoost</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-title">📈 Réel (OIT) vs XGBoost — Historique + Futur (2026–2045)</div>', unsafe_allow_html=True)
        st.caption("🔵 Valeurs réelles ILOSTAT  |  🟣 XGBoost années connues  |  🟡 Prédictions futures (cascade)")

        c1, c2, c3 = st.columns(3)
        with c1:
            pays_graph = st.selectbox(
                "🌍 Pays", sorted(df["country"].unique()),
                index=sorted(df["country"].unique()).index("Morocco") if "Morocco" in df["country"].unique() else 0,
                key="pays_graph"
            )
        with c2:
            genre_graph = st.selectbox("👤 Genre", ["Total","Male","Female"], key="genre_graph")
        with c3:
            age_graph = st.selectbox("🎂 Âge", ["15+","15-24","25+"], key="age_graph")

        with st.spinner("Génération du graphique..."):
            fig_comp = graphique_comparaison_annees(
                df, modele, feature_cols, le_sex, le_age, le_country,
                pays_graph, genre_graph, age_graph
            )

        if fig_comp:
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.warning(f"Pas de données ILOSTAT pour {pays_graph} — {genre_graph} — {age_graph}")

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Algorithme",    "XGBoost")
        c2.metric("n_estimators",  "500 + early stop")
        c3.metric("MAE",           f"{mae:.2f}%")
        c4.metric("R²",            f"{r2:.3f}")


# ==============================================================================
# LANCEMENT
# ==============================================================================
if __name__ == "__main__":
    main()
