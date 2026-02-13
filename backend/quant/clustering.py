from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.cluster.hierarchy import linkage
import scipy.spatial.distance as ssd


def prepare_clustering_data(metrics_dict: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prépare les données en filtrant les indices et en normalisant.

    Retourne :
      - scaled_features: numpy array normalisé
      - df_metrics: DataFrame nettoyé
    """
    # 1. Conversion en DataFrame
    df_metrics = pd.DataFrame(metrics_dict).T

    # 2. Filtre indices (ex: '^GSPC')
    df_metrics = df_metrics[~df_metrics.index.str.startswith("^")]

    # 3. Sélection des colonnes numériques (Beta & Specific Risk)
    features = df_metrics[["beta", "specific_risk_pct"]].apply(pd.to_numeric)

    # 4. Normalisation
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return scaled_features, df_metrics


def apply_kmeans_risk(metrics_dict: dict, n_clusters: int = 4) -> pd.DataFrame:
    """
    Clustering simple sur (bêta, risque spécifique) pour cohérence visuelle.
    """
    df_metrics = pd.DataFrame(metrics_dict).T
    df_metrics = df_metrics[~df_metrics.index.str.startswith("^")]
    features = df_metrics[["beta", "specific_risk_pct"]].apply(pd.to_numeric)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_features)

    df_result = pd.DataFrame(index=features.index)
    df_result["cluster"] = [f"Groupe {c}" for c in clusters]
    df_result["beta"] = features["beta"]
    df_result["specific_risk_pct"] = features["specific_risk_pct"]
    return df_result


def find_optimal_k(scaled_data, k_max: int = 10):
    """
    Calcule l'inertie pour k=1..k_max (méthode du coude).
    Renvoie une liste d'inerties (pas d'affichage).
    """
    inertia = []
    for k in range(1, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)
    return inertia


def apply_kmeans_expert(
    corr_matrix: pd.DataFrame,
    df_metrics_cleaned: pd.DataFrame,
    n_clusters: int = 4,
) -> pd.DataFrame:
    """
    Fusionne une matrice de corrélation et le bêta pour un clustering avancé.
    """
    # 1. On récupère uniquement le Bêta
    df_beta = df_metrics_cleaned[["beta"]].apply(pd.to_numeric)

    # 2. Fusion corrélations + bêta (index alignés)
    df_features = pd.concat([corr_matrix, df_beta], axis=1).dropna()

    # 3. Normalisation
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_features)

    # 4. Entraînement du K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_features)

    # 5. Construction du résultat final
    df_result = pd.DataFrame(index=df_features.index)
    df_result["cluster"] = [f"Groupe {c}" for c in clusters]
    df_result["beta"] = df_features["beta"]
    df_result["specific_risk_pct"] = df_metrics_cleaned.loc[
        df_features.index, "specific_risk_pct"
    ]

    return df_result


def build_correlation_dendrogram(corr_matrix: pd.DataFrame, n_clusters: int = 4):
    """
    Construit les données nécessaires à un dendrogramme à partir d'une matrice
    de corrélation (sans affichage).

    Retourne :
      - linkage_matrix (Z)
      - threshold (seuil de coupe)
      - labels (tickers)
    """
    # 1. Corrélation -> distance
    dist_matrix = (2 * (1 - corr_matrix.clip(-1, 1))) ** 0.5

    # 2. Format condensé pour Scipy
    condensed_dist = ssd.squareform(dist_matrix, checks=False)

    # 3. Clustering hiérarchique
    Z = linkage(condensed_dist, method="ward")

    # 4. Seuil de coupe (pour n_clusters)
    if n_clusters >= 2 and Z.shape[0] >= n_clusters:
        threshold = (Z[-n_clusters + 1, 2] + Z[-n_clusters, 2]) / 2
    else:
        threshold = float(Z[-1, 2])

    return Z, threshold, list(corr_matrix.columns)


def apply_gmm_expert(
    corr_matrix: pd.DataFrame,
    metrics_dict: dict,
    n_components: int = 4,
) -> pd.DataFrame:
    """
    Applique le GMM sur un mix de Profil de Corrélation et de Bêta.
    """
    # 1. Extraction des Bêtas du dictionnaire de métriques
    df_beta = pd.DataFrame(metrics_dict).T[["beta"]]
    df_beta = df_beta[~df_beta.index.str.startswith("^")]

    # 2. Fusion : corrélation + bêta (index alignés)
    df_features = pd.concat([corr_matrix, df_beta], axis=1).dropna()

    # 3. Normalisation
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_features)

    # 4. Entraînement du GMM
    gmm = GaussianMixture(
        n_components=n_components,
        random_state=42,
        covariance_type="full",
        reg_covar=1e-3,
    )
    clusters = gmm.fit_predict(scaled_features)
    probs = gmm.predict_proba(scaled_features)

    # 5. Construction du résultat
    df_result = pd.DataFrame(index=df_features.index)
    df_result["cluster"] = [f"Groupe {c}" for c in clusters]

    # Certitude basée sur la distance au centroïde (variation réelle entre actifs)
    means = gmm.means_
    distances = np.linalg.norm(
        scaled_features - means[clusters], axis=1
    )
    d_min, d_max = distances.min(), distances.max()
    if np.isclose(d_max, d_min):
        certitude = np.full_like(distances, 50.0)
    else:
        certitude = 100 * (1 - (distances - d_min) / (d_max - d_min))
    df_result["certitude_ia_pct"] = np.round(certitude, 2)
    df_result["beta"] = df_beta.loc[df_features.index, "beta"]
    df_result["specific_risk_pct"] = [
        metrics_dict[t]["specific_risk_pct"] for t in df_features.index
    ]

    return df_result


def apply_gmm_risk(metrics_dict: dict, n_components: int = 4) -> pd.DataFrame:
    """
    GMM sur (bêta, risque spécifique) pour clusters cohérents avec le scatter.
    """
    df_metrics = pd.DataFrame(metrics_dict).T
    df_metrics = df_metrics[~df_metrics.index.str.startswith("^")]
    features = df_metrics[["beta", "specific_risk_pct"]].apply(pd.to_numeric)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    gmm = GaussianMixture(
        n_components=n_components,
        random_state=42,
        covariance_type="full",
        reg_covar=1e-3,
    )
    clusters = gmm.fit_predict(scaled_features)

    means = gmm.means_
    distances = np.linalg.norm(scaled_features - means[clusters], axis=1)
    d_min, d_max = distances.min(), distances.max()
    if np.isclose(d_max, d_min):
        certitude = np.full_like(distances, 50.0)
    else:
        certitude = 100 * (1 - (distances - d_min) / (d_max - d_min))

    df_result = pd.DataFrame(index=features.index)
    df_result["cluster"] = [f"Groupe {c}" for c in clusters]
    df_result["certitude_ia_pct"] = np.round(certitude, 2)
    df_result["beta"] = features["beta"]
    df_result["specific_risk_pct"] = features["specific_risk_pct"]
    return df_result


def get_gmm_analysis_table(df_result: pd.DataFrame) -> pd.DataFrame:
    """
    Renvoie un tableau des actions les plus ambiguës selon l'IA.
    """
    ambiguous_stocks = df_result.sort_values("certitude_ia_pct").head(10)
    return ambiguous_stocks[["cluster", "certitude_ia_pct", "beta", "specific_risk_pct"]]
