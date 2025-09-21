# pxy_sami/estimators/sami_core.py
from __future__ import annotations
import uuid
from typing import List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from django.conf import settings

# Headless backend for saving PNGs in containers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pxy_contracts.contracts import (
    SAMIRunRequest,
    SAMIRunResponse,
    SAMICity,
    SAMIPoint,  # ← for 56B interactive scatter
)
from pxy_contracts.version import SPEC_VERSION
from pxy_de.providers.base import get_provider


def _fit_loglog(df: pd.DataFrame) -> Tuple[float, float, float, np.ndarray]:
    import statsmodels.api as sm
    """
    Ajusta: log(value) = alpha + beta * log(N)  (OLS)
    Regresa: (alpha, beta, R^2, residuales)
    """
    df = df.copy()
    df["logY"] = np.log(df["value"].astype(float))
    df["logN"] = np.log(df["N"].astype(float))
    X = sm.add_constant(df["logN"].values)
    y = df["logY"].values
    model = sm.OLS(y, X).fit()
    alpha = float(model.params[0])
    beta = float(model.params[1])
    r2 = float(model.rsquared) if model.nobs and model.nobs >= 2 else 0.0
    resid = model.resid
    return alpha, beta, r2, resid


def _color_for_sami(s: float) -> str:
    """Colores sencillos: verde = arriba, rojo = abajo, gris = ~0."""
    if s > 0.15:
        return "#2ca02c"  # green
    if s < -0.15:
        return "#d62728"  # red
    return "#7f7f7f"       # gray


def _size_for_N(N: float, N_med: float) -> float:
    """Tamaño del punto ~ sqrt(N/mediana), acotado para demo."""
    if N <= 0 or N_med <= 0:
        return 60.0
    s = 80.0 * np.sqrt(N / N_med)
    return float(np.clip(s, 40.0, 300.0))


def _save_chart(
    df: pd.DataFrame, alpha: float, beta: float, r2: float, run_id: str, indicator: str
) -> str | None:
    """
    Crea un gráfico bonito para demo:
      - Izquierda: scatter log–log con línea de regresión, puntos coloreados por SAMI,
                   tamaño por N, etiquetas de ciudades y textbox con ecuación.
      - Derecha: ranking horizontal por SAMI (barh).
    Devuelve URL pública (/media/...).
    """
    try:
        media_dir = Path(settings.MEDIA_ROOT) / "sami"  # ensure Path
        media_dir.mkdir(parents=True, exist_ok=True)
        out_path = media_dir / f"sami_{run_id}.png"

        # Preparación de datos
        df = df.copy()
        df["logN"] = np.log(df["N"].astype(float))
        df["logY"] = np.log(df["value"].astype(float))
        x = df["logN"].values
        y = df["logY"].values

        # Línea de regresión
        xs = np.linspace(x.min(), x.max(), 100)
        ys = alpha + beta * xs

        # Tamaños por N
        N_med = float(df["N"].median())
        sizes = [_size_for_N(n, N_med) for n in df["N"].values]

        # Colores por SAMI
        colors = [_color_for_sami(s) for s in df["sami"].values]

        # Orden para ranking
        df_rank = df[["city", "sami"]].sort_values("sami", ascending=True).reset_index(drop=True)

        # Figure
        fig, axes = plt.subplots(
            1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.35, 1.0]}
        )
        ax, axr = axes

        # --- (L) Scatter log–log ---
        ax.scatter(
            x, y, s=sizes, c=colors, alpha=0.9, edgecolors="white", linewidths=0.8, zorder=3
        )
        ax.plot(xs, ys, linewidth=2.0, zorder=2)

        # Etiquetas por ciudad (offset según signo SAMI)
        for _, row in df.iterrows():
            dx = 0.02 * (x.max() - x.min() if x.max() > x.min() else 1.0)
            dy = 0.02 * (y.max() - y.min() if y.max() > y.min() else 1.0)
            offset_y = dy if row["sami"] >= 0 else -dy
            ax.annotate(
                row["city"],
                (row["logN"], row["logY"]),
                xytext=(row["logN"] + dx, row["logY"] + offset_y),
                fontsize=9,
                color="#303030",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#888888", alpha=0.8),
            )

        # Texto con ecuación y métricas
        eq_txt = (
            f"log(Value) = {alpha:.2f} + {beta:.3f}·log(N)\n"
            f"$R^2$ = {r2:.3f}   n = {len(df)}   indicador: {indicator}"
        )
        ax.text(
            0.02,
            0.98,
            eq_txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="#dddddd", alpha=0.9),
        )

        # Estética
        ax.set_xlabel("log(N)")
        ax.set_ylabel("log(Value)")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.set_title("Escalamiento urbano y SAMI", fontsize=12, pad=8)

        # --- (R) Ranking SAMI (barh) ---
        y_pos = np.arange(len(df_rank))
        bar_colors = [_color_for_sami(s) for s in df_rank["sami"].values]
        axr.barh(y_pos, df_rank["sami"].values, color=bar_colors, alpha=0.9)
        axr.set_yticks(y_pos, labels=df_rank["city"].values, fontsize=9)
        axr.set_xlabel("SAMI (z)")
        axr.axvline(0, color="#444444", linewidth=0.8)
        axr.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.6)
        for spine in ["top", "right"]:
            axr.spines[spine].set_visible(False)
        axr.set_title("Ranking por desviación (SAMI)", fontsize=12, pad=8)

        # Anotar top y bottom
        try:
            top_city = df_rank.iloc[-1]
            bottom_city = df_rank.iloc[0]
            axr.text(
                float(top_city["sami"]),
                float(len(df_rank) - 1),
                f"  ▲ {top_city['sami']:.2f}",
                va="center",
                ha="left",
                fontsize=9,
                color="#2ca02c",
                weight="bold",
            )
            axr.text(
                float(bottom_city["sami"]),
                0,
                f"  ▼ {bottom_city['sami']:.2f}",
                va="center",
                ha="left",
                fontsize=9,
                color="#d62728",
                weight="bold",
            )
        except Exception:
            pass

        fig.tight_layout()
        fig.savefig(out_path, dpi=144)
        plt.close(fig)

        return f"{settings.MEDIA_URL}sami/{out_path.name}"
    except Exception:
        return None


def run_sami(req: SAMIRunRequest) -> SAMIRunResponse:
    """
    SAMI v2 (demo ready):
      - Fit OLS log–log
      - SAMI = resid / std(resid)
      - Gráfico mejorado (scatter + ranking)
      - 56B: return alpha + raw points for interactive scatter
    """
    provider = get_provider()
    warnings: List[str] = []

    # 1) Cargar datos
    try:
        df = provider.indicator(req.indicator, req.cities or [])
    except Exception as e:
        warnings.append(f"data_provider_error: {e}")
        residuals = [SAMICity(city=c, sami=0.0, rank=i + 1) for i, c in enumerate(req.cities or [])]
        return SAMIRunResponse(
            model_id="sami-ols-v2.0.0",
            spec_version=SPEC_VERSION,
            run_id=str(uuid.uuid4()),
            indicator=req.indicator,
            beta=1.0,
            r2=0.0,
            residuals=residuals,
            chart_url=None,
            data_release=req.data_release,
            warnings=warnings or ["stub implementation"],
        )

    # 2) Limpieza mínima
    n_before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["value", "N"])
    df = df[(df["value"] > 0) & (df["N"] > 0)].copy()
    n_after = len(df)
    if n_before - n_after > 0:
        warnings.append(f"filtered_nonpositive_or_nan: {n_before - n_after}")
    if n_after < 2:
        warnings.append("not_enough_data_for_fit")
        residuals = [SAMICity(city=c, sami=0.0, rank=i + 1) for i, c in enumerate(req.cities or [])]
        return SAMIRunResponse(
            model_id="sami-ols-v2.0.0",
            spec_version=SPEC_VERSION,
            run_id=str(uuid.uuid4()),
            indicator=req.indicator,
            beta=1.0,
            r2=0.0,
            residuals=residuals,
            chart_url=None,
            data_release=req.data_release,
            warnings=warnings,
        )

    # 3) Ajuste y SAMI
    try:
        alpha, beta, r2, resid = _fit_loglog(df)
    except Exception as e:
        warnings.append(f"ols_fit_error: {e}")
        residuals = [SAMICity(city=c, sami=0.0, rank=i + 1) for i, c in enumerate(df["city"].tolist())]
        return SAMIRunResponse(
            model_id="sami-ols-v2.0.0",
            spec_version=SPEC_VERSION,
            run_id=str(uuid.uuid4()),
            indicator=req.indicator,
            beta=1.0,
            r2=0.0,
            residuals=residuals,
            chart_url=None,
            data_release=req.data_release,
            warnings=warnings,
        )

    std = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
    sami_vals = (resid / std) if std > 0 else np.zeros_like(resid)

    # 56B: build raw points (with logs) for interactive scatter
    df_pts = df.copy()
    df_pts["log_value"] = np.log(df_pts["value"].astype(float))
    df_pts["log_N"] = np.log(df_pts["N"].astype(float))
    points: List[SAMIPoint] = []
    for row in df_pts.itertuples(index=False):
        try:
            points.append(
                SAMIPoint(
                    city=str(row.city),
                    value=float(row.value),
                    N=float(row.N),
                    log_value=float(row.log_value),
                    log_N=float(row.log_N),
                )
            )
        except Exception:
            # If any row is malformed, skip it; interactive chart is best-effort.
            continue

    out = df[["city", "value", "N"]].copy()
    out["sami"] = sami_vals
    out = out.sort_values("sami", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)

    residuals = [
        SAMICity(city=row.city, sami=float(row.sami), rank=int(row.rank))
        for row in out.itertuples(index=False)
    ]

    # 4) Guardar gráfico bonito
    run_id = str(uuid.uuid4())
    chart_url = _save_chart(out, alpha, beta, r2, run_id, req.indicator)
    if chart_url is None:
        warnings.append("chart_save_failed")
    else:
        warnings.append("chart_saved")

    warnings.append(f"fit_ok_n={n_after}")

    return SAMIRunResponse(
        model_id="sami-ols-v2.0.0",
        spec_version=SPEC_VERSION,
        run_id=run_id,
        indicator=req.indicator,
        beta=float(beta),
        r2=float(r2),
        residuals=residuals,
        chart_url=chart_url,
        data_release=req.data_release,
        warnings=warnings,
        # 56B extras
        alpha=float(alpha),
        points=points,
    )
