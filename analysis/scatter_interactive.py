# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Interactive (hover) version of the scatter plots, from
25_FEB_2026_vqe_energy_table.csv.

Produces ONE self-contained HTML file containing all 12 plots. Hovering any
dot shows: molecule, active space (Ne,No), qubits, %AS, and the x/y values --
so every active-space configuration is uniquely identifiable.

  figures/scatter/scatter_interactive.html
"""

import os
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

BASE    = os.path.dirname(os.path.abspath(__file__))
CSV     = os.path.join(BASE, "vqe_energy_table.csv")
OUT_DIR = os.path.join(BASE, "figures", "scatter")
OUT     = os.path.join(OUT_DIR, "scatter_interactive.html")
os.makedirs(OUT_DIR, exist_ok=True)

# ── column names ───────────────────────────────────────────────────
AS   = "Active Space (Ne,No)"
ORB  = "%AS wrt Orbitals (No/Total Orbitals)"
ELE  = "%AS wrt Electrons (Ne/Total Electrons)"
GHF  = "E_VQE_GPU - E_HF [Ha]"
CHF  = "E_VQE_CPU - E_HF [Ha]"
GAP  = "E_VQE_GPU - E_VQE_CPU [Ha]"
CORR = "Correlation Energy (E_VQE_GPU-E_HF)/E_VQE_GPU"
SPD  = "GPU Speedup (t_CPU/t_GPU)"
REC  = "CCSD Recovery (E_VQE_GPU-E_HF)/(E_CCSD-E_HF)"
RECC = "CASCI Recovery (E_VQE_GPU-E_HF)/(E_CASCI-E_HF)"

MOL_ORDER = [
    "Methylene", "Ethylene",
    "Benzene", "Naphthalene", "Benzaanthracene", "Pentacene",
    "NH2-", "Methanamide",
    "Adenine", "Thymine", "Uracil", "Cytosine", "Guanine",
]
DISPLAY = {"NH2-": "NH2-", "Benzaanthracene": "Benz[a]anthracene"}

# 13 distinct colours + symbols
COLORS = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
          "#d62728", "#e377c2", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
          "#e7ba52"]
SYMBOLS = ["circle", "square", "triangle-up", "triangle-down", "diamond",
           "cross", "x", "triangle-left", "triangle-right", "pentagon",
           "hexagon", "star", "hexagram"]

# (key, x-col, y-col, title, xlab, ylab, ylog, extras)
SPECS = [
    ("1", ORB,  GHF,  "VQE-HF energy vs orbital fraction",
     "Active space (% of orbitals)", "E_VQE_GPU - E_HF [Ha]", False, None),
    ("2", ELE,  GHF,  "VQE-HF energy vs electron fraction",
     "Active space (% of electrons)", "E_VQE_GPU - E_HF [Ha]", False, None),
    ("3", ORB,  CORR, "Correlation ratio vs orbital fraction",
     "Active space (% of orbitals)", "(E_VQE_GPU-E_HF)/E_VQE_GPU", False, None),
    ("4", ELE,  CORR, "Correlation ratio vs electron fraction",
     "Active space (% of electrons)", "(E_VQE_GPU-E_HF)/E_VQE_GPU", False, None),
    ("5", CHF,  GHF,  "CPU vs GPU parity",
     "E_VQE_CPU - E_HF [Ha]", "E_VQE_GPU - E_HF [Ha]", False, "parity"),
    ("6", ORB,  GAP,  "GPU-CPU gap vs orbital fraction (log y)",
     "Active space (% of orbitals)", "E_VQE_GPU - E_VQE_CPU [Ha]", True, None),
    ("7", ORB,  SPD,  "GPU speedup vs orbital fraction",
     "Active space (% of orbitals)", "GPU speedup t_CPU/t_GPU", False, ("hline", 1.0, "break-even")),
    ("8", "qubits", SPD, "GPU speedup vs qubits",
     "Qubits (= 2 No)", "GPU speedup t_CPU/t_GPU", False, ("hline", 1.0, "break-even")),
    ("9", SPD,  CORR, "Correlation ratio vs speedup",
     "GPU speedup t_CPU/t_GPU", "(E_VQE_GPU-E_HF)/E_VQE_GPU", False, None),
    ("10", ORB, REC,  "CCSD recovery vs orbital fraction",
     "Active space (% of orbitals)", "CCSD recovery", False, None),
    ("11", SPD, REC,  "CCSD recovery vs speedup",
     "GPU speedup t_CPU/t_GPU", "CCSD recovery", False, None),
    ("12", SPD, RECC, "CASCI recovery vs speedup",
     "GPU speedup t_CPU/t_GPU", "CASCI recovery", False, ("hline", 1.0, "VQE = CASCI")),
]


def load():
    df = pd.read_csv(CSV)
    for c in [ORB, ELE, GHF, CHF, GAP, CORR, SPD, REC, RECC]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    p = df[AS].str.extract(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
    df["Ne"] = pd.to_numeric(p[0], errors="coerce")
    df["No"] = pd.to_numeric(p[1], errors="coerce")
    df["qubits"] = 2 * df["No"]
    present = [m for m in MOL_ORDER if m in set(df["Molecule"])]
    present += sorted(set(df["Molecule"]) - set(present))
    return df, present


def build_fig(df, present, spec):
    key, xcol, ycol, title, xlab, ylab, ylog, extra = spec
    fig = go.Figure()

    for i, mol in enumerate(present):
        sub = df[df["Molecule"] == mol]
        # customdata columns shown in the hover box
        cd = sub[[AS, "qubits", ORB, ELE, SPD]].to_numpy()
        fig.add_trace(go.Scatter(
            x=sub[xcol], y=sub[ycol], mode="markers", name=DISPLAY.get(mol, mol),
            marker=dict(size=10, color=COLORS[i % len(COLORS)],
                        symbol=SYMBOLS[i % len(SYMBOLS)],
                        line=dict(width=0.6, color="black")),
            customdata=cd,
            hovertemplate=(
                f"<b>{DISPLAY.get(mol, mol)}</b><br>"
                "active space (Ne,No) = %{customdata[0]}<br>"
                "qubits = %{customdata[1]}<br>"
                "%AS orb = %{customdata[2]:.2f}%  |  %AS elec = %{customdata[3]:.2f}%<br>"
                "speedup = %{customdata[4]:.2f}x<br>"
                f"{xlab} = " "%{x}<br>"
                f"{ylab} = " "%{y}<extra></extra>"
            ),
        ))

    if extra == "parity":
        import numpy as np
        v = pd.concat([df[xcol], df[ycol]]).dropna()
        lo, hi = float(v.min()), float(v.max())
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                 name="y = x", line=dict(color="black", dash="dash"),
                                 hoverinfo="skip"))
    elif isinstance(extra, tuple) and extra[0] == "hline":
        _, yval, lbl = extra
        xv = df[xcol].dropna()
        fig.add_trace(go.Scatter(x=[float(xv.min()), float(xv.max())], y=[yval, yval],
                                 mode="lines", name=lbl,
                                 line=dict(color="gray", dash="dash"), hoverinfo="skip"))

    fig.update_layout(
        title=f"Plot {key}: {title}",
        xaxis_title=xlab, yaxis_title=ylab,
        template="plotly_white", height=560,
        legend=dict(title="Molecule", font=dict(size=10)),
        margin=dict(l=70, r=30, t=60, b=60),
    )
    if ylog:
        fig.update_yaxes(type="log")
    return fig


def main():
    df, present = load()
    print(f"  {len(df)} rows, {len(present)} molecules")

    parts = [
        "<html><head><meta charset='utf-8'>",
        "<title>VQE scatter plots - interactive</title></head><body>",
        "<h2 style='font-family:sans-serif'>VQE benchmark scatter plots "
        "(25 Feb 2026) - hover any point for its active-space configuration</h2>",
        "<p style='font-family:sans-serif;color:#555'>Each point is one "
        "(molecule, active-space) run. Hover shows molecule, (Ne,No), qubits, "
        "%AS and the plotted values. Click legend entries to toggle molecules; "
        "double-click to isolate one.</p>",
    ]
    for j, spec in enumerate(SPECS):
        fig = build_fig(df, present, spec)
        parts.append(pio.to_html(fig, full_html=False,
                                 include_plotlyjs="cdn" if j == 0 else False))
        print(f"    + plot {spec[0]}")
    parts.append("</body></html>")

    with open(OUT, "w") as fh:
        fh.write("\n".join(parts))
    print(f"\n  -> {OUT}")


if __name__ == "__main__":
    main()
