# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Scatter plots from 25_FEB_2026_vqe_energy_table.csv

Generates 6 scatter plots, each point coloured/marked per molecule:
  1  %AS wrt Orbitals   vs  E_VQE_GPU - E_HF      (correlation vs orbital frac)
  2  %AS wrt Electrons  vs  E_VQE_GPU - E_HF      (correlation vs electron frac)
  3  %AS wrt Orbitals   vs  Correlation ratio
  4  %AS wrt Electrons  vs  Correlation ratio
  5  E_VQE_CPU - E_HF   vs  E_VQE_GPU - E_HF       (CPU/GPU parity, y=x line)
  6  %AS wrt Orbitals   vs  E_VQE_GPU - E_VQE_CPU  (GPU-CPU gap, log y)

Outputs:
  figures/scatter/<key>.png          one PNG per plot
  figures/scatter/scatter_panel.png  combined 2x3 panel
  tex_out/<key>.tex / _embed.tex     pgfplots (standalone + embeddable)
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE     = os.path.dirname(os.path.abspath(__file__))
CSV      = os.path.join(BASE, "vqe_energy_table.csv")
PNG_DIR  = os.path.join(BASE, "figures", "scatter")
TEX_DIR  = os.path.join(BASE, "tex_out")
os.makedirs(PNG_DIR, exist_ok=True)
os.makedirs(TEX_DIR, exist_ok=True)

# ── CSV column names ───────────────────────────────────────────────
ORB  = "%AS wrt Orbitals (No/Total Orbitals)"
ELE  = "%AS wrt Electrons (Ne/Total Electrons)"
GHF  = "E_VQE_GPU - E_HF [Ha]"
CHF  = "E_VQE_CPU - E_HF [Ha]"
GAP  = "E_VQE_GPU - E_VQE_CPU [Ha]"
CORR = "Correlation Energy (E_VQE_GPU-E_HF)/E_VQE_GPU"
SPD  = "GPU Speedup (t_CPU/t_GPU)"
REC  = "CCSD Recovery (E_VQE_GPU-E_HF)/(E_CCSD-E_HF)"
RECC = "CASCI Recovery (E_VQE_GPU-E_HF)/(E_CASCI-E_HF)"

# ── molecule order / display / style ───────────────────────────────
MOL_ORDER = [
    "Methylene", "Ethylene",
    "Benzene", "Naphthalene", "Benzaanthracene", "Pentacene",
    "NH2-", "Methanamide",
    "Adenine", "Thymine", "Uracil", "Cytosine", "Guanine",
]
DISPLAY_MPL = {
    "NH2-": r"NH$_2^-$", "Benzaanthracene": "Benz[a]anthracene",
}
DISPLAY_TEX = {
    "NH2-": r"NH$_2^-$", "Benzaanthracene": r"Benz[a]anthracene",
}
# distinct colours (matplotlib) + marker shapes, 13 molecules
MPL_COLORS = plt.get_cmap("tab20").colors
MPL_MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">", "h", "p", "d"]

# pgfplots: 13 colours + marks
TIKZ_COLORS = [
    "green!50!black", "violet", "orange", "teal",
    "blue!70!black", "red!70!black", "magenta", "gray!60!black",
    "brown", "cyan!60!black", "purple!60!black", "red!50!yellow", "black",
]
TIKZ_MARKS = [
    # open oplus/otimes (not filled *): the filled variants hide the inner
    # +/x and look like plain filled circles (collides with '*').
    "*", "square*", "triangle*", "diamond*", "pentagon*", "o", "square",
    "triangle", "diamond", "pentagon", "oplus", "otimes", "star",
]

# ── pgfplots standalone / embed wrappers (match latex generator) ───
HEADER = r"""\documentclass[border=4pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{amsmath}
\usepackage{xcolor}
\pgfplotsset{
  every axis/.append style={
    tick label style={font=\scriptsize},
    label style={font=\scriptsize},
    legend style={font=\tiny},
    grid=both, grid style={dotted,gray!30},
  }
}
\begin{document}
"""
FOOTER = "\n\\end{document}\n"
HEADER_EMBED = r"""%% ── VQE scatter figure (embed) ─────────────────────────────────
%% Requires: \usepackage{pgfplots} \pgfplotsset{compat=1.18}
%%           \usepackage{xcolor} \usepackage{amsmath}
\pgfplotsset{
  every axis/.append style={
    tick label style={font=\scriptsize},
    label style={font=\scriptsize},
    legend style={font=\tiny},
    grid=both, grid style={dotted,gray!30},
  }
}
"""
FOOTER_EMBED = "%% ── end VQE scatter figure ──────────────────────────────────────\n"


def display(mol, tex=False):
    return (DISPLAY_TEX if tex else DISPLAY_MPL).get(mol, mol)


# ── plot specifications ────────────────────────────────────────────
SPECS = [
    dict(key="scatter1_corrGPU_vs_orbitals", x=ORB, y=GHF,
         xlabel=r"Active space size (\% of orbitals)",
         ylabel=r"$E_{\mathrm{VQE}}^{\mathrm{GPU}}-E_{\mathrm{HF}}$ [Ha]",
         xlabel_mpl="Active space size (% of orbitals)",
         ylabel_mpl=r"$E_{VQE}^{GPU}-E_{HF}$ [Ha]",
         logy=False, parity=False),
    dict(key="scatter2_corrGPU_vs_electrons", x=ELE, y=GHF,
         xlabel=r"Active space size (\% of electrons)",
         ylabel=r"$E_{\mathrm{VQE}}^{\mathrm{GPU}}-E_{\mathrm{HF}}$ [Ha]",
         xlabel_mpl="Active space size (% of electrons)",
         ylabel_mpl=r"$E_{VQE}^{GPU}-E_{HF}$ [Ha]",
         logy=False, parity=False),
    dict(key="scatter3_ratio_vs_orbitals", x=ORB, y=CORR,
         xlabel=r"Active space size (\% of orbitals)",
         ylabel=r"$(E_{\mathrm{VQE}}^{\mathrm{GPU}}-E_{\mathrm{HF}})/E_{\mathrm{VQE}}^{\mathrm{GPU}}$",
         xlabel_mpl="Active space size (% of orbitals)",
         ylabel_mpl=r"$(E_{VQE}^{GPU}-E_{HF})/E_{VQE}^{GPU}$",
         logy=False, parity=False),
    dict(key="scatter4_ratio_vs_electrons", x=ELE, y=CORR,
         xlabel=r"Active space size (\% of electrons)",
         ylabel=r"$(E_{\mathrm{VQE}}^{\mathrm{GPU}}-E_{\mathrm{HF}})/E_{\mathrm{VQE}}^{\mathrm{GPU}}$",
         xlabel_mpl="Active space size (% of electrons)",
         ylabel_mpl=r"$(E_{VQE}^{GPU}-E_{HF})/E_{VQE}^{GPU}$",
         logy=False, parity=False),
    dict(key="scatter5_parity_CPU_vs_GPU", x=CHF, y=GHF,
         xlabel=r"$E_{\mathrm{VQE}}^{\mathrm{CPU}}-E_{\mathrm{HF}}$ [Ha]",
         ylabel=r"$E_{\mathrm{VQE}}^{\mathrm{GPU}}-E_{\mathrm{HF}}$ [Ha]",
         xlabel_mpl=r"$E_{VQE}^{CPU}-E_{HF}$ [Ha]",
         ylabel_mpl=r"$E_{VQE}^{GPU}-E_{HF}$ [Ha]",
         logy=False, parity=True),
    dict(key="scatter6_gap_vs_orbitals", x=ORB, y=GAP,
         xlabel=r"Active space size (\% of orbitals)",
         ylabel=r"$E_{\mathrm{VQE}}^{\mathrm{GPU}}-E_{\mathrm{VQE}}^{\mathrm{CPU}}$ [Ha]",
         xlabel_mpl="Active space size (% of orbitals)",
         ylabel_mpl=r"$E_{VQE}^{GPU}-E_{VQE}^{CPU}$ [Ha]",
         logy=True, parity=False),
    dict(key="scatter7_speedup_vs_orbitals", x=ORB, y=SPD,
         xlabel=r"Active space size (\% of orbitals)",
         ylabel=r"GPU speedup $t_{\mathrm{CPU}}/t_{\mathrm{GPU}}$",
         xlabel_mpl="Active space size (% of orbitals)",
         ylabel_mpl=r"GPU speedup $t_{CPU}/t_{GPU}$",
         logy=False, parity=False, hline=1.0),
    dict(key="scatter8_speedup_vs_qubits", x="qubits", y=SPD,
         xlabel=r"Active-space size (qubits $=2N_o$)",
         ylabel=r"GPU speedup $t_{\mathrm{CPU}}/t_{\mathrm{GPU}}$",
         xlabel_mpl=r"Active-space size (qubits $= 2N_o$)",
         ylabel_mpl=r"GPU speedup $t_{CPU}/t_{GPU}$",
         logy=False, parity=False, hline=1.0),
    dict(key="scatter9_corrRecovery_vs_speedup", x=SPD, y=CORR,
         xlabel=r"GPU speedup $t_{\mathrm{CPU}}/t_{\mathrm{GPU}}$",
         ylabel=r"Correlation recovery $(E_{\mathrm{VQE}}^{\mathrm{GPU}}-E_{\mathrm{HF}})/E_{\mathrm{VQE}}^{\mathrm{GPU}}$",
         xlabel_mpl=r"GPU speedup $t_{CPU}/t_{GPU}$",
         ylabel_mpl=r"Corr. recovery $(E_{VQE}^{GPU}-E_{HF})/E_{VQE}^{GPU}$",
         logy=False, parity=False),
    dict(key="scatter10_ccsdRecovery_vs_orbitals", x=ORB, y=REC,
         xlabel=r"Active space size (\% of orbitals)",
         ylabel=r"CCSD recovery $(E_{\mathrm{VQE}}^{\mathrm{GPU}}-E_{\mathrm{HF}})/(E_{\mathrm{CCSD}}-E_{\mathrm{HF}})$",
         xlabel_mpl="Active space size (% of orbitals)",
         ylabel_mpl=r"CCSD recovery $(E_{VQE}^{GPU}-E_{HF})/(E_{CCSD}-E_{HF})$",
         logy=False, parity=False),
    dict(key="scatter11_ccsdRecovery_vs_speedup", x=SPD, y=REC,
         xlabel=r"GPU speedup $t_{\mathrm{CPU}}/t_{\mathrm{GPU}}$",
         ylabel=r"CCSD recovery $(E_{\mathrm{VQE}}^{\mathrm{GPU}}-E_{\mathrm{HF}})/(E_{\mathrm{CCSD}}-E_{\mathrm{HF}})$",
         xlabel_mpl=r"GPU speedup $t_{CPU}/t_{GPU}$",
         ylabel_mpl=r"CCSD recovery $(E_{VQE}^{GPU}-E_{HF})/(E_{CCSD}-E_{HF})$",
         logy=False, parity=False),
    dict(key="scatter12_casciRecovery_vs_speedup", x=SPD, y=RECC,
         xlabel=r"GPU speedup $t_{\mathrm{CPU}}/t_{\mathrm{GPU}}$",
         ylabel=r"CASCI recovery $(E_{\mathrm{VQE}}^{\mathrm{GPU}}-E_{\mathrm{HF}})/(E_{\mathrm{CASCI}}-E_{\mathrm{HF}})$",
         xlabel_mpl=r"GPU speedup $t_{CPU}/t_{GPU}$",
         ylabel_mpl=r"CASCI recovery $(E_{VQE}^{GPU}-E_{HF})/(E_{CASCI}-E_{HF})$",
         logy=False, parity=False, hline=1.0, hline_label="VQE = CASCI (ideal)"),
    
        dict(key="scatter13_casciRecovery_vs_orbitals", x=ORB, y=RECC,
           xlabel=r"Active space size (\% of orbitals)",
         ylabel=r"CASCI recovery $(E_{\mathrm{VQE}}^{\mathrm{GPU}}-E_{\mathrm{HF}})/(E_{\mathrm{CASCI}}-E_{\mathrm{HF}})$",
         xlabel_mpl=r"Active space size (% of orbitals)",
         ylabel_mpl=r"CASCI recovery $(E_{VQE}^{GPU}-E_{HF})/(E_{CASCI}-E_{HF})$",
         logy=False, parity=False, hline=1.0, hline_label="VQE = CASCI (ideal)"),
]


def load():
    df = pd.read_csv(CSV)
    for c in [ORB, ELE, GHF, CHF, GAP, CORR, SPD, REC, RECC]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # derive (Ne, No) and qubit count from the "(Ne,No)" string
    parsed = df["Active Space (Ne,No)"].str.extract(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
    df["Ne"] = pd.to_numeric(parsed[0], errors="coerce")
    df["No"] = pd.to_numeric(parsed[1], errors="coerce")
    df["qubits"] = 2 * df["No"]
    present = [m for m in MOL_ORDER if m in set(df["Molecule"])]
    present += sorted(set(df["Molecule"]) - set(present))
    return df, present


def style_for(i):
    return (MPL_COLORS[i % len(MPL_COLORS)],
            MPL_MARKERS[i % len(MPL_MARKERS)])


# ── matplotlib ─────────────────────────────────────────────────────
def draw_mpl_axis(ax, df, present, spec):
    for i, mol in enumerate(present):
        sub = df[df["Molecule"] == mol]
        color, marker = style_for(i)
        ax.scatter(sub[spec["x"]], sub[spec["y"]], s=42, color=color,
                   marker=marker, edgecolors="black", linewidths=0.4,
                   alpha=0.9, label=display(mol))
    if spec["logy"]:
        ax.set_yscale("log")
    if spec["parity"]:
        vals = pd.concat([df[spec["x"]], df[spec["y"]]]).dropna()
        lo, hi = vals.min(), vals.max()
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, zorder=0, label="y = x")
    if spec.get("hline") is not None:
        ax.axhline(spec["hline"], color="0.4", ls="--", lw=1.0, zorder=0,
                   label=spec.get("hline_label", f"break-even ({spec['hline']:g})"))
    ax.set_xlabel(spec["xlabel_mpl"])
    ax.set_ylabel(spec["ylabel_mpl"])
    ax.grid(True, ls=":", color="0.8")


def make_png(df, present, spec):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    draw_mpl_axis(ax, df, present, spec)
    ax.legend(fontsize=7, ncol=2, loc="center left",
              bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(PNG_DIR, f"{spec['key']}.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"    -> {out}")
    plt.close(fig)


def make_panel(df, present):
    import math
    n = len(SPECS)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.5 * nrows))
    axflat = axes.ravel()
    merged = {}
    for ax, spec in zip(axflat, SPECS):
        draw_mpl_axis(ax, df, present, spec)
        ax.set_title(spec["key"].split("_", 1)[1].replace("_", " "), fontsize=9)
        for h, l in zip(*ax.get_legend_handles_labels()):
            merged.setdefault(l, h)          # dedupe legend entries by label
    for ax in axflat[n:]:                     # hide unused cells
        ax.set_visible(False)
    fig.legend(merged.values(), merged.keys(), fontsize=8, ncol=8,
               loc="lower center", bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "pdf"):
        out = os.path.join(PNG_DIR, f"scatter_panel.{ext}")
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"    -> {out}")
    plt.close(fig)


# ── pgfplots ───────────────────────────────────────────────────────
def fmt_coords(sub, xcol, ycol):
    pts = []
    for _, r in sub.iterrows():
        x, y = r[xcol], r[ycol]
        if pd.isna(x) or pd.isna(y):
            continue
        pts.append(f"({x:.6g},{y:.8e})")
    return " ".join(pts)


def make_tex(df, present, spec):
    axis_opts = [
        "width=11cm", "height=7.5cm",
        f"xlabel={{{spec['xlabel']}}}", f"ylabel={{{spec['ylabel']}}}",
        "legend pos=outer north east", "legend cell align=left",
        "legend columns=1", "scatter/use mapped color=false",
    ]
    if spec["logy"]:
        axis_opts.append("ymode=log")

    body = "\\begin{tikzpicture}\n\\begin{axis}[\n  "
    body += ",\n  ".join(axis_opts) + ",\n]\n"

    for i, mol in enumerate(present):
        sub = df[df["Molecule"] == mol]
        coords = fmt_coords(sub, spec["x"], spec["y"])
        if not coords:
            continue
        color = TIKZ_COLORS[i % len(TIKZ_COLORS)]
        mark  = TIKZ_MARKS[i % len(TIKZ_MARKS)]
        body += (f"\\addplot[only marks, color={color}, mark={mark}, "
                 f"mark size=1.7pt, mark options={{draw=black, line width=0.2pt}}] "
                 f"coordinates {{{coords}}};\n")
        body += f"\\addlegendentry{{{display(mol, tex=True)}}}\n"

    if spec["parity"]:
        vals = pd.concat([df[spec["x"]], df[spec["y"]]]).dropna()
        lo, hi = vals.min(), vals.max()
        body += (f"\\addplot[black, dashed, mark=none, line width=0.9pt] "
                 f"coordinates {{({lo:.8e},{lo:.8e}) ({hi:.8e},{hi:.8e})}};\n")
        body += "\\addlegendentry{$y=x$}\n"

    if spec.get("hline") is not None:
        xlo, xhi = df[spec["x"]].min(), df[spec["x"]].max()
        hv = spec["hline"]
        body += (f"\\addplot[gray, dashed, mark=none, line width=0.9pt] "
                 f"coordinates {{({xlo:.6g},{hv}) ({xhi:.6g},{hv})}};\n")
        body += f"\\addlegendentry{{{spec.get('hline_label', 'break-even')}}}\n"

    body += "\\end{axis}\n\\end{tikzpicture}\n"

    write(os.path.join(TEX_DIR, spec["key"] + ".tex"), HEADER + body + FOOTER)
    write(os.path.join(TEX_DIR, spec["key"] + "_embed.tex"),
          HEADER_EMBED + body + FOOTER_EMBED)


def write(path, content):
    with open(path, "w") as fh:
        fh.write(content)
    print(f"    -> {path}")


def main():
    df, present = load()
    print(f"  {len(df)} rows, {len(present)} molecules")
    print("\n── PNG (matplotlib) ───────────────────")
    for spec in SPECS:
        make_png(df, present, spec)
    make_panel(df, present)
    print("\n── TeX (pgfplots) ─────────────────────")
    for spec in SPECS:
        make_tex(df, present, spec)
    print("\n  done.")


if __name__ == "__main__":
    main()
