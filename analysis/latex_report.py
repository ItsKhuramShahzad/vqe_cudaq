# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
  VQE Benchmark -- LaTeX / PGFPlots Code Generator
  Reads CPU + GPU pkl files → writes ready-to-compile .tex figures

  *** UPDATED: HF / VQE / CCSD + convergence + CCSD diagnostics ***
═══════════════════════════════════════════════════════════════════════════

Output files (all in --out directory)
──────────────────────────────────────
  ── GPU Acceleration ──────────────────────────────────────────────────
  fig1-9   (runtime, speedup, energy CPU vs GPU)

  ── Quantum Accuracy -- HF / VQE / CCSD ───────────────────────────────
  fig10  VQE-HF improvement bars + CCSD reference line
  fig11  |E_VQE - E_CCSD| log-scale per config
  fig12  Correlation energy recovery η [%]
  fig13  Energy landscape: HF/CCSD lines + VQE scatter
  fig14  Signed ΔE [mHa] + chemical accuracy line
  fig16  VQE-HF bars (no CCSD, zoomed to VQE range)

  ── Performance Summary ───────────────────────────────────────────────
  fig15  4-panel subfigure: speedup / GPU-CPU delta / VQE-HF / relative VQE-HF

  ── New Research Analyses (from pkl internals) ────────────────────────
  fig17  VQE optimisation convergence curves (energy vs iteration)
  fig18  Runtime breakdown: quantum vs optimizer vs overhead
  fig19  CCSD T1/T2 diagnostics -- multireference character per molecule
  fig20  Circuit complexity: n_params + ham_terms vs qubits

  ── Summary ───────────────────────────────────────────────────────────
  vqe_benchmark_summary.csv   Full numerical table
"""

import os, sys, glob, pickle, argparse, warnings, math, colorsys
import numpy as np
from collections import defaultdict

warnings.filterwarnings("ignore")

JITTER = 0.06   # horizontal jitter per Ne level at the same No

# ──────────────────────────────────────────────────────────────────
#  CANONICAL MOLECULE ORDER  (from research grouping image)
#  Group 1 -- small reference molecules
#  Group 2 -- polycyclic aromatic hydrocarbons (PAH / acenes)
#  Group 3 -- functional group radicals
#  Group 4 -- DNA/RNA nucleobases
# ──────────────────────────────────────────────────────────────────
MOL_ORDER = [
    # small
    "Methylene", "Ethylene",
    # PAH / acenes
    "Benzene", "Naphthalene", "Benzaanthracene", "Pentacene",
    # functional groups
    "NH2-", "Methanamide",
    # nucleobases
    "Adenine", "Thymine", "Uracil", "Cytosine", "Guanine",
]

def sort_molecules(molecules):
    """Return molecules sorted by MOL_ORDER; unknowns appended alphabetically."""
    order_map = {m: i for i, m in enumerate(MOL_ORDER)}
    known   = [m for m in MOL_ORDER if m in molecules]
    unknown = sorted(m for m in molecules if m not in order_map)
    return known + unknown

# ──────────────────────────────────────────────────────────────────
#  CANONICAL ACTIVE-SPACE CONFIGURATIONS
#  These 9 (nele_cas, norb_cas) pairs are shared across all 13 molecules.
#  Non-standard configs (e.g. (2,2) or (4,6) for Methylene/Pentacene)
#  are excluded from plots so the color map is consistent everywhere.
# ──────────────────────────────────────────────────────────────────
CANONICAL_AS = sorted([
    (2, 3), (2, 4),
    (4, 3), (4, 4), (4, 5),
    (6, 4), (6, 5), (6, 6), (6, 7),
])

# ──────────────────────────────────────────────────────────────────
#  PKL DISPLAY NAMES
# ──────────────────────────────────────────────────────────────────
DISPLAY = {
    "Methylene"       : "Methylene",
    "Ethylene"        : "Ethylene",
    "Benzene"         : "Benzene",
    "Naphthalene"     : "Naphthalene",
    "Benzaanthracene" : r"Benz[a]anthracene",
    "Pentacene"       : "Pentacene",
    "NH2-"            : r"NH$_2^-$",
    "Methanamide"     : r"Methanamide",
    "Adenine"         : "Adenine",
    "Thymine"         : "Thymine",
    "Uracil"          : "Uracil",
    "Cytosine"        : "Cytosine",
    "Guanine"         : "Guanine",
}

def display(mol):
    return DISPLAY.get(mol, mol)

# ──────────────────────────────────────────────────────────────────
#  DATA LOADING
# ──────────────────────────────────────────────────────────────────

def mol_name_from_path(fp):
    base  = os.path.splitext(os.path.basename(fp))[0]
    parts = base.split("_")
    return parts[3] if len(parts) >= 4 else base


def parse_pkl(fp):
    with open(fp, "rb") as fh:
        raw = pickle.load(fh)

    mol_key   = next(iter(raw))
    mol       = raw[mol_key]
    ref_block = mol.get("references", {})
    ccsd_blk  = mol.get("ccsd", {})
    inp       = mol.get("input_spec", {})
    timing    = mol.get("timing", {})
    sys_sz    = mol.get("system_sizes", {})

    refs = {
        "E_hf"        : ref_block.get("E_hf_full"),
        "E_ccsd"      : ref_block.get("E_ccsd_full"),
        "basis"       : mol.get("basis", "cc-pVDZ"),
        # CCSD diagnostics (multireference character)
        "t1_norm"     : ccsd_blk.get("t1_norm"),
        "t2_norm"     : ccsd_blk.get("t2_norm"),
        "E_ccsd_corr" : ccsd_blk.get("E_ccsd_corr"),   # correlation energy
        # molecular metadata
        "formula"     : inp.get("formula", ""),
        "n_electrons" : inp.get("Total Electrons"),
        "n_orbitals"  : inp.get("Total Spatial Orbitals"),
        "nmo"         : sys_sz.get("nmo"),
        "nocc"        : sys_sz.get("nocc"),
        "nvir"        : sys_sz.get("nvir"),
        "scf_time_s"  : timing.get("pyscf_run_scf_seconds"),
    }

    runs = []
    for r in mol.get("active_space_runs", []):
        if r.get("skipped", False):
            continue
        sp   = r.get("space",       {})
        sz   = r.get("sizes",       {})
        vqe  = r.get("vqe",         {})
        cas  = r.get("casci",       {})
        cmp  = r.get("compare",     {})
        ham  = r.get("hamiltonian", {})
        th0  = r.get("theta0",      {})
        ne   = sp.get("nele_cas", 0)
        no   = sp.get("norb_cas", 0)
        E    = vqe.get("E_total")
        rt   = vqe.get("runtime")
        if E is None:
            continue

        # convergence history (subsample to <=200 pts to keep tex small)
        econv_raw = vqe.get("energy_convergence", [])
        if len(econv_raw) > 200:
            step = len(econv_raw) // 200
            econv = econv_raw[::step]
        else:
            econv = list(econv_raw)
        # absolute energies: E_nc_opt is the active-space part; add c0 for total
        c0 = ham.get("c0", 0.0)
        econv_total = [e + c0 for e in econv]

        # per-iteration quantum circuit times — full distribution stats
        qtimes_raw = vqe.get("quantum_times", [])
        if qtimes_raw:
            qt_arr   = np.array(qtimes_raw)
            qt_mean  = float(qt_arr.mean())
            qt_std   = float(qt_arr.std())
            qt_min   = float(qt_arr.min())
            qt_max   = float(qt_arr.max())
            qt_q1    = float(np.percentile(qt_arr, 25))
            qt_med   = float(np.percentile(qt_arr, 50))
            qt_q3    = float(np.percentile(qt_arr, 75))
        else:
            qt_mean = qt_std = qt_min = qt_max = qt_q1 = qt_med = qt_q3 = None

        # timing breakdown
        rt_quantum   = vqe.get("simulated_quantum_runtime", rt)
        rt_optimizer = vqe.get("optimizer_runtime", 0.0)
        rt_overhead  = (rt - rt_quantum - rt_optimizer) if (rt and rt_quantum) else 0.0

        # cycle-by-cycle energy improvement (mHa) and per-cycle runtime
        cycle_sums = vqe.get("cycle_summaries", [])
        cycle_dE = [cs.get("dE_vs_prev_best", 0.0) * 1000.0 for cs in cycle_sums]
        cycle_rt = [cs.get("runtime_total", 0.0) for cs in cycle_sums]

        # UCCSD optimised parameters (for parameter-magnitude analysis)
        theta_opt_raw = vqe.get("theta_opt", [])
        theta_opt = [float(x) for x in theta_opt_raw] if len(theta_opt_raw) else []

        # optimizer efficiency: fraction of evals that strictly improve energy
        if len(econv_raw) > 1:
            ec_arr     = np.array(econv_raw)
            n_improve  = int(np.sum(np.diff(ec_arr) < 0))
        else:
            n_improve  = 0

        runs.append({
            "ne"         : ne,  "no"       : no,
            "qubits"     : sz.get("qubits", 0),
            "n_params"   : sz.get("uccsd_num_parameters", 0),
            "E_vqe"      : E,
            "E_casci"    : cas.get("E_casci_total"),
            "runtime"    : rt,
            "rt_q"       : rt_quantum,
            "rt_opt"     : rt_optimizer,
            "rt_overhead": max(rt_overhead, 0.0),
            "converged"  : vqe.get("converged", False),
            "success_any": vqe.get("success_any", False),
            "cycles"     : vqe.get("cycles", 0),
            "nfev"       : vqe.get("nfev_total", 0),
            "ham_terms"  : ham.get("num_qubit_terms_nonconstant", 0),
            "ncore"      : sp.get("ncore", 0),
            "key"        : (ne, no),
            "label"      : f"({ne},{no})",
            "d_hf"       : cmp.get("d_vqe_minus_hf_full"),
            "d_ccsd"     : cmp.get("d_vqe_minus_ccsd_full"),
            "d_casci"    : cmp.get("d_vqe_minus_casci"),
            # convergence trace (subsampled <=200 pts, absolute E in Ha)
            "econv"      : econv_total,
            "nfev_actual": len(econv_raw),
            # quantum time distribution
            "qt_mean"    : qt_mean,
            "qt_std"     : qt_std,
            "qt_min"     : qt_min,
            "qt_max"     : qt_max,
            "qt_q1"      : qt_q1,
            "qt_med"     : qt_med,
            "qt_q3"      : qt_q3,
            # parameter analysis
            "theta_opt"  : theta_opt,
            "theta0_norm": th0.get("theta0_norm"),
            # cycle analysis
            "cycle_dE"   : cycle_dE,
            "cycle_rt"   : cycle_rt,
            # optimizer efficiency
            "n_improve"  : n_improve,
            "best_per_cycle": vqe.get("best_energy_per_cycle", []),
        })

    # MO orbital energies for orbital-spectrum figure (Fig B)
    pyscf_blk   = mol.get("pyscf_insights", {}).get("pyscf", {})
    mo_energy_r = pyscf_blk.get("mo_energy", [])
    refs["mo_energy"] = [float(x) for x in mo_energy_r]  # full MO energy list
    refs["nocc"]      = sys_sz.get("nocc", 0)

    return mol_key, runs, refs


def load_directory(dirpath):
    out  = {}
    pkls = sorted(glob.glob(os.path.join(dirpath, "*.pkl")))
    if not pkls:
        print(f"  WARNING: no .pkl files found in '{dirpath}'")
    for fp in pkls:
        try:
            mol_key, runs, refs = parse_pkl(fp)
            name = mol_name_from_path(fp)
            out[name] = {"runs": runs, "refs": refs, "mol_key": mol_key}
            print(f"  ✓  {name:26s}  {len(runs):2d} runs")
        except Exception as exc:
            print(f"  ✗  {fp}  →  {exc}")
    return out


def load_data(cpu_dir, gpu_dir):
    print("\n── CPU pkl files ──────────────────────────")
    cpu = load_directory(cpu_dir)
    print("\n── GPU pkl files ──────────────────────────")
    gpu = load_directory(gpu_dir)
    matched  = sort_molecules(set(cpu) & set(gpu))   # canonical order
    only_cpu = sorted(set(cpu) - set(gpu))
    only_gpu = sorted(set(gpu) - set(cpu))
    if only_cpu: print(f"\n  CPU-only: {only_cpu}")
    if only_gpu: print(f"  GPU-only: {only_gpu}")
    print(f"\n  Matched molecules ({len(matched)}): {matched}")
    return cpu, gpu, matched


# ──────────────────────────────────────────────────────────────────
#  DATA UTILITIES
# ──────────────────────────────────────────────────────────────────

def pair_runs(cpu_runs, gpu_runs):
    cpu_map = {r["key"]: r for r in cpu_runs}
    gpu_map = {r["key"]: r for r in gpu_runs}
    pairs   = []
    for key in sorted(set(cpu_map) | set(gpu_map)):
        cr = cpu_map.get(key)
        gr = gpu_map.get(key)
        if not (cr and gr):
            continue
        sp = (cr["runtime"] / gr["runtime"]
              if cr["runtime"] and gr["runtime"] and gr["runtime"] > 0
              else None)
        pairs.append({
            "key"    : key,
            "ne"     : key[0],  "no"    : key[1],
            "label"  : f"({key[0]},{key[1]})",
            "qubits" : cr["qubits"],
            "E_cpu"  : cr["E_vqe"],
            "E_gpu"  : gr["E_vqe"],
            "rt_cpu" : cr["runtime"],
            "rt_gpu" : gr["runtime"],
            "speedup": sp,
            "cpu"    : cr,      "gpu"   : gr,
        })
    return pairs


def best_per_norb(runs):
    bst = {}
    for r in runs:
        no, E = r["no"], r["E_vqe"]
        if no not in bst or E < bst[no]:
            bst[no] = E
    return bst


def jitter_pairs(pairs):
    """
    Return [(jittered_x, cpu_y, gpu_y, speedup, label, pair), ...]
    with small x-offsets so multiple Ne at the same No don't overlap.
    """
    by_no = defaultdict(list)
    for p in pairs:
        by_no[p["no"]].append(p)

    out = []
    for no in sorted(by_no):
        pts = sorted(by_no[no], key=lambda p: p["ne"])
        n   = len(pts)
        off = np.linspace(-JITTER*(n-1)/2, JITTER*(n-1)/2, n)
        for i, p in enumerate(pts):
            out.append((no + off[i], p["rt_cpu"], p["rt_gpu"],
                        p["speedup"], p["label"], p))
    return out


def corr_recovery(E_vqe, E_hf, E_ccsd):
    """
    Fraction of CCSD correlation energy captured by VQE.
    Returns value in [0, 1] (can exceed 1 if VQE < CCSD).
    Returns None if references are missing or denominator is ~0.
    """
    if None in (E_vqe, E_hf, E_ccsd):
        return None
    denom = E_ccsd - E_hf          # negative: correlation energy
    if abs(denom) < 1e-10:
        return None
    return (E_vqe - E_hf) / denom  # dimensionless, ideally → 1


# ──────────────────────────────────────────────────────────────────
#  LaTeX HELPERS
# ──────────────────────────────────────────────────────────────────

def sym(ne, no):
    return f"{ne}-{no}"


def coords_str(items, fmt=".6f"):
    return " ".join(f"({x},{y:{fmt}})" for x, y in items)


HEADER = r"""\documentclass[border=4pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{pgfplotstable}
\usepackage{amsmath}
\usepackage{xcolor}
\usepgfplotslibrary{groupplots}
\pgfplotsset{
  every axis/.append style={
    tick label style={font=\scriptsize},
    label style={font=\scriptsize},
    legend style={font=\scriptsize},
    grid=both, grid style={dotted,gray!30},
  }
}
\begin{document}
"""

FOOTER = r"""
\end{document}
"""

# ── Embeddable header/footer (no \documentclass, no \begin{document}) ──────
# Use these when \input{}-ing figures into an existing .tex document.
# The pgfplotsset block applies only if not already set in the main preamble;
# it is safe to include multiple times (append style accumulates).
HEADER_EMBED = r"""\pgfplotsset{
  every axis/.append style={
    tick label style={font=\scriptsize},
    label style={font=\scriptsize},
    legend style={font=\scriptsize},
    grid=both, grid style={dotted,gray!30},
  }
}
"""

FOOTER_EMBED = ""

# 11 visually distinct TikZ colours
TIKZ_COLORS = [
    "green!50!black", "violet", "orange", "teal",
    "blue!70!black",  "red!70!black", "magenta", "gray!60!black",
    "brown",          "cyan!60!black","purple!60!black",
]


def write(path, content):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"    → {path}")


def write_readme(cpu_dir, gpu_dir, outdir, cpu, gpu, molecules, args):
    """Write README.txt to outdir documenting the full provenance of this run."""
    import datetime, platform, re

    now = datetime.datetime.now()
    sep = "=" * 72

    def file_info_block(dirpath, label):
        pkls = sorted(glob.glob(os.path.join(dirpath, "*.pkl")))
        lines = [f"{label} DIRECTORY", f"  Path : {os.path.abspath(dirpath)}",
                 f"  Files ({len(pkls)} .pkl files):"]
        for fp in pkls:
            st = os.stat(fp)
            mtime = datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            size_kb = st.st_size / 1024
            lines.append(f"    {os.path.abspath(fp)}")
            lines.append(f"      size: {size_kb:.1f} KB    modified: {mtime}")
        return "\n".join(lines)

    only_cpu = sorted(set(cpu) - set(gpu))
    only_gpu = sorted(set(gpu) - set(cpu))

    lines = [
        sep,
        "  VQE LaTeX Generator -- Run Provenance README",
        sep,
        "",
        f"Generated     : {now.strftime('%Y-%m-%d  %H:%M:%S')}",
        f"Script        : {os.path.abspath(__file__)}",
        f"Python        : {sys.version.split()[0]}  ({platform.python_implementation()})",
        f"Platform      : {platform.system()} {platform.release()} ({platform.machine()})",
        f"Command line  : {' '.join(sys.argv)}",
        "",
        sep,
        "  OUTPUT DIRECTORY",
        sep,
        f"  Path : {os.path.abspath(outdir)}",
        "",
        sep,
        "  INPUT FILES",
        sep,
        "",
        file_info_block(cpu_dir, "CPU"),
        "",
        file_info_block(gpu_dir, "GPU"),
        "",
        sep,
        "  MOLECULES",
        sep,
        f"  Matched ({len(molecules)}):",
    ]
    for i, mol in enumerate(molecules, 1):
        n_cpu = len(cpu[mol]["runs"]) if mol in cpu else 0
        n_gpu = len(gpu[mol]["runs"]) if mol in gpu else 0
        lines.append(f"    {i:2d}. {mol:<26s}  CPU runs: {n_cpu}   GPU runs: {n_gpu}")
    if only_cpu:
        lines.append(f"\n  CPU-only (no GPU match): {only_cpu}")
    if only_gpu:
        lines.append(f"  GPU-only (no CPU match): {only_gpu}")
    lines += [
        "",
        sep,
        "  FIGURE SELECTION",
        sep,
        f"  Figures requested : {sorted(args.figs, key=lambda x: (0, int(x)) if x.isdigit() else (1, x))}",
        "",
        sep,
        "  FIGURE LABELS  (\\ref{} names)",
        sep,
        "  For each embeddable figure: the \\input path and its \\label.",
        "  Reference in text with \\ref{<label>} or \\cref{<label>}.",
        "",
    ]

    def _fig_key(fp):
        b = os.path.basename(fp)
        m = re.match(r"fig(\d+)", b)
        return (0, int(m.group(1)), b) if m else (1, 0, b)

    for fp in sorted(glob.glob(os.path.join(outdir, "*_embed*.tex")), key=_fig_key):
        base = os.path.splitext(os.path.basename(fp))[0]
        try:
            labs = re.findall(r"\\label\{([^}]+)\}", open(fp, encoding="utf-8").read())
        except Exception:
            labs = []
        lab_str = ", ".join(labs) if labs else "(no \\label)"
        lines.append(f"    \\input{{{outdir}/{base}}}")
        lines.append(f"        label: {lab_str}")
    lines += ["", sep]

    readme_path = os.path.join(outdir, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"    → {readme_path}")


def write_both(outdir, basename, figure_body):
    """
    Write two files:
      basename.tex        -- standalone, compilable with pdflatex on its own
      basename_embed.tex  -- embeddable, use \\input{basename_embed.tex} in
                            your main document (no \\documentclass wrapper)

    figure_body must be the raw figure environment content -- everything
    between HEADER/FOOTER and HEADER_EMBED/FOOTER_EMBED is identical.
    """
    # standalone version
    standalone = HEADER + figure_body + FOOTER
    path_sa = os.path.join(outdir, f"{basename}.tex")
    write(path_sa, standalone)

    # embed version -- strip the standalone document wrapper
    embed = HEADER_EMBED + figure_body + FOOTER_EMBED
    path_em = os.path.join(outdir, f"{basename}_embed.tex")
    write(path_em, embed)


def export_preamble(outdir):
    """
    Write a vqe_preamble.tex that the user can \\input{} once in the
    preamble of their main document.  Idempotent -- safe to call multiple times.
    """
    content = r"""% ════════════════════════════════════════════════════════════════
% vqe_preamble.tex  --  include ONCE in your document preamble:
%     \input{tex_out/vqe_preamble.tex}
% ════════════════════════════════════════════════════════════════
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{groupplots}
\usepackage{pgfplotstable}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{subcaption}

\pgfplotsset{
  every axis/.append style={
    tick label style={font=\scriptsize},
    label style={font=\scriptsize},
    legend style={font=\scriptsize},
    grid=both, grid style={dotted,gray!30},
  }
}
% ════════════════════════════════════════════════════════════════
"""
    path = os.path.join(outdir, "vqe_preamble.tex")
    write(path, content)





# ──────────────────────────────────────────────────────────────────
#  LABEL-PLACEMENT HELPER
#  Returns the \node annotation string for molecule name labels.
#  - Default : top-right corner (anchor=north east)
#  - Cytosine: bottom-left corner (anchor=south west) -- avoids the
#              flat, tightly-clustered data that sits at top-right.
# ──────────────────────────────────────────────────────────────────
def mol_label_node(mol_name, mol_disp, bold=False, anchor=None, pos=None):
    """
    Returns a LaTeX \node command placing the molecule name label inside the panel.
    Placed AFTER \addplot commands so it renders above bars (correct z-order).

    mol_name : canonical key (e.g. 'Cytosine')
    mol_disp : LaTeX display string (e.g. r'Benz[a]anthracene')
    bold     : if True, adds \bfseries
    anchor   : explicit anchor override (e.g. 'north west', 'center', 'south west')
    pos      : explicit rel axis cs position override (e.g. '0.02,0.98')

    Position presets (pass anchor + pos):
      top-right   : anchor='north east', pos='0.98,0.98'
      top-left    : anchor='north west', pos='0.02,0.98'
      center      : anchor='center',     pos='0.5,0.5'
      bottom-left : anchor='south west', pos='0.02,0.02'
    """
    font_cmd = r"\scriptsize\bfseries" if bold else r"\scriptsize"
    if anchor is not None and pos is not None:
        # Explicit override — use as given
        pass
    elif mol_name == "Cytosine":
        anchor = "south west"
        pos    = "0.02,0.02"
    else:
        anchor = "north east"
        pos    = "0.98,0.98"
    return (
        f"\\node[anchor={anchor}, font={font_cmd}] at "
        f"(rel axis cs:{pos}) {{{mol_disp}}};\n"
    )

# ══════════════════════════════════════════════════════════════════
#  FIG 1 -- E_VQE vs N_o^(a)  (line plots, best energy per No)
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
#  FIG 1 -- E_VQE vs N_o^(a)  (line plots, best energy per No)
# ══════════════════════════════════════════════════════════════════

def fig1_energy_vs_norb(cpu, gpu, molecules, outdir):
    """
    VQE total energy for EVERY active-space configuration vs the complete
    active space (nele_cas, norb_cas), one panel per molecule.

    Marker design:
      - CPU series : solid blue line + filled blue circle (mark=*)
      - GPU series : dashed red line + OPEN red triangle (mark=triangle)
      Open GPU triangles let the blue CPU circle show through when CPU
      and GPU coincide to sub-mHa precision.

    Axis design (fixes the "missing blue at rightmost x" problem):
      The raw data span for molecules like Thymine and Guanine can be as
      small as ~5 mHa across the full N_o^(a) range. Without explicit
      y-padding, pgfplots places the lowest data point (which is usually
      the rightmost x = highest N_o^(a), since more correlation = lower
      energy) exactly on ymin. pgfplots then CLIPS the marker against
      the axis frame, and the blue CPU point at x=6 or x=7 disappears
      into the bottom border.
      Fix: compute ymin/ymax explicitly per panel with 8% headroom on
      both sides, and disable marker clipping. Both markers remain
      fully visible even when they sit at the extreme value.
    """
    print("\n[Fig 1] E_VQE vs active orbitals …")
    NCOLS = 4
    n     = len(molecules)
    nrows = (n + NCOLS) // NCOLS

    blocks = []
    for panel_idx, mol in enumerate(molecules):
        col_idx = panel_idx % NCOLS
        ylabel_opt = r"ylabel={$E_{\mathrm{VQE}}$ [Ha]}" if col_idx == 0 else "ylabel={}"

        # x-axis = the COMPLETE active space (N_e^(a), N_o^(a)) = (nele_cas,
        # norb_cas). Each configuration gets its OWN x slot (symbolic coord),
        # so configs that previously stacked at a shared N_o are now spread
        # out and nothing overlaps. With one energy per x slot a single line
        # connects every configuration. Matched CPU/GPU pairs so each slot
        # carries both a CPU and a GPU energy. Ordered by (N_e, N_o).
        pairs = pair_runs(cpu[mol]["runs"], gpu[mol]["runs"])
        if not pairs:
            continue
        pairs.sort(key=lambda p: (p["ne"], p["no"]))

        sym_coords = [sym(p["ne"], p["no"]) for p in pairs]   # e.g. "2-4"
        sym_labels = [p["label"]            for p in pairs]   # e.g. "(2,4)"
        sym_list   = ",".join(sym_coords)
        lbl_list   = ",".join(sym_labels)

        cpu_pts = " ".join(f"({sym(p['ne'],p['no'])},{p['E_cpu']:.6f})"
                           for p in pairs if p["E_cpu"] is not None)
        gpu_pts = " ".join(f"({sym(p['ne'],p['no'])},{p['E_gpu']:.6f})"
                           for p in pairs if p["E_gpu"] is not None)

        # ── Explicit y-range with padding so extremum markers aren't clipped.
        all_E = ([p["E_cpu"] for p in pairs if p["E_cpu"] is not None]
                 + [p["E_gpu"] for p in pairs if p["E_gpu"] is not None])
        if all_E:
            ymin_raw = min(all_E)
            ymax_raw = max(all_E)
            span     = max(ymax_raw - ymin_raw, 1e-6)
            pad      = span * 0.08
            ymin_v   = ymin_raw - pad
            ymax_v   = ymax_raw + pad
            yrange = (
                f"  ymin={ymin_v:.6f}, ymax={ymax_v:.6f},\n"
                f"  scaled y ticks=false,\n"
                f"  y tick label style={{/pgf/number format/fixed,"
                f" /pgf/number format/precision=3}},\n"
            )
        else:
            yrange = ""

        # clip marker paths=false keeps boundary markers fully drawn.
        # CPU solid blue line + filled circles; GPU dashed red line + open
        # triangles (drawn second so the hollow triangle reveals the CPU
        # circle wherever the two energies coincide).
        blocks.append(
            f"% === {mol} ===\n"
            f"\\nextgroupplot[\n"
            f"  {ylabel_opt},\n"
            f"  symbolic x coords={{{sym_list}}},\n"
            f"  xtick=data,\n"
            f"  xticklabels={{{lbl_list}}},\n"
            f"  x tick label style={{font=\\tiny, rotate=45, anchor=east}},\n"
            f"{yrange}"
            f"  clip marker paths=false,\n"
            f"  enlarge x limits=0.08,\n"
            f"]\n"
            f"\\addplot[blue, thick, mark=*, mark size=2pt]"
            f" coordinates {{{cpu_pts}}};\n"
            f"\\addplot[red, dashed, thick, mark=triangle,"
            f" mark size=3pt, mark options={{fill=none, line width=0.9pt}}]"
            f" coordinates {{{gpu_pts}}};\n"
            + mol_label_node(mol, display(mol))
        )

    blocks.append(
        r"% === Legend ===" + "\n"
        r"\nextgroupplot[hide axis]" + "\n"
        r"\addlegendimage{mark=*, blue, thick, mark size=2.2pt}" + "\n"
        r"\addlegendentry{VQE (CPU)}" + "\n"
        r"\addlegendimage{mark=triangle, red, dashed, thick,"
        r" mark size=3pt, mark options={fill=none, line width=0.9pt}}" + "\n"
        r"\addlegendentry{VQE (GPU)}" + "\n"
    )

    body = (
        HEADER
        + r"\begin{figure*}[htbp]" + "\n"
        + r"\centering\resizebox{1.1\textwidth}{!}{%" + "\n"
        + r"\begin{tikzpicture}" + "\n"
        + r"\begin{groupplot}[" + "\n"
        + f"  group style={{group size={NCOLS} by {nrows},"
        + r"horizontal sep=1.4cm,vertical sep=1.1cm}," + "\n"
        + r"  width=0.26\textwidth, height=0.32\textwidth," + "\n"
        + r"  tick label style={font=\scriptsize}," + "\n"
        + r"  label style={font=\scriptsize}," + "\n"
        + r"  legend style={font=\scriptsize, at={(0.5,-0.25)},"
        + r"anchor=north, legend columns=2}" + "\n"
        + "]\n\n"
        + "\n".join(blocks)
        + r"\end{groupplot}" + "\n"
        + r"\end{tikzpicture}" + "\n"
        + r"}\caption{$E_{\mathrm{VQE}}^{\mathrm{CPU}}$ (solid blue, filled"
        + r" circles) and $E_{\mathrm{VQE}}^{\mathrm{GPU}}$ (dashed red,"
        + r" open triangles) for every active-space configuration, plotted"
        + r" against the \emph{complete} active space"
        + r" $(N_e^{(a)}, N_o^{(a)}) = (\texttt{nele\_cas}, \texttt{norb\_cas})$,"
        + r" cc-pVDZ basis. Each $(N_e^{(a)}, N_o^{(a)})$ occupies its own"
        + r" $x$ position (ordered by $N_e^{(a)}$ then $N_o^{(a)}$), so"
        + r" configurations are no longer stacked: for a fixed orbital count"
        + r" the energy still depends on the electron count, with half-filled"
        + r" active spaces capturing the most correlation (lowest energy)."
        + r" Open red triangles allow the blue CPU markers to remain visible"
        + r" at configurations where CPU and GPU energies coincide to within"
        + r" sub-mHa precision.}" + "\n"
        + r"\label{fig:vqe_cpu_gpu_energy_allpairs}" + "\n"
        + r"\end{figure*}" + "\n"
        + FOOTER
    )
    write_both(outdir, "fig1_energy_vs_norb", body.replace(HEADER, "").replace(HEADER_SUBCAP, "").replace(FOOTER, "").strip())

# ══════════════════════════════════════════════════════════════════
#  FIG 2 -- E_VQE per (Ne,No) config  (grouped bar, y dir=reverse)
# ══════════════════════════════════════════════════════════════════

def fig2_energy_per_config(cpu, gpu, molecules, outdir):
    print("\n[Fig 2] E_VQE per (Ne,No) configuration …")
    NCOLS = 4
    nrows = (len(molecules) + NCOLS) // NCOLS

    blocks = []
    for panel_idx, mol in enumerate(molecules):
        col_idx = panel_idx % NCOLS
        ylabel_opt = "" if col_idx == 0 else "  ylabel={},\n"
        pairs = pair_runs(cpu[mol]["runs"], gpu[mol]["runs"])
        if not pairs:
            continue
        pairs.sort(key=lambda p: p["E_cpu"])
        sym_coords = [sym(p["ne"], p["no"]) for p in pairs]
        sym_labels = [p["label"]             for p in pairs]
        cpu_pts = " ".join(f"({sym(p['ne'],p['no'])},{p['E_cpu']:.6f})" for p in pairs)
        gpu_pts = " ".join(f"({sym(p['ne'],p['no'])},{p['E_gpu']:.6f})" for p in pairs)
        sym_list = ",".join(sym_coords); lbl_list = ",".join(sym_labels)
        blocks.append(
            f"% === {mol} ===\n"
            f"\\nextgroupplot[\n  symbolic x coords={{{sym_list}}},\n"
            f"  xticklabels={{{lbl_list}}},\n"
            f"  xticklabel style={{font=\\tiny,xshift=4.5pt,yshift=-3pt}},\n"
            f"{ylabel_opt}]\n"
            f"\\addplot+[ybar, bar shift=-2pt, fill=blue!60] coordinates {{{cpu_pts}}};\n"
            f"\\addplot+[ybar, bar shift=+2pt, fill=red!60]  coordinates {{{gpu_pts}}};\n"
            + mol_label_node(mol, display(mol), anchor='north east', pos='0.98,0.98') + "\n"
        )
    blocks.append(
        r"\nextgroupplot[hide axis, legend style={at={(0.5,0.5)}, anchor=center}]" + "\n"
        r"\addlegendimage{ybar, bar shift=-2pt, fill=blue!60}" + "\n"
        r"\addlegendentry{VQE (CPU)}" + "\n"
        r"\addlegendimage{ybar, bar shift=+2pt, fill=red!60}" + "\n"
        r"\addlegendentry{VQE (GPU)}" + "\n"
    )

    body = (
        HEADER
        + r"\begin{figure*}[htbp]" + "\n"
        + r"\centering\resizebox{1.1\textwidth}{!}{%" + "\n"
        + r"\begin{tikzpicture}" + "\n"
        + r"\begin{groupplot}[" + "\n"
        + f"  group style={{group size={NCOLS} by {nrows},"
        + r"horizontal sep=1.4cm,vertical sep=1cm}," + "\n"
        + r"  width=0.27\textwidth, height=0.34\textwidth," + "\n"
        + r"  ybar, bar width=3pt, y dir=reverse," + "\n"
        + r"  xtick=data, x tick label style={rotate=45, anchor=east}," + "\n"
        + r"  tick label style={font=\scriptsize}, label style={font=\scriptsize}," + "\n"
        + r"  ymajorgrids=true, grid style={dotted,gray!30}," + "\n"
        + r"  ylabel={$E_{\mathrm{VQE}}$ [Ha]}," + "\n"
        + "]\n\n"
        + "\n".join(blocks)
        + r"\end{groupplot}" + "\n"
        + r"\end{tikzpicture}" + "\n"
        + r"}\caption{$E_{\mathrm{VQE}}^{\mathrm{CPU}}$ (blue) and "
        + r"$E_{\mathrm{VQE}}^{\mathrm{GPU}}$ (red) per $(N_e^{(a)}, N_o^{(a)})$ config.}" + "\n"
        + r"\label{fig:vqe_cpu_gpu_energy_allpairs_2}" + "\n"
        + r"\end{figure*}" + "\n"
        + FOOTER
    )
    write_both(outdir, "fig2_energy_per_config", body.replace(HEADER, "").replace(HEADER_SUBCAP, "").replace(FOOTER, "").strip())


# ══════════════════════════════════════════════════════════════════
#  FIG 3 -- Runtime per (Ne,No) config  (log-scale grouped bar)
# ══════════════════════════════════════════════════════════════════

def fig3_runtime_per_config(cpu, gpu, molecules, outdir):
    print("\n[Fig 3] Runtime per (Ne,No) configuration …")
    NCOLS = 4
    nrows = (len(molecules) + NCOLS) // NCOLS

    blocks = []
    for panel_idx, mol in enumerate(molecules):
        col_idx = panel_idx % NCOLS
        ylabel_opt = "" if col_idx == 0 else "  ylabel={},\n"
        pairs = pair_runs(cpu[mol]["runs"], gpu[mol]["runs"])
        if not pairs:
            continue
        pairs.sort(key=lambda p: (p["qubits"], p["ne"], p["no"]))
        sym_coords = [sym(p["ne"], p["no"]) for p in pairs]
        sym_labels = [p["label"]             for p in pairs]
        cpu_pts = " ".join(f"({sym(p['ne'],p['no'])},{max(p['rt_cpu'],0.001):.6f})" for p in pairs)
        gpu_pts = " ".join(f"({sym(p['ne'],p['no'])},{max(p['rt_gpu'],0.001):.6f})" for p in pairs)
        sym_list = ",".join(sym_coords); lbl_list = ",".join(sym_labels)
        blocks.append(
            f"% === {mol} ===\n"
            f"\\nextgroupplot[\n  symbolic x coords={{{sym_list}}},\n"
            f"  xticklabels={{{lbl_list}}},\n"
            f"  xticklabel style={{font=\\tiny,xshift=4.5pt,yshift=-3pt}},\n"
            f"{ylabel_opt}]\n"
            f"\\addplot+[ybar, bar shift=-2pt, fill=blue!60] coordinates {{{cpu_pts}}};\n"
            f"\\addplot+[ybar, bar shift=+2pt, fill=red!60]  coordinates {{{gpu_pts}}};\n"
            + mol_label_node(mol, display(mol), anchor='north west', pos='0.02,0.98') + "\n"
        )
    blocks.append(
        r"\nextgroupplot[hide axis, legend style={at={(0.5,0.5)}, anchor=center}]" + "\n"
        r"\addlegendimage{ybar, bar shift=-2pt, fill=blue!60}" + "\n"
        r"\addlegendentry{CPU runtime}" + "\n"
        r"\addlegendimage{ybar, bar shift=+2pt, fill=red!60}" + "\n"
        r"\addlegendentry{GPU runtime}" + "\n"
    )

    body = (
        HEADER
        + r"\begin{figure*}[htbp]" + "\n"
        + r"\centering\resizebox{1.1\textwidth}{!}{%" + "\n"
        + r"\begin{tikzpicture}" + "\n"
        + r"\begin{groupplot}[" + "\n"
        + f"  group style={{group size={NCOLS} by {nrows},"
        + r"horizontal sep=1.4cm,vertical sep=1cm}," + "\n"
        + r"  width=0.27\textwidth, height=0.34\textwidth," + "\n"
        + r"  ybar, bar width=3pt, xtick=data," + "\n"
        + r"  x tick label style={rotate=45, anchor=east}," + "\n"
        + r"  tick label style={font=\scriptsize}, label style={font=\scriptsize}," + "\n"
        + r"  ymajorgrids=true, grid style={dotted,gray!30}," + "\n"
        + r"  ymode=log, ylabel={Runtime (s)}," + "\n"
        + "]\n\n"
        + "\n".join(blocks)
        + r"\end{groupplot}" + "\n"
        + r"\end{tikzpicture}" + "\n"
        + r"}\caption{CPU (blue) vs.\ GPU (red) VQE runtime per $(N_e^{(a)}, N_o^{(a)})$. "
        + r"Logarithmic scale.}" + "\n"
        + r"\label{fig:vqe_cpu_gpu_runtime_allpairs}" + "\n"
        + r"\end{figure*}" + "\n"
        + FOOTER
    )
    write_both(outdir, "fig3_runtime_per_config", body.replace(HEADER, "").replace(HEADER_SUBCAP, "").replace(FOOTER, "").strip())


# ══════════════════════════════════════════════════════════════════
#  FIG 4 -- GPU Speedup per (Ne,No)
# ══════════════════════════════════════════════════════════════════

def fig4_speedup_per_config(cpu, gpu, molecules, outdir):
    print("\n[Fig 4] Speedup per (Ne,No) configuration …")
    NCOLS = 4
    nrows = (len(molecules) + NCOLS) // NCOLS

    blocks = []
    for panel_idx, mol in enumerate(molecules):
        col_idx = panel_idx % NCOLS
        ylabel_opt = "" if col_idx == 0 else "  ylabel={},\n"
        pairs = [p for p in pair_runs(cpu[mol]["runs"], gpu[mol]["runs"])
                 if p["speedup"] is not None]
        if not pairs:
            continue
        pairs.sort(key=lambda p: p["speedup"])
        sym_coords = [sym(p["ne"], p["no"]) for p in pairs]
        sym_labels = [p["label"]             for p in pairs]
        spd_pts = " ".join(f"({sym(p['ne'],p['no'])},{p['speedup']:.6f})" for p in pairs)
        sym_list = ",".join(sym_coords); lbl_list = ",".join(sym_labels)
        blocks.append(
            f"% === {mol} ===\n"
            f"\\nextgroupplot[\n  symbolic x coords={{{sym_list}}},\n"
            f"  xticklabels={{{lbl_list}}},\n"
            f"  xticklabel style={{font=\\tiny,xshift=4.5pt,yshift=-3pt}},\n"
            f"{ylabel_opt}]\n"
            f"\\addplot+[ybar, fill=green!65!black] coordinates {{{spd_pts}}};\n"
            + mol_label_node(mol, display(mol), anchor='north west', pos='0.02,0.98') + "\n"
        )
    blocks.append(
        r"\nextgroupplot[hide axis, legend style={at={(0.5,0.5)}, anchor=center}]" + "\n"
        r"\addlegendimage{ybar, fill=green!65!black}" + "\n"
        r"\addlegendentry{Speedup $t_{\mathrm{CPU}}/t_{\mathrm{GPU}}$}" + "\n"
    )

    body = (
        HEADER
        + r"\begin{figure*}[htbp]" + "\n"
        + r"\centering\resizebox{1.1\textwidth}{!}{%" + "\n"
        + r"\begin{tikzpicture}" + "\n"
        + r"\begin{groupplot}[" + "\n"
        + f"  group style={{group size={NCOLS} by {nrows},"
        + r"horizontal sep=1.3cm,vertical sep=1.0cm}," + "\n"
        + r"  width=0.27\textwidth, height=0.34\textwidth," + "\n"
        + r"  ybar, bar width=3pt, xtick=data," + "\n"
        + r"  x tick label style={rotate=45, anchor=east}," + "\n"
        + r"  tick label style={font=\scriptsize}, label style={font=\scriptsize}," + "\n"
        + r"  ymajorgrids=true, grid style={dotted,gray!30}," + "\n"
        + r"  ymode=log, ymin=1," + "\n"
        + r"  ylabel={Speedup ($t_{\mathrm{CPU}}/t_{\mathrm{GPU}}$)}," + "\n"
        + "]\n\n"
        + "\n".join(blocks)
        + r"\end{groupplot}" + "\n"
        + r"\end{tikzpicture}" + "\n"
        + r"}\caption{GPU speedup $t_{\mathrm{CPU}}/t_{\mathrm{GPU}}$ per "
        + r"$(N_e^{(a)}, N_o^{(a)})$ across all molecules.}" + "\n"
        + r"\label{fig:vqe_cpu_gpu_speedup_allpairs_2}" + "\n"
        + r"\end{figure*}" + "\n"
        + FOOTER
    )
    write_both(outdir, "fig4_speedup_per_config", body.replace(HEADER, "").replace(HEADER_SUBCAP, "").replace(FOOTER, "").strip())


# ══════════════════════════════════════════════════════════════════
#  FIG 5 -- Runtime vs qubit count
# ══════════════════════════════════════════════════════════════════

def fig5_runtime_vs_qubits(cpu, gpu, molecules, outdir):
    print("\n[Fig 5] Runtime vs qubit count …")
    NCOLS = 4
    nrows = (len(molecules) + NCOLS) // NCOLS

    blocks = []
    for panel_idx, mol in enumerate(molecules):
        col_idx = panel_idx % NCOLS
        ylabel_opt = "" if col_idx == 0 else "ylabel={}"
        pairs = pair_runs(cpu[mol]["runs"], gpu[mol]["runs"])
        if not pairs:
            continue
        by_q = defaultdict(list)
        for p in pairs:
            by_q[p["qubits"]].append(p)
        reps   = {q: min(ps, key=lambda p: (p["ne"], p["no"])) for q, ps in by_q.items()}
        qvals  = sorted(reps)
        cpu_pts = " ".join(f"({q},{max(reps[q]['rt_cpu'],0.001):.3f})" for q in qvals)
        gpu_pts = " ".join(f"({q},{max(reps[q]['rt_gpu'],0.001):.3f})" for q in qvals)
        blocks.append(
            f"% === {mol} ===\n"
            f"\\nextgroupplot[{ylabel_opt}]\n"
            f"\\addplot+[ybar, bar shift=-2pt, fill=blue!60] coordinates {{{cpu_pts}}};\n"
            f"\\addplot+[ybar, bar shift=+2pt, fill=red!60]  coordinates {{{gpu_pts}}};\n"
            + mol_label_node(mol, display(mol), anchor='north west', pos='0.02,0.98') + "\n"
        )
    blocks.append(
        r"\nextgroupplot[hide axis, legend style={at={(0.5,0.5)}, anchor=center}]" + "\n"
        r"\addlegendimage{ybar, bar shift=-2pt, fill=blue!60}" + "\n"
        r"\addlegendentry{Runtime (CPU)}" + "\n"
        r"\addlegendimage{ybar, bar shift=+2pt, fill=red!60}" + "\n"
        r"\addlegendentry{Runtime (GPU)}" + "\n"
    )

    body = (
        HEADER
        + r"\begin{figure*}[htbp]" + "\n"
        + r"\centering\resizebox{1.1\textwidth}{!}{%" + "\n"
        + r"\begin{tikzpicture}" + "\n"
        + r"\begin{groupplot}[" + "\n"
        + f"  group style={{group size={NCOLS} by {nrows},"
        + r"horizontal sep=1.5cm,vertical sep=1.0cm}," + "\n"
        + r"  width=0.27\textwidth, height=0.34\textwidth," + "\n"
        + r"  ybar, bar width=3pt, ymode=log, ymin=1," + "\n"
        + r"  xtick=data, x tick label style={font=\scriptsize}," + "\n"
        + r"  tick label style={font=\scriptsize}, label style={font=\scriptsize}," + "\n"
        + r"  ymajorgrids=true, grid style={dotted,gray!30}," + "\n"
        + r"  xlabel={Qubit count}, ylabel={Runtime [s]}," + "\n"
        + "]\n\n"
        + "\n".join(blocks)
        + r"\end{groupplot}" + "\n"
        + r"\end{tikzpicture}" + "\n"
        + r"}\caption{CPU (blue) vs.\ GPU (red) runtime as a function of qubit count. "
        + r"Log-scale highlights scalability advantage.}" + "\n"
        + r"\label{fig:vqe_cpu_gpu_runtime_vs_qubits}" + "\n"
        + r"\end{figure*}" + "\n"
        + FOOTER
    )
    write_both(outdir, "fig5_runtime_vs_qubits", body.replace(HEADER, "").replace(HEADER_SUBCAP, "").replace(FOOTER, "").strip())


# ══════════════════════════════════════════════════════════════════
#  FIG 6 -- Speedup vs N_o^(a)  (line + jitter)
# ══════════════════════════════════════════════════════════════════

def fig6_speedup_vs_norb(cpu, gpu, molecules, outdir):
    print("\n[Fig 6] Speedup vs active orbitals (line + jitter) …")
    NCOLS = 4
    nrows = (len(molecules) + NCOLS) // NCOLS

    blocks = []
    for idx, mol in enumerate(molecules):
        col_idx = idx % NCOLS
        ylabel_opt = r"ylabel={Speedup ($t_{\mathrm{CPU}}/t_{\mathrm{GPU}}$)}" if col_idx == 0 else "ylabel={}"
        pairs = [p for p in pair_runs(cpu[mol]["runs"], gpu[mol]["runs"])
                 if p["speedup"] is not None]
        if not pairs:
            continue
        pairs.sort(key=lambda p: (p["ne"], p["no"]))
        sym_list = ",".join(sym(p["ne"], p["no"]) for p in pairs)
        lbl_list = ",".join(p["label"] for p in pairs)
        pts_str  = " ".join(f"({sym(p['ne'],p['no'])},{p['speedup']:.6f})"
                            for p in pairs)
        blocks.append(
            f"% === {mol} ===\n"
            f"\\nextgroupplot[\n  {ylabel_opt},\n"
            f"  symbolic x coords={{{sym_list}}},\n"
            f"  xtick=data, xticklabels={{{lbl_list}}},\n"
            f"  x tick label style={{font=\\tiny, rotate=45, anchor=east}},\n"
            f"]\n"
            f"\\addplot[mark=*, black, thick] coordinates {{{pts_str}}};\n"
            + mol_label_node(mol, display(mol), anchor='north west', pos='0.02,0.98') + "\n"
        )

    body = (
        HEADER
        + r"\begin{figure*}[htbp]" + "\n"
        + r"\centering\resizebox{1.1\textwidth}{!}{%" + "\n"
        + r"\begin{tikzpicture}" + "\n"
        + r"\begin{groupplot}[" + "\n"
        + f"  group style={{group size={NCOLS} by {nrows},"
        + r"horizontal sep=1.2cm,vertical sep=1.3cm}," + "\n"
        + r"  width=0.27\textwidth, height=0.34\textwidth," + "\n"
        + r"  ymin=0.9," + "\n"
        + r"  tick label style={font=\scriptsize}, label style={font=\scriptsize}," + "\n"
        + "]\n\n"
        + "\n".join(blocks)
        + r"\end{groupplot}" + "\n"
        + r"\end{tikzpicture}" + "\n"
        + r"}\caption{GPU speedup per active-space configuration"
        + r" $(N_e^{(a)},N_o^{(a)})$ for all molecules (cc-pVDZ).}" + "\n"
        + r"\label{fig:speedup_all_No_a}" + "\n"
        + r"\end{figure*}" + "\n"
        + FOOTER
    )
    write_both(outdir, "fig6_speedup_vs_norb", body.replace(HEADER, "").replace(HEADER_SUBCAP, "").replace(FOOTER, "").strip())


# ══════════════════════════════════════════════════════════════════
#  FIG 7 -- Runtime vs N_o^(a)  (log-scale line + jitter)
# ══════════════════════════════════════════════════════════════════

def fig7_runtime_vs_norb(cpu, gpu, molecules, outdir):
    print("\n[Fig 7] Runtime vs active orbitals (log line + jitter) …")
    NCOLS = 4
    nrows = (len(molecules) + NCOLS) // NCOLS

    blocks = []
    for panel_idx, mol in enumerate(molecules):
        col_idx = panel_idx % NCOLS
        ylabel_opt = "ylabel={Runtime [s]}" if col_idx == 0 else "ylabel={}"
        pairs = pair_runs(cpu[mol]["runs"], gpu[mol]["runs"])
        if not pairs:
            continue
        pairs.sort(key=lambda p: (p["ne"], p["no"]))
        sym_list = ",".join(sym(p["ne"], p["no"]) for p in pairs)
        lbl_list = ",".join(p["label"] for p in pairs)
        c_str = " ".join(f"({sym(p['ne'],p['no'])},{p['rt_cpu']:.4f})"
                         for p in pairs if p["rt_cpu"])
        g_str = " ".join(f"({sym(p['ne'],p['no'])},{p['rt_gpu']:.4f})"
                         for p in pairs if p["rt_gpu"])
        blocks.append(
            f"% === {mol} ===\n"
            f"\\nextgroupplot[\n  {ylabel_opt},\n"
            f"  symbolic x coords={{{sym_list}}},\n"
            f"  xtick=data, xticklabels={{{lbl_list}}},\n"
            f"  x tick label style={{font=\\tiny, rotate=45, anchor=east}},\n"
            f"]\n"
            f"\\addplot+[mark=*, blue, thick] coordinates {{{c_str}}};\n"
            f"\\addplot+[mark=triangle*, red, dashed, thick] coordinates {{{g_str}}};\n"
            + mol_label_node(mol, display(mol), anchor='north west', pos='0.02,0.98') + "\n"
        )
    blocks.append(r"\nextgroupplot[hide axis]" + "\n"
                  r"\legend{CPU runtime, GPU runtime}" + "\n")

    body = (
        HEADER
        + r"\begin{figure*}[htbp]" + "\n"
        + r"\centering\resizebox{1.1\textwidth}{!}{%" + "\n"
        + r"\begin{tikzpicture}" + "\n"
        + r"\begin{groupplot}[" + "\n"
        + f"  group style={{group size={NCOLS} by {nrows},"
        + r"horizontal sep=1.2cm,vertical sep=1.3cm}," + "\n"
        + r"  width=0.27\textwidth, height=0.34\textwidth," + "\n"
        + r"  ymode=log," + "\n"
        + r"  tick label style={font=\scriptsize}, label style={font=\scriptsize}," + "\n"
        + r"  legend style={font=\scriptsize, at={(0.5,-0.25)},"
        + r"anchor=north, legend columns=2}," + "\n"
        + "]\n\n"
        + "\n".join(blocks)
        + r"\end{groupplot}" + "\n"
        + r"\end{tikzpicture}" + "\n"
        + r"}\caption{CPU (solid blue) vs.\ GPU (dashed red) runtimes per"
        + r" active-space configuration $(N_e^{(a)},N_o^{(a)})$ "
        + r"(cc-pVDZ).}" + "\n"
        + r"\label{fig:ccpvdz_runtime_all_molecules}" + "\n"
        + r"\end{figure*}" + "\n"
        + FOOTER
    )
    write_both(outdir, "fig7_runtime_vs_norb", body.replace(HEADER, "").replace(HEADER_SUBCAP, "").replace(FOOTER, "").strip())


# ══════════════════════════════════════════════════════════════════
#  FIG 8 -- Speedup heatmap
# ══════════════════════════════════════════════════════════════════

def fig8_speedup_heatmap(cpu, gpu, molecules, outdir):
    print("\n[Fig 8] Speedup heatmap (matrix) …")
    all_q = sorted({p["qubits"]
                    for mol in molecules
                    for p in pair_runs(cpu[mol]["runs"], gpu[mol]["runs"])})
    rows = []
    for mol in molecules:
        by_q = defaultdict(list)
        for p in pair_runs(cpu[mol]["runs"], gpu[mol]["runs"]):
            if p["speedup"] is not None:
                by_q[p["qubits"]].append(p["speedup"])
        rows.append([max(by_q[q]) if by_q.get(q) else 0 for q in all_q])

    matrix_block = ""
    for i, mol in enumerate(molecules):
        for j, q in enumerate(all_q):
            v = rows[i][j]
            if v > 0:
                matrix_block += f"({j},{i},{v:.2f})\n"

    body = (
        HEADER
        + r"\begin{figure}[htbp]" + "\n"
        + r"\centering" + "\n"
        + r"\begin{tikzpicture}" + "\n"
        + r"\begin{axis}[" + "\n"
        + f"  width={max(10, len(all_q)*1.4)}cm," + "\n"
        + f"  height={max(6, len(molecules)*0.7)}cm," + "\n"
        + r"  colormap/RdYlGn, colorbar," + "\n"
        + r"  colorbar style={ylabel={Speedup}}," + "\n"
        + r"  view={0}{90}," + "\n"
        + r"  xtick={" + ",".join(str(j) for j in range(len(all_q))) + "}," + "\n"
        + r"  xticklabels={" + ",".join(f"{q}q" for q in all_q) + "}," + "\n"
        + r"  ytick={" + ",".join(str(i) for i in range(len(molecules))) + "}," + "\n"
        + r"  yticklabels={" + ",".join(display(m) for m in molecules) + "}," + "\n"
        + r"  tick label style={font=\small}," + "\n"
        + r"  title={GPU Speedup $t_{\mathrm{CPU}}/t_{\mathrm{GPU}}$}," + "\n"
        + "]\n"
        + r"\addplot3[matrix plot*, point meta=explicit," + "\n"
        + r"  mesh/cols=" + str(len(all_q)) + r", shader=flat corner]" + "\n"
        + r"coordinates {" + "\n" + matrix_block + r"};" + "\n"
        + r"\end{axis}" + "\n"
        + r"\end{tikzpicture}" + "\n"
        + r"\caption{Maximum GPU speedup $t_{\mathrm{CPU}}/t_{\mathrm{GPU}}$ "
        + r"per molecule and qubit count.}" + "\n"
        + r"\label{fig:speedup_heatmap}" + "\n"
        + r"\end{figure}" + "\n"
        + FOOTER
    )
    write_both(outdir, "fig8_speedup_heatmap", body.replace(HEADER, "").replace(HEADER_SUBCAP, "").replace(FOOTER, "").strip())


# ══════════════════════════════════════════════════════════════════
#  FIG 9 -- All molecules combined speedup on ONE axis
# ══════════════════════════════════════════════════════════════════

def fig9_speedup_combined(cpu, gpu, molecules, outdir):
    print("\n[Fig 9] Combined speedup -- all molecules, single panel …")
    series_blocks = []
    for idx, mol in enumerate(molecules):
        col   = TIKZ_COLORS[idx % len(TIKZ_COLORS)]
        pairs = [p for p in pair_runs(cpu[mol]["runs"], gpu[mol]["runs"])
                 if p["speedup"] is not None]
        if not pairs:
            continue
        pts_str = " ".join(
            f"({x:.4f},{p['speedup']:.6f})"
            for x, _, _, sp, lbl, p in jitter_pairs(pairs)
        )
        series_blocks.append(
            f"\\addplot+[mark=*, {col}, thick] coordinates {{{pts_str}}};\n"
            f"\\addlegendentry{{{display(mol)}}}\n"
        )

    body = (
        HEADER
        + r"\begin{figure}[htbp]" + "\n"
        + r"\centering" + "\n"
        + r"\begin{tikzpicture}" + "\n"
        + r"\begin{axis}[" + "\n"
        + r"  width=14cm, height=9cm," + "\n"
        + r"  xlabel={Active orbitals $N_o^{(a)}$}," + "\n"
        + r"  ylabel={Speedup ($t_{\mathrm{CPU}}/t_{\mathrm{GPU}}$)}," + "\n"
        + r"  xtick={3,4,5,6,7}, ymin=0.9," + "\n"
        + r"  tick label style={font=\small}, label style={font=\small}," + "\n"
        + r"  legend style={font=\tiny, at={(1.01,1)}, anchor=north west, legend columns=1}," + "\n"
        + r"  grid=both, grid style={dotted,gray!30}," + "\n"
        + r"  title={GPU Speedup vs Active Orbitals --- All Molecules (cc-pVDZ)}," + "\n"
        + "]\n"
        + r"\draw[gray,dashed] ({rel axis cs:0,0} |- {axis cs:0,1}) --"
        + r" ({rel axis cs:1,0} |- {axis cs:0,1});" + "\n\n"
        + "\n".join(series_blocks)
        + r"\end{axis}" + "\n"
        + r"\end{tikzpicture}" + "\n"
        + r"\caption{GPU speedup vs.\ $N_o^{(a)}$ for all molecules on one panel.}" + "\n"
        + r"\label{fig:speedup_combined}" + "\n"
        + r"\end{figure}" + "\n"
        + FOOTER
    )
    write_both(outdir, "fig9_speedup_combined", body.replace(HEADER, "").replace(HEADER_SUBCAP, "").replace(FOOTER, "").strip())


# ══════════════════════════════════════════════════════════════════
#  FIG 10 -- 4-WAY ENERGY COMPARISON: HF / VQE_CPU / VQE_GPU / CCSD
#           Best (lowest) VQE energy per molecule, grouped bar
#           Highlights: how VQE sits between HF and CCSD, GPU == CPU accuracy
# ══════════════════════════════════════════════════════════════════

def fig10_energy_comparison(cpu, gpu, molecules, outdir):
    """
    Per-molecule GROUPED BAR CHART -- VQE improvement over HF.

    KEY PHYSICS:
      E_VQE - E_HF  is NEGATIVE  →  VQE lowers energy below HF  (improvement)
      E_CCSD - E_HF is NEGATIVE  →  CCSD is the maximum recoverable target
      More negative = more correlation energy captured = better result

    DESIGN:
      - Blue bars : E_VQE_CPU - E_HF  [mHa]  (bars go DOWN from 0 = HF baseline)
      - Red  bars : E_VQE_GPU - E_HF  [mHa]
      - Green dashed horizontal line : E_CCSD - E_HF  (target ceiling)
        drawn via \\pgfplotsextra{\\draw} so it is NOT rendered as a bar
      - y-axis ZOOMED to the VQE range -- so bars fill the subplot
        ymax = 0  (HF baseline),  ymin = min(VQE-HF) × 1.25
      - NO y dir=reverse -- bars going DOWN is intuitive for "lowering energy"

    This makes VQE bars fully visible and the CCSD line shows how close VQE
    gets to the correlation energy ceiling.
    """
    print("\n[Fig 10] VQE improvement over HF -- zoomed bars + CCSD reference …")
    NCOLS = 4
    nrows = (len(molecules) + NCOLS - 1) // NCOLS + 1   # +1 for legend row

    blocks = []
    for mol in molecules:
        refs   = cpu[mol]["refs"]
        E_hf   = refs.get("E_hf")
        E_ccsd = refs.get("E_ccsd")
        pairs  = pair_runs(cpu[mol]["runs"], gpu[mol]["runs"])
        if not pairs or E_hf is None:
            continue

        pairs.sort(key=lambda p: (p["ne"], p["no"]))
        sym_coords = [sym(p["ne"], p["no"]) for p in pairs]
        sym_labels = [p["label"]             for p in pairs]
        sym_list   = ",".join(sym_coords)
        lbl_list   = ",".join(sym_labels)

        # ΔE = E_method - E_HF [mHa]  →  always negative for improvement
        vc_pts, vg_pts = [], []
        vqe_deltas = []
        for p in pairs:
            sc = sym(p["ne"], p["no"])
            if p["E_cpu"] is not None:
                d = (p["E_cpu"] - E_hf) * 1000
                vc_pts.append(f"({sc},{d:.4f})")
                vqe_deltas.append(d)
            if p["E_gpu"] is not None:
                d = (p["E_gpu"] - E_hf) * 1000
                vg_pts.append(f"({sc},{d:.4f})")
                vqe_deltas.append(d)

        if not vqe_deltas:
            continue

        # CCSD reference level (single value, drawn as horizontal line)
        ccsd_delta_mha = (E_ccsd - E_hf) * 1000 if E_ccsd is not None else None

        # Zoom y-axis to VQE range: ymax=0, ymin below the most-negative VQE bar
        ymin_v = min(vqe_deltas) * 1.30   # 30% headroom below lowest bar
        if ccsd_delta_mha is not None:
            ymin_v = min(ymin_v, ccsd_delta_mha * 1.15)
        ymax_v = max(vqe_deltas) * 0.10   # small positive headroom above 0

        # CCSD reference line via pgfplotsextra (bypasses ybar renderer)
        ccsd_line = ""
        if ccsd_delta_mha is not None:
            first_sc = sym_coords[0]
            last_sc  = sym_coords[-1]
            ccsd_line = (
                r"\pgfplotsextra{%" + "\n"
                f"  \\draw[green!60!black, dashed, line width=1.4pt]\n"
                f"    ({{axis cs:{first_sc},{ccsd_delta_mha:.4f}}}-|"
                f"{{rel axis cs:0,0}}) -- "
                f"({{axis cs:{last_sc},{ccsd_delta_mha:.4f}}}-|"
                f"{{rel axis cs:1,0}});\n"
                r"}%" + "\n"
            )

        # Also draw the HF baseline at y=0 for clarity
        hf_line = (
            r"\pgfplotsextra{%" + "\n"
            r"  \draw[gray!70, solid, line width=1pt]"
            r" (rel axis cs:0,0|-{axis cs:0,0}) -- "
            r"(rel axis cs:1,0|-{axis cs:0,0});" + "\n"
            r"}%" + "\n"
        )

        col_idx10 = len(blocks) % NCOLS
        ylabel_10 = "" if col_idx10 == 0 else "  ylabel={},\n"
        blocks.append(
            f"% === {mol} ===\n"
            f"\\nextgroupplot[\n"
            f"  symbolic x coords={{{sym_list}}},\n"
            f"  xticklabels={{{lbl_list}}},\n"
            f"  xticklabel style={{font=\\tiny, rotate=45, anchor=east}},\n"
            f"  ymin={ymin_v:.2f}, ymax={ymax_v:.2f},\n"
            f"  scaled y ticks=false,\n"
            f"  y tick label style={{/pgf/number format/fixed, font=\\tiny}},\n"
            f"{ylabel_10}]\n"
            + hf_line
            + ccsd_line
            + (f"\\addplot+[ybar, bar shift=-2.5pt, bar width=3.5pt,"
               f" fill=blue!60, draw=blue!80] coordinates {{{' '.join(vc_pts)}}};\n"
               if vc_pts else "")
            + (f"\\addplot+[ybar, bar shift=+2.5pt, bar width=3.5pt,"
               f" fill=red!55, draw=red!80]  coordinates {{{' '.join(vg_pts)}}};\n"
               if vg_pts else "")
            + mol_label_node(mol, display(mol), bold=True, anchor='center', pos='0.5,0.5')
            + "\n"
        )

    if not blocks:
        print("  SKIP -- no data with E_hf reference.")
        return

    blocks.append(
        r"\nextgroupplot[hide axis,"
        r" legend style={at={(0.5,0.5)}, anchor=center,"
        r" legend columns=1, font=\small, row sep=4pt}]" + "\n"
        r"\addlegendimage{ybar, fill=blue!60, draw=blue!80}" + "\n"
        r"\addlegendentry{$E_{\mathrm{VQE}}^{\mathrm{CPU}} - E_{\mathrm{HF}}$ [mHa]}" + "\n"
        r"\addlegendimage{ybar, fill=red!55, draw=red!80}" + "\n"
        r"\addlegendentry{$E_{\mathrm{VQE}}^{\mathrm{GPU}} - E_{\mathrm{HF}}$ [mHa]}" + "\n"
        r"\addlegendimage{green!60!black, dashed, line width=1.4pt}" + "\n"
        r"\addlegendentry{$E_{\mathrm{CCSD}} - E_{\mathrm{HF}}$ (target) [mHa]}" + "\n"
        r"\addlegendimage{gray!70, solid, line width=1pt}" + "\n"
        r"\addlegendentry{HF baseline ($\Delta E = 0$)}" + "\n"
    )

    body = (
        HEADER
        + r"\begin{figure*}[htbp]" + "\n"
        + r"\centering\resizebox{1.1\textwidth}{!}{%" + "\n"
        + r"\begin{tikzpicture}" + "\n"
        + r"\begin{groupplot}[" + "\n"
        + f"  group style={{group size={NCOLS} by {nrows},"
        + r"horizontal sep=1.6cm,vertical sep=1.5cm}," + "\n"
        + r"  width=0.27\textwidth, height=0.34\textwidth," + "\n"
        + r"  ybar, xtick=data," + "\n"
        + r"  tick label style={font=\scriptsize}," + "\n"
        + r"  label style={font=\scriptsize}," + "\n"
        + r"  ymajorgrids=true, grid style={dotted,gray!30}," + "\n"
        + r"  ylabel={$\Delta E$ from HF [mHa]}," + "\n"
        + "]\n\n"
        + "\n".join(blocks)
        + r"\end{groupplot}" + "\n"
        + r"\end{tikzpicture}" + "\n"
        + r"}" + "\n"
        + r"\caption{VQE energy improvement over Hartree--Fock per active-space"
        + r" configuration $(N_e^{(a)}, N_o^{(a)})$."
        + r" Bars show $\Delta E = E_{\mathrm{VQE}} - E_{\mathrm{HF}}$ [mHa]"
        + r" (always negative: lower = more correlation captured):"
        + r" CPU (blue) and GPU (red)."
        + r" The grey line marks the HF baseline ($\Delta E = 0$)."
        + r" The dashed green line marks $E_{\mathrm{CCSD}} - E_{\mathrm{HF}}$"
        + r" --- the maximum correlation energy recoverable at the CCSD level."
        + r" The $y$-axis is zoomed to the VQE range so that the per-configuration"
        + r" trends are clearly visible; as $(N_e^{(a)}, N_o^{(a)})$ grows,"
        + r" VQE bars extend further down, confirming systematic improvement over HF"
        + r" and convergence toward the CCSD target.}" + "\n"
        + r"\label{fig:energy_comparison_hf_vqe_ccsd}" + "\n"
        + r"\end{figure*}" + "\n"
        + FOOTER
    )
    write_both(outdir, "fig10_energy_comparison", body.replace(HEADER, "").replace(HEADER_SUBCAP, "").replace(FOOTER, "").strip())


# ══════════════════════════════════════════════════════════════════
#  FIG 16 -- VQE IMPROVEMENT OVER HF  (same as fig10, NO CCSD line)
#           Pure VQE_CPU vs VQE_GPU comparison relative to HF
#           y-axis zoomed to VQE range only
# ══════════════════════════════════════════════════════════════════

def fig16_vqe_vs_hf_only(cpu, gpu, molecules, outdir):
    """
    Identical layout to fig10 but with the CCSD reference line removed.
    Focuses purely on VQE_CPU vs VQE_GPU improvement over HF, making
    the small CPU/GPU differences more visible without the CCSD ceiling
    drawing the eye.

    x-axis : active-space config (Ne,No)
    y-axis : E_VQE - E_HF [mHa]  (negative = improvement over HF)
      - Blue bars : E_VQE_CPU - E_HF
      - Red  bars : E_VQE_GPU - E_HF
      - Grey line : HF baseline at y = 0
    y-axis zoomed to VQE range so per-config variation is clearly visible.
    """
    print("\n[Fig 16] VQE improvement over HF -- no CCSD reference …")
    NCOLS = 4
    nrows = (len(molecules) + NCOLS - 1) // NCOLS + 1

    blocks = []
    for mol in molecules:
        refs  = cpu[mol]["refs"]
        E_hf  = refs.get("E_hf")
        pairs = pair_runs(cpu[mol]["runs"], gpu[mol]["runs"])
        if not pairs or E_hf is None:
            continue

        pairs.sort(key=lambda p: (p["ne"], p["no"]))
        sym_coords = [sym(p["ne"], p["no"]) for p in pairs]
        sym_labels = [p["label"]             for p in pairs]
        sym_list   = ",".join(sym_coords)
        lbl_list   = ",".join(sym_labels)

        vc_pts, vg_pts = [], []
        vqe_deltas     = []
        for p in pairs:
            sc = sym(p["ne"], p["no"])
            if p["E_cpu"] is not None:
                d = (p["E_cpu"] - E_hf) * 1000
                vc_pts.append(f"({sc},{d:.4f})")
                vqe_deltas.append(d)
            if p["E_gpu"] is not None:
                d = (p["E_gpu"] - E_hf) * 1000
                vg_pts.append(f"({sc},{d:.4f})")
                vqe_deltas.append(d)

        if not vqe_deltas:
            continue

        ymin_v = min(vqe_deltas) * 1.30
        ymax_v = max(vqe_deltas) * 0.10

        # HF baseline at y=0
        hf_line = (
            r"\pgfplotsextra{%" + "\n"
            r"  \draw[gray!70, solid, line width=1pt]"
            r" (rel axis cs:0,0|-{axis cs:0,0}) -- "
            r"(rel axis cs:1,0|-{axis cs:0,0});" + "\n"
            r"}%" + "\n"
        )

        col_idx16 = len(blocks) % NCOLS
        ylabel_16 = "" if col_idx16 == 0 else "  ylabel={},\n"
        blocks.append(
            f"% === {mol} ===\n"
            f"\\nextgroupplot[\n"
            f"  symbolic x coords={{{sym_list}}},\n"
            f"  xticklabels={{{lbl_list}}},\n"
            f"  xticklabel style={{font=\\tiny, rotate=45, anchor=east}},\n"
            f"  ymin={ymin_v:.2f}, ymax={ymax_v:.2f},\n"
            f"  scaled y ticks=false,\n"
            f"  y tick label style={{/pgf/number format/fixed, font=\\tiny}},\n"
            f"{ylabel_16}]\n"
            + hf_line
            + (f"\\addplot+[ybar, bar shift=-2.5pt, bar width=3.5pt,"
               f" fill=blue!60, draw=blue!80] coordinates {{{' '.join(vc_pts)}}};\n"
               if vc_pts else "")
            + (f"\\addplot+[ybar, bar shift=+2.5pt, bar width=3.5pt,"
               f" fill=red!55, draw=red!80]  coordinates {{{' '.join(vg_pts)}}};\n"
               if vg_pts else "")
            + mol_label_node(mol, display(mol), bold=True, anchor='south west', pos='0.02,0.02')
            + "\n"
        )

    if not blocks:
        print("  SKIP -- no data with E_hf reference.")
        return

    blocks.append(
        r"\nextgroupplot[hide axis,"
        r" legend style={at={(0.5,0.5)}, anchor=center,"
        r" legend columns=1, font=\small, row sep=4pt}]" + "\n"
        r"\addlegendimage{ybar, fill=blue!60, draw=blue!80}" + "\n"
        r"\addlegendentry{$E_{\mathrm{VQE}}^{\mathrm{CPU}} - E_{\mathrm{HF}}$ [mHa]}" + "\n"
        r"\addlegendimage{ybar, fill=red!55, draw=red!80}" + "\n"
        r"\addlegendentry{$E_{\mathrm{VQE}}^{\mathrm{GPU}} - E_{\mathrm{HF}}$ [mHa]}" + "\n"
        r"\addlegendimage{gray!70, solid, line width=1pt}" + "\n"
        r"\addlegendentry{HF baseline ($\Delta E = 0$)}" + "\n"
    )

    body = (
        HEADER
        + r"\begin{figure*}[htbp]" + "\n"
        + r"\centering\resizebox{1.1\textwidth}{!}{%" + "\n"
        + r"\begin{tikzpicture}" + "\n"
        + r"\begin{groupplot}[" + "\n"
        + f"  group style={{group size={NCOLS} by {nrows},"
        + r"horizontal sep=1.6cm,vertical sep=1.5cm}," + "\n"
        + r"  width=0.27\textwidth, height=0.34\textwidth," + "\n"
        + r"  ybar, xtick=data," + "\n"
        + r"  tick label style={font=\scriptsize}," + "\n"
        + r"  label style={font=\scriptsize}," + "\n"
        + r"  ymajorgrids=true, grid style={dotted,gray!30}," + "\n"
        + r"  ylabel={$\Delta E$ from HF [mHa]}," + "\n"
        + "]\n\n"
        + "\n".join(blocks)
        + r"\end{groupplot}" + "\n"
        + r"\end{tikzpicture}" + "\n"
        + r"}" + "\n"
        + r"\caption{VQE energy improvement over Hartree--Fock per active-space"
        + r" configuration $(N_e^{(a)}, N_o^{(a)})$, without CCSD reference."
        + r" Bars show $\Delta E = E_{\mathrm{VQE}} - E_{\mathrm{HF}}$ [mHa]"
        + r" (always negative: lower = more correlation captured):"
        + r" CPU (blue) and GPU (red)."
        + r" The grey line marks the HF baseline ($\Delta E = 0$)."
        + r" The $y$-axis is zoomed to the VQE range, making CPU/GPU"
        + r" agreement and per-configuration trends clearly visible."
        + r" As $(N_e^{(a)}, N_o^{(a)})$ grows, bars extend further down,"
        + r" confirming systematic VQE improvement over HF with increasing active space.}" + "\n"
        + r"\label{fig:vqe_vs_hf_only}" + "\n"
        + r"\end{figure*}" + "\n"
        + FOOTER
    )
    write_both(outdir, "fig16_vqe_vs_hf_only", body.replace(HEADER, "").replace(HEADER_SUBCAP, "").replace(FOOTER, "").strip())


# ══════════════════════════════════════════════════════════════════
#  FIG 11 -- ENERGY ERROR vs CCSD REFERENCE
#           |E_VQE - E_CCSD|  and  |E_HF - E_CCSD|  per (Ne,No) config
#           Shows how much better VQE is vs the HF baseline
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
#  FIG 11 -- ENERGY ERROR vs CCSD REFERENCE
#           |E_VQE - E_CCSD|  per (Ne,No) config
# ══════════════════════════════════════════════════════════════════

def fig11_energy_error_vs_ccsd(cpu, gpu, molecules, outdir):
    """
    Absolute VQE energy error relative to CCSD (log scale).
      - blue bars  = |E_VQE_CPU - E_CCSD|  (Ha)
      - red bars   = |E_VQE_GPU - E_CCSD|  (Ha)
    HF is intentionally omitted here -- its error is 2-3 orders of magnitude
    larger than VQE and would compress the VQE bars into invisibility on any
    shared axis. The HF gap is shown in fig14 (mHa comparison) and fig13
    (energy landscape). Uses d_ccsd field where available; falls back to
    direct subtraction.

    Fixes vs. previous version:
    (1) Each panel gets an EXPLICIT (ymin, ymax) derived from its own
        data plus log origin y=-infty. Without this, pgfplots auto-ranges
        log-scale panels independently; for molecules whose VQE-CCSD
        errors cluster in a tiny linear window (Guanine ~1.67, Uracil
        ~1.22, Cytosine ~1.22, Thymine ~1.39) the resulting degenerate log
        axis made bars render as tiny slivers or inverted (hanging from
        the top). The explicit ~1-decade window plus fixed bar origin
        forces every panel to render bars consistently upward from the
        axis floor.
    (2) The molecule-name label is forced to the top-right (north east)
        on every panel, overriding `mol_label_node`'s Cytosine default of
        south-west. That default was pushing the Cytosine label to the
        bottom-left corner where it visually collided with the Uracil
        panel beneath it.
    """
    print("\n[Fig 11] Energy error vs CCSD reference …")
    NCOLS = 4
    nrows = (len(molecules) + NCOLS) // NCOLS

    blocks = []
    for panel_idx, mol in enumerate(molecules):
        col_idx    = panel_idx % NCOLS
        ylabel_opt = "" if col_idx == 0 else "  ylabel={},\n"
        refs       = cpu[mol]["refs"]
        E_ccsd     = refs.get("E_ccsd")
        pairs      = pair_runs(cpu[mol]["runs"], gpu[mol]["runs"])
        if not pairs or E_ccsd is None:
            continue

        pairs.sort(key=lambda p: (p["ne"], p["no"]))
        sym_coords = [sym(p["ne"], p["no"]) for p in pairs]
        sym_labels = [p["label"]            for p in pairs]
        sym_list   = ",".join(sym_coords)
        lbl_list   = ",".join(sym_labels)

        vc_err_pts, vg_err_pts, all_errs = [], [], []
        for p in pairs:
            dc = p["cpu"].get("d_ccsd") if p["cpu"].get("d_ccsd") is not None \
                 else (p["E_cpu"] - E_ccsd if p["E_cpu"] is not None else None)
            dg = p["gpu"].get("d_ccsd") if p["gpu"].get("d_ccsd") is not None \
                 else (p["E_gpu"] - E_ccsd if p["E_gpu"] is not None else None)
            sc = sym(p["ne"], p["no"])
            if dc is not None and abs(dc) > 1e-12:
                v = abs(dc)
                vc_err_pts.append(f"({sc},{v:.10f})")
                all_errs.append(v)
            if dg is not None and abs(dg) > 1e-12:
                v = abs(dg)
                vg_err_pts.append(f"({sc},{v:.10f})")
                all_errs.append(v)

        # ── FIX 1: explicit per-panel log window + fixed bar origin ──
        # ymin ~= 0.3 * min(data), ymax ~= 2.5 * max(data) gives every
        # panel a comparable ~1-decade log window regardless of how tight
        # the raw data is. log origin y=-infty forces bars to always
        # extend from the visible axis floor, independent of auto-range
        # heuristics, so bars render consistently in every panel.
        if all_errs:
            vmax = max(all_errs)
            vmin = min(all_errs)
            ymin_v = max(vmin * 0.95, 1e-8)
            ymax_v = vmax * 1.05  
            yrange_str = (
                f"  ymin={ymin_v:.10f}, ymax={ymax_v:.10f},\n"
                f"  log origin y=-infty,\n"
            )
        else:
            yrange_str = ""

        blocks.append(
            f"% === {mol} ===\n"
            f"\\nextgroupplot[\n  symbolic x coords={{{sym_list}}},\n"
            f"  xticklabels={{{lbl_list}}},\n"
            f"  xticklabel style={{font=\\tiny,xshift=4.5pt,yshift=-3pt}},\n"
            f"{yrange_str}"
            f"{ylabel_opt}]\n"
            + (f"\\addplot+[ybar, bar shift=-2pt, fill=blue!60, draw=blue!80]"
               f" coordinates {{{' '.join(vc_err_pts)}}};\n"
               if vc_err_pts else "")
            + (f"\\addplot+[ybar, bar shift=+2pt, fill=red!55,  draw=red!80]"
               f"  coordinates {{{' '.join(vg_err_pts)}}};\n"
               if vg_err_pts else "")
            # ── FIX 2: force uniform top-right label position on every
            # panel, overriding mol_label_node's Cytosine default.
            + mol_label_node(mol, display(mol),
                             anchor='north east', pos='0.98,0.98')
            + "\n"
        )

    if not blocks:
        print("  SKIP -- no molecules have E_ccsd reference.")
        return

    blocks.append(
        r"\nextgroupplot[hide axis, legend style={at={(0.5,0.5)}, anchor=center}]" + "\n"
        r"\addlegendimage{ybar, bar shift=-2pt, fill=blue!60, draw=blue!80}" + "\n"
        r"\addlegendentry{$|E_{\mathrm{VQE}}^{\mathrm{CPU}} - E_{\mathrm{CCSD}}|$ [Ha]}" + "\n"
        r"\addlegendimage{ybar, bar shift=+2pt, fill=red!55,  draw=red!80}" + "\n"
        r"\addlegendentry{$|E_{\mathrm{VQE}}^{\mathrm{GPU}} - E_{\mathrm{CCSD}}|$ [Ha]}" + "\n"
    )

    body = (
        HEADER
        + r"\begin{figure*}[htbp]" + "\n"
        + r"\centering\resizebox{1.1\textwidth}{!}{%" + "\n"
        + r"\begin{tikzpicture}" + "\n"
        + r"\begin{groupplot}[" + "\n"
        + f"  group style={{group size={NCOLS} by {nrows},"
        + r"horizontal sep=1.4cm,vertical sep=1cm}," + "\n"
        + r"  width=0.27\textwidth, height=0.34\textwidth," + "\n"
        + r"  ybar, bar width=4pt, xtick=data," + "\n"
        + r"  x tick label style={rotate=45, anchor=east}," + "\n"
        + r"  tick label style={font=\scriptsize}, label style={font=\scriptsize}," + "\n"
        + r"  ymajorgrids=true, grid style={dotted,gray!30}," + "\n"
        + r"  ymode=log," + "\n"
        + r"  ylabel={$|E_{\mathrm{VQE}} - E_{\mathrm{CCSD}}|$ [Ha]}," + "\n"
        + "]\n\n"
        + "\n".join(blocks)
        + r"\end{groupplot}" + "\n"
        + r"\end{tikzpicture}" + "\n"
        + r"}" + "\n"
        + r"\caption{Absolute VQE energy error relative to CCSD for each"
        + r" active-space configuration $(N_e^{(a)}, N_o^{(a)})$:"
        + r" $|E_{\mathrm{VQE}}^{\mathrm{CPU}} - E_{\mathrm{CCSD}}|$ (blue) and"
        + r" $|E_{\mathrm{VQE}}^{\mathrm{GPU}} - E_{\mathrm{CCSD}}|$ (red)."
        + r" Logarithmic scale. The HF error is 2--3 orders of magnitude larger"
        + r" (see Fig.~\ref{fig:energy_landscape}); both CPU and GPU VQE achieve"
        + r" near-identical accuracy.}" + "\n"
        + r"\label{fig:energy_error_vs_ccsd}" + "\n"
        + r"\end{figure*}" + "\n"
        + FOOTER
    )
    write_both(outdir, "fig11_energy_error_vs_ccsd",
               body.replace(HEADER, "").replace(HEADER_SUBCAP, "").replace(FOOTER, "").strip())
# ══════════════════════════════════════════════════════════════════
#  FIG 12 -- CORRELATION ENERGY RECOVERY  (% of CCSD correlation)
#           η = (E_VQE - E_HF) / (E_CCSD - E_HF) × 100 %
#           η = 0 % → pure HF,  η = 100 % → full CCSD correlation
# ══════════════════════════════════════════════════════════════════

def fig12_correlation_recovery(cpu, gpu, molecules, outdir):
    """
    Per molecule subplot: percentage of CCSD correlation energy recovered by VQE.
    Grouped bars per (Ne,No) config; CPU blue, GPU red.
    A dashed line at 100 % marks the CCSD target.
    """
    print("\n[Fig 12] Correlation energy recovery (%) …")
    NCOLS = 4
    nrows = (len(molecules) + NCOLS) // NCOLS

    blocks = []
    for mol in molecules:
        refs   = cpu[mol]["refs"]
        E_hf   = refs.get("E_hf")
        E_ccsd = refs.get("E_ccsd")
        pairs  = pair_runs(cpu[mol]["runs"], gpu[mol]["runs"])
        if not pairs or E_hf is None or E_ccsd is None:
            continue

        denom = E_ccsd - E_hf   # < 0
        if abs(denom) < 1e-10:
            continue

        pairs.sort(key=lambda p: (p["ne"], p["no"]))
        sym_coords = [sym(p["ne"], p["no"]) for p in pairs]
        sym_labels = [p["label"]             for p in pairs]
        sym_list   = ",".join(sym_coords)
        lbl_list   = ",".join(sym_labels)

        vc_pts, vg_pts = [], []
        for p in pairs:
            sc    = sym(p["ne"], p["no"])
            eta_c = corr_recovery(p["E_cpu"], E_hf, E_ccsd)
            eta_g = corr_recovery(p["E_gpu"], E_hf, E_ccsd)
            if eta_c is not None:
                vc_pts.append(f"({sc},{eta_c*100:.3f})")
            if eta_g is not None:
                vg_pts.append(f"({sc},{eta_g*100:.3f})")

        # 100 % reference line via \pgfplotsextra{\draw} -- avoids the
        # ybar handler which would render \addplot[mark=none] as a bar.
        first_sym = sym_coords[0]
        ref_line = (
            r"\pgfplotsextra{%" + "\n"
            f"  \\draw[green!60!black, dashed, line width=1.2pt]\n"
            f"    ({{rel axis cs:0,0}}|-{{axis cs:{{{first_sym}}},100}}) --\n"
            f"    ({{rel axis cs:1,0}}|-{{axis cs:{{{first_sym}}},100}});\n"
            r"}%" + "\n"
        )

        col_idx12 = len(blocks) % NCOLS
        ylabel_12 = "" if col_idx12 == 0 else "  ylabel={},\n"
        blocks.append(
            f"% === {mol} ===\n"
            f"\\nextgroupplot[\n  symbolic x coords={{{sym_list}}},\n"
            f"  xticklabels={{{lbl_list}}},\n"
            f"  xticklabel style={{font=\\tiny,xshift=4.5pt,yshift=-3pt}},\n"
            f"{ylabel_12}]\n"
            + (f"\\addplot+[ybar, bar shift=-2pt, fill=blue!60, draw=blue!80]"
               f" coordinates {{{' '.join(vc_pts)}}};\n"
               if vc_pts else "")
            + (f"\\addplot+[ybar, bar shift=+2pt, fill=red!55,  draw=red!80]"
               f"  coordinates {{{' '.join(vg_pts)}}};\n"
               if vg_pts else "")
            + ref_line
            + mol_label_node(mol, display(mol), anchor='center', pos='0.5,0.5') + "\n"
        )

    if not blocks:
        print("  SKIP -- no molecules have both E_hf and E_ccsd.")
        return

    blocks.append(
        r"\nextgroupplot[hide axis, legend style={at={(0.5,0.5)}, anchor=center}]" + "\n"
        r"\addlegendimage{ybar, bar shift=-2pt, fill=blue!60, draw=blue!80}" + "\n"
        r"\addlegendentry{$\eta_{\mathrm{CPU}}$ [\%]}" + "\n"
        r"\addlegendimage{ybar, bar shift=+2pt, fill=red!55,  draw=red!80}" + "\n"
        r"\addlegendentry{$\eta_{\mathrm{GPU}}$ [\%]}" + "\n"
        r"\addlegendimage{no marks, green!60!black, dashed, line width=1.2pt}" + "\n"
        r"\addlegendentry{100\% (CCSD target)}" + "\n"
    )

    body = (
        HEADER
        + r"\begin{figure*}[htbp]" + "\n"
        + r"\centering\resizebox{1.1\textwidth}{!}{%" + "\n"
        + r"\begin{tikzpicture}" + "\n"
        + r"\begin{groupplot}[" + "\n"
        + f"  group style={{group size={NCOLS} by {nrows},"
        + r"horizontal sep=1.4cm,vertical sep=1cm}," + "\n"
        + r"  width=0.27\textwidth, height=0.34\textwidth," + "\n"
        + r"  ybar, bar width=3pt, xtick=data," + "\n"
        + r"  x tick label style={rotate=45, anchor=east}," + "\n"
        + r"  tick label style={font=\scriptsize}, label style={font=\scriptsize}," + "\n"
        + r"  ymajorgrids=true, grid style={dotted,gray!30}," + "\n"
        + r"  ymin=0, ymax=110," + "\n"
        + r"  ylabel={Correlation recovery $\eta$ [\%]}," + "\n"
        + "]\n\n"
        + "\n".join(blocks)
        + r"\end{groupplot}" + "\n"
        + r"\end{tikzpicture}" + "\n"
        + r"}" + "\n"
        + r"\caption{Percentage of CCSD correlation energy recovered by VQE: "
        + r"$\eta = (E_{\mathrm{VQE}} - E_{\mathrm{HF}}) / "
        + r"(E_{\mathrm{CCSD}} - E_{\mathrm{HF}}) \times 100\,\%$. "
        + r"The dashed green line marks the 100\,\% CCSD target. "
        + r"CPU and GPU results are shown in blue and red, respectively.}" + "\n"
        + r"\label{fig:correlation_recovery}" + "\n"
        + r"\end{figure*}" + "\n"
        + FOOTER
    )
    write_both(outdir, "fig12_correlation_recovery", body.replace(HEADER, "").replace(HEADER_SUBCAP, "").replace(FOOTER, "").strip())


# ══════════════════════════════════════════════════════════════════
#  FIG 13 -- ENERGY LANDSCAPE: HF / CCSD reference lines + VQE scatter
#           One subplot per molecule; x = active orbitals N_o^(a)
#           Horizontal dashed lines for E_HF and E_CCSD (full system)
#           Scatter points for all VQE configs (CPU blue, GPU red)
# ══════════════════════════════════════════════════════════════════

def fig13_energy_landscape(cpu, gpu, molecules, outdir):
    """
    Energy landscape plot per molecule.
    - Thick grey dashed horizontal line: E_HF (full system)
    - Thick green dotted horizontal line: E_CCSD (full system)
    - Blue scatter + line: E_VQE_CPU vs N_o^(a)  (best per No)
    - Red scatter + dashed: E_VQE_GPU vs N_o^(a) (best per No)
    Visually shows VQE converging toward CCSD as active space grows.
    """
    print("\n[Fig 13] Energy landscape (HF / CCSD lines + VQE scatter) …")
    NCOLS = 4
    nrows = (len(molecules) + NCOLS) // NCOLS

    blocks = []
    for mol in molecules:
        refs   = cpu[mol]["refs"]
        E_hf   = refs.get("E_hf")
        E_ccsd = refs.get("E_ccsd")
        pairs  = pair_runs(cpu[mol]["runs"], gpu[mol]["runs"])
        if not pairs:
            continue

        # ALL configs vs the complete active space (Ne,No).
        pairs.sort(key=lambda p: (p["ne"], p["no"]))
        sym_coords = [sym(p["ne"], p["no"]) for p in pairs]
        sym_list   = ",".join(sym_coords)
        lbl_list   = ",".join(p["label"] for p in pairs)
        first_sym, last_sym = sym_coords[0], sym_coords[-1]
        cc = " ".join(f"({sym(p['ne'],p['no'])},{p['E_cpu']:.6f})"
                      for p in pairs if p["E_cpu"] is not None)
        gc = " ".join(f"({sym(p['ne'],p['no'])},{p['E_gpu']:.6f})"
                      for p in pairs if p["E_gpu"] is not None)

        # HF / CCSD horizontal reference lines span the symbolic x-range.
        hf_line   = ""
        ccsd_line = ""
        if E_hf is not None:
            hf_line = (
                f"\\addplot[mark=none, gray, dashed, thick, line width=1.2pt]"
                f" coordinates {{({first_sym},{E_hf:.6f}) ({last_sym},{E_hf:.6f})}};\n"
            )
        if E_ccsd is not None:
            ccsd_line = (
                f"\\addplot[mark=none, green!60!black, dotted, thick, line width=1.5pt]"
                f" coordinates {{({first_sym},{E_ccsd:.6f}) ({last_sym},{E_ccsd:.6f})}};\n"
            )

        col_idx13 = len(blocks) % NCOLS
        ylabel_13 = "ylabel={Energy [Ha]}, y dir=reverse" if col_idx13 == 0 else "ylabel={}, y dir=reverse"
        blocks.append(
            f"% === {mol} ===\n"
            f"\\nextgroupplot[\n  {ylabel_13},\n"
            f"  symbolic x coords={{{sym_list}}},\n"
            f"  xtick=data, xticklabels={{{lbl_list}}},\n"
            f"  x tick label style={{font=\\tiny, rotate=45, anchor=east}},\n"
            f"]\n"
            + hf_line
            + ccsd_line
            + f"\\addplot[blue, thick, mark=*, mark size=1.7pt] coordinates {{{cc}}};\n"
            + f"\\addplot[red, dashed, thick, mark=triangle, mark size=2.6pt,"
            f" mark options={{fill=none}}] coordinates {{{gc}}};\n"
            + mol_label_node(mol, display(mol), anchor='center', pos='0.5,0.5') + "\n"
        )

    if not blocks:
        print("  SKIP -- no data.")
        return

    blocks.append(
        r"\nextgroupplot[hide axis]" + "\n"
        r"\addlegendimage{mark=none, gray, dashed, thick, line width=1.2pt}" + "\n"
        r"\addlegendentry{$E_{\mathrm{HF}}$}" + "\n"
        r"\addlegendimage{mark=none, green!60!black, dotted, thick, line width=1.5pt}" + "\n"
        r"\addlegendentry{$E_{\mathrm{CCSD}}$}" + "\n"
        r"\addlegendimage{mark=*, blue, thick}" + "\n"
        r"\addlegendentry{$E_{\mathrm{VQE}}$ (CPU)}" + "\n"
        r"\addlegendimage{mark=triangle, red, dashed, thick, mark options={fill=none}}" + "\n"
        r"\addlegendentry{$E_{\mathrm{VQE}}$ (GPU)}" + "\n"
    )

    body = (
        HEADER
        + r"\begin{figure*}[htbp]" + "\n"
        + r"\centering\resizebox{1.1\textwidth}{!}{%" + "\n"
        + r"\begin{tikzpicture}" + "\n"
        + r"\begin{groupplot}[" + "\n"
        + f"  group style={{group size={NCOLS} by {nrows},"
        + r"horizontal sep=1.5cm,vertical sep=1.5cm}," + "\n"
        + r"  width=0.27\textwidth, height=0.34\textwidth," + "\n"
        + r"  tick label style={font=\scriptsize}, label style={font=\scriptsize}," + "\n"
        + r"  legend style={font=\scriptsize, at={(0.5,-0.28)},"
        + r"anchor=north, legend columns=2}," + "\n"
        + "]\n\n"
        + "\n".join(blocks)
        + r"\end{groupplot}" + "\n"
        + r"\end{tikzpicture}" + "\n"
        + r"}" + "\n"
        + r"\caption{Energy landscape per molecule: $E_{\mathrm{HF}}$ (dashed grey),"
        + r" $E_{\mathrm{CCSD}}$ (dotted green), and $E_{\mathrm{VQE}}$ on CPU (blue)"
        + r" and GPU (red) across all active-space configurations"
        + r" $(N_e^{(a)},N_o^{(a)})$. VQE energies"
        + r" approach the CCSD target as the active space grows, with CPU and GPU"
        + r" results in close agreement.}" + "\n"
        + r"\label{fig:energy_landscape}" + "\n"
        + r"\end{figure*}" + "\n"
        + FOOTER
    )
    write_both(outdir, "fig13_energy_landscape", body.replace(HEADER, "").replace(HEADER_SUBCAP, "").replace(FOOTER, "").strip())


# ══════════════════════════════════════════════════════════════════
#  FIG 14 -- SIGNED Δ E = E_VQE - E_CCSD  per (Ne,No) config
#           Positive → VQE above CCSD (typical: variational principle)
#           Near-zero → VQE has converged to CCSD quality
#           Also draws |E_HF - E_CCSD| as a grey reference bar
# ══════════════════════════════════════════════════════════════════

def fig14_vqe_ccsd_delta(cpu, gpu, molecules, outdir):
    """
    Per molecule subplot: signed ΔE = E_VQE - E_CCSD in milli-Hartree.
    CPU blue bars, GPU red bars.
    A dashed orange line marks chemical accuracy (1.6 mHa ~ 1 kcal/mol).
    HF is intentionally excluded -- its gap (100x-1000x mHa) is on a
    completely different scale and would compress the VQE bars to zero.
    The HF gap is shown as context in fig13 (energy landscape).
    Uses d_ccsd field where available; falls back to direct subtraction.
    """
    print("\n[Fig 14] Signed ΔE = E_VQE - E_CCSD (milli-Hartree) …")
    NCOLS = 4
    nrows = (len(molecules) + NCOLS) // NCOLS
    CHEM_ACC = 1.6   # mHa  ~ 1 kcal/mol (chemical accuracy threshold)

    blocks = []
    for mol in molecules:
        refs   = cpu[mol]["refs"]
        E_ccsd = refs.get("E_ccsd")
        pairs  = pair_runs(cpu[mol]["runs"], gpu[mol]["runs"])
        if not pairs or E_ccsd is None:
            continue

        pairs.sort(key=lambda p: (p["ne"], p["no"]))
        sym_coords = [sym(p["ne"], p["no"]) for p in pairs]
        sym_labels = [p["label"]             for p in pairs]
        sym_list   = ",".join(sym_coords)
        lbl_list   = ",".join(sym_labels)

        vc_pts, vg_pts = [], []
        for p in pairs:
            sc = sym(p["ne"], p["no"])
            dc = p["cpu"].get("d_ccsd")
            if dc is None and p["E_cpu"] is not None:
                dc = p["E_cpu"] - E_ccsd
            dg = p["gpu"].get("d_ccsd")
            if dg is None and p["E_gpu"] is not None:
                dg = p["E_gpu"] - E_ccsd
            if dc is not None:
                vc_pts.append(f"({sc},{dc*1000:.4f})")   # Ha → mHa
            if dg is not None:
                vg_pts.append(f"({sc},{dg*1000:.4f})")

        # Chemical accuracy threshold as a horizontal line via pgfplotsextra
        # (avoids the ybar handler rendering it as a bar)
        first_sym = sym_coords[0]
        chem_line = (
            r"\pgfplotsextra{%" + "\n"
            f"  \\draw[orange!80!black, dashed, line width=1pt]\n"
            f"    ({{rel axis cs:0,0}}|-{{axis cs:{{{first_sym}}},{CHEM_ACC:.2f}}}) --\n"
            f"    ({{rel axis cs:1,0}}|-{{axis cs:{{{first_sym}}},{CHEM_ACC:.2f}}});\n"
            r"}%" + "\n"
        )

        col_idx14 = len(blocks) % NCOLS
        ylabel_14 = "" if col_idx14 == 0 else "  ylabel={},\n"
        blocks.append(
            f"% === {mol} ===\n"
            f"\\nextgroupplot[\n  symbolic x coords={{{sym_list}}},\n"
            f"  xticklabels={{{lbl_list}}},\n"
            f"  xticklabel style={{font=\\tiny,xshift=4.5pt,yshift=-3pt}},\n"
            f"{ylabel_14}]\n"
            + (f"\\addplot+[ybar, bar shift=-2pt, fill=blue!60, draw=blue!80]"
               f" coordinates {{{' '.join(vc_pts)}}};\n"
               if vc_pts else "")
            + (f"\\addplot+[ybar, bar shift=+2pt, fill=red!55,  draw=red!80]"
               f"  coordinates {{{' '.join(vg_pts)}}};\n"
               if vg_pts else "")
            + chem_line
            + mol_label_node(mol, display(mol), anchor='center', pos='0.5,0.5') + "\n"
        )

    if not blocks:
        print("  SKIP -- no E_ccsd reference found.")
        return

    blocks.append(
        r"\nextgroupplot[hide axis, legend style={at={(0.5,0.5)}, anchor=center}]" + "\n"
        r"\addlegendimage{ybar, bar shift=-2pt, fill=blue!60, draw=blue!80}" + "\n"
        r"\addlegendentry{$\Delta E_{\mathrm{CPU}}$ [mHa]}" + "\n"
        r"\addlegendimage{ybar, bar shift=+2pt, fill=red!55,  draw=red!80}" + "\n"
        r"\addlegendentry{$\Delta E_{\mathrm{GPU}}$ [mHa]}" + "\n"
        r"\addlegendimage{no marks, orange!80!black, dashed, line width=1pt}" + "\n"
        r"\addlegendentry{Chem.\ accuracy (1.6\,mHa)}" + "\n"
    )

    body = (
        HEADER
        + r"\begin{figure*}[htbp]" + "\n"
        + r"\centering\resizebox{1.1\textwidth}{!}{%" + "\n"
        + r"\begin{tikzpicture}" + "\n"
        + r"\begin{groupplot}[" + "\n"
        + f"  group style={{group size={NCOLS} by {nrows},"
        + r"horizontal sep=1.4cm,vertical sep=1cm}," + "\n"
        + r"  width=0.27\textwidth, height=0.34\textwidth," + "\n"
        + r"  ybar, bar width=4pt, xtick=data," + "\n"
        + r"  x tick label style={rotate=45, anchor=east}," + "\n"
        + r"  tick label style={font=\scriptsize}, label style={font=\scriptsize}," + "\n"
        + r"  ymajorgrids=true, grid style={dotted,gray!30}," + "\n"
        + r"  ymin=0," + "\n"
        + r"  ylabel={$\Delta E = E_{\mathrm{VQE}} - E_{\mathrm{CCSD}}$ [mHa]}," + "\n"
        + "]\n\n"
        + "\n".join(blocks)
        + r"\end{groupplot}" + "\n"
        + r"\end{tikzpicture}" + "\n"
        + r"}" + "\n"
        + r"\caption{Signed energy deviation $\Delta E = E_{\mathrm{VQE}} - E_{\mathrm{CCSD}}$"
        + r" [mHa] for CPU (blue) and GPU (red) VQE per active-space configuration."
        + r" The dashed orange line marks chemical accuracy ($1.6\,\mathrm{mHa}"
        + r" \approx 1\,\mathrm{kcal/mol}$). Configurations below this line achieve"
        + r" chemical accuracy relative to CCSD. HF errors (100--1000$\times$ larger)"
        + r" are shown in Fig.~\ref{fig:energy_landscape}.}" + "\n"
        + r"\label{fig:vqe_ccsd_delta}" + "\n"
        + r"\end{figure*}" + "\n"
        + FOOTER
    )
    write_both(outdir, "fig14_vqe_ccsd_delta", body.replace(HEADER, "").replace(HEADER_SUBCAP, "").replace(FOOTER, "").strip())


# ══════════════════════════════════════════════════════════════════
#  CSV EXPORT  (extended with energy accuracy columns)
# ══════════════════════════════════════════════════════════════════

def export_csv(cpu, gpu, molecules, outdir):
    import csv
    fpath   = os.path.join(outdir, "vqe_benchmark_summary.csv")
    headers = [
        "Molecule", "Ne", "No", "Qubits", "NParams",
        # Energies
        "E_HF", "E_CCSD",
        "E_VQE_CPU", "E_VQE_GPU", "dE_GPU_minus_CPU",
        # VQE accuracy vs references
        "dE_VQE_CPU_minus_HF", "dE_VQE_GPU_minus_HF",
        "dE_VQE_CPU_minus_CCSD", "dE_VQE_GPU_minus_CCSD",
        "CorrRecovery_CPU_pct", "CorrRecovery_GPU_pct",
        # Runtime & speedup
        "Runtime_CPU_s", "Runtime_GPU_s", "Speedup",
        # Convergence
        "Converged_CPU", "Converged_GPU",
        "Cycles_CPU", "Cycles_GPU", "NFev_CPU", "NFev_GPU",
        "HamTerms",
    ]
    with open(fpath, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        for mol in molecules:
            refs   = cpu[mol]["refs"]
            E_hf   = refs.get("E_hf")
            E_ccsd = refs.get("E_ccsd")
            for p in sorted(pair_runs(cpu[mol]["runs"], gpu[mol]["runs"]),
                            key=lambda x: (x["ne"], x["no"])):
                cr, gr = p["cpu"], p["gpu"]
                dE     = (p["E_gpu"] - p["E_cpu"]
                          if None not in (p["E_cpu"], p["E_gpu"]) else "")

                def _fmt(v, d=8): return f"{v:.{d}f}" if v is not None else ""
                def _fmte(v): return f"{v:.6e}" if v is not None else ""

                # deltas vs HF
                dc_hf = (p["E_cpu"] - E_hf   if None not in (p["E_cpu"], E_hf)   else None)
                dg_hf = (p["E_gpu"] - E_hf   if None not in (p["E_gpu"], E_hf)   else None)
                # deltas vs CCSD (prefer stored value)
                dc_cc = cr.get("d_ccsd") or (p["E_cpu"] - E_ccsd if None not in (p["E_cpu"], E_ccsd) else None)
                dg_cc = gr.get("d_ccsd") or (p["E_gpu"] - E_ccsd if None not in (p["E_gpu"], E_ccsd) else None)
                # correlation recovery
                eta_c = corr_recovery(p["E_cpu"], E_hf, E_ccsd)
                eta_g = corr_recovery(p["E_gpu"], E_hf, E_ccsd)

                w.writerow([
                    mol, p["ne"], p["no"], p["qubits"], cr["n_params"],
                    _fmt(E_hf),  _fmt(E_ccsd),
                    _fmt(p["E_cpu"]), _fmt(p["E_gpu"]),
                    _fmte(dE) if dE != "" else "",
                    _fmte(dc_hf), _fmte(dg_hf),
                    _fmte(dc_cc), _fmte(dg_cc),
                    f"{eta_c*100:.3f}" if eta_c is not None else "",
                    f"{eta_g*100:.3f}" if eta_g is not None else "",
                    _fmt(p["rt_cpu"], 4), _fmt(p["rt_gpu"], 4),
                    f"{p['speedup']:.3f}" if p["speedup"] else "",
                    cr["converged"], gr["converged"],
                    cr["cycles"], gr["cycles"],
                    cr["nfev"],   gr["nfev"],
                    cr["ham_terms"],
                ])
    print(f"\n  → CSV: {fpath}")


# ══════════════════════════════════════════════════════════════════
#  PRINT SUMMARY TABLE  (terminal)
# ══════════════════════════════════════════════════════════════════

def print_summary(cpu, gpu, molecules):
    hdr = (f"{'Molecule':20s}  {'(Ne,No)':9s}  {'Q':4s}"
           f"  {'E_CPU':>13s}  {'E_GPU':>13s}"
           f"  {'E_HF':>13s}  {'E_CCSD':>13s}"
           f"  {'η_CPU%':>7s}  {'ΔE_CC(mHa)':>11s}"
           f"  {'rt_CPU(s)':>10s}  {'rt_GPU(s)':>10s}  {'Speedup':>8s}")
    print("\n" + "─" * len(hdr))
    print(hdr)
    print("─" * len(hdr))
    for mol in molecules:
        refs   = cpu[mol]["refs"]
        E_hf   = refs.get("E_hf")
        E_ccsd = refs.get("E_ccsd")
        for p in sorted(pair_runs(cpu[mol]["runs"], gpu[mol]["runs"]),
                        key=lambda x: (x["ne"], x["no"])):
            cr    = p["cpu"]
            sp    = f"{p['speedup']:7.2f}×" if p["speedup"] else "    N/A "
            rcp   = f"{p['rt_cpu']:10.2f}"  if p["rt_cpu"] else "       N/A"
            rgp   = f"{p['rt_gpu']:10.2f}"  if p["rt_gpu"] else "       N/A"
            eta   = corr_recovery(p["E_cpu"], E_hf, E_ccsd)
            eta_s = f"{eta*100:7.2f}" if eta is not None else "    N/A"
            dc_cc = cr.get("d_ccsd") or (p["E_cpu"] - E_ccsd if None not in (p["E_cpu"], E_ccsd) else None)
            dc_s  = f"{dc_cc*1000:11.4f}" if dc_cc is not None else "        N/A"
            E_hf_s   = f"{E_hf:13.6f}"   if E_hf   else "          N/A"
            E_ccsd_s = f"{E_ccsd:13.6f}" if E_ccsd else "          N/A"
            print(f"{mol:20s}  {p['label']:9s}  {p['qubits']:4d}"
                  f"  {p['E_cpu']:13.6f}  {p['E_gpu']:13.6f}"
                  f"  {E_hf_s}  {E_ccsd_s}"
                  f"  {eta_s}  {dc_s}"
                  f"  {rcp}  {rgp}  {sp}")
    print("─" * len(hdr))


# ══════════════════════════════════════════════════════════════════
#  FIG 15 -- PERFORMANCE SUMMARY  (3-panel subfigure, template style)
#
#  Panel (a) top:   GPU speedup bar chart  per molecule (best config)
#  Panel (b) bot-L: |E_VQE_CPU - E_CCSD|  line plot    per molecule
#  Panel (c) bot-R: Normalized speedup + normalized |ΔE_CCSD| (grouped bar)
#
#  Mirrors exactly the template layout:
#    \begin{subfigure}[t]{0.9\textwidth} …speedup bar…
#    \begin{subfigure}[t]{0.45\textwidth}…energy dev line…
#    \begin{subfigure}[t]{0.50\textwidth}…normalized bar…
# ══════════════════════════════════════════════════════════════════

HEADER_SUBCAP = r"""\documentclass[border=4pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{pgfplotstable}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{subcaption}
\usepackage{caption}
\pgfplotsset{
  every axis/.append style={
    tick label style={font=\scriptsize},
    label style={font=\scriptsize},
    legend style={font=\scriptsize},
    grid=both, grid style={dotted,gray!30},
  }
}
\begin{document}
"""

# ══════════════════════════════════════════════════════════════════
#  FIG 15 -- PERFORMANCE SUMMARY  (4-panel subfigure, template style)
#
#  Panel (a) top:   GPU speedup DISTRIBUTION per molecule
#                   (median + min/max whiskers across all configs)
#  Panel (b) mid-L: |ΔE(GPU−CPU)| DISTRIBUTION across all configs
#                   (median marker + min/max whiskers per molecule)
#  Panel (c) mid-R: MAX |VQE−HF| per molecule (best-correlation config)
#  Panel (d) bot-L: MIN |VQE−CCSD| per molecule (best-agreement config)
#  Panel (e) bot-R: Normalised multi-metric bar (preferred config)
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
#  FIG 15 -- PERFORMANCE SUMMARY  (3-panel subfigure)
#
#  Panel (a) top:   GPU speedup DISTRIBUTION per molecule
#                   (median + min/max whiskers across all configs)
#  Panel (b) bot-L: |ΔE(GPU−CPU)| DISTRIBUTION across all configs
#                   (median marker + min/max whiskers per molecule)
#  Panel (c) bot-R: MAX |VQE−HF| per molecule (best-correlation config)
#
#  NOTE: panels (d) [MIN |VQE−CCSD|] and (e) [normalised multi-metric]
#  moved to fig22_accuracy_summary so each figure tells a focused story:
#    Fig 15 -- GPU-acceleration story (speedup + CPU/GPU agreement +
#              correlation captured over HF)
#    Fig 22 -- Accuracy / gold-standard comparison story (CCSD gap +
#              normalised multi-metric view)
# ══════════════════════════════════════════════════════════════════

def fig15_performance_summary(cpu, gpu, molecules, outdir):
    """
    Three-panel GPU-performance summary figure.

    Panel (a) -- top full-width:
        GPU speedup DISTRIBUTION (median square + min/max whiskers)
        of t_CPU/t_GPU across ALL paired active-space configs per
        molecule. Dashed orange reference line at 10× marks the
        commonly cited HPC-porting-worthwhile threshold.

    Panel (b) -- bottom-left:
        |E_VQE_GPU - E_VQE_CPU| [mHa] DISTRIBUTION across ALL paired
        configs per molecule. Median marker + min/max whiskers.
        Dashed red reference at 1.6 mHa = chemical accuracy.

    Panel (c) -- bottom-right:
        MAX VQE-HF improvement per molecule: for each molecule,
        the active-space config that captures the most correlation
        energy over HF. Positive y-axis (taller = more correlation).
    """
    print("\n[Fig 15] GPU performance summary -- 3-panel subfigure …")

    # ── Data collection ────────────────────────────────────────────
    mol_labels     = []
    # panel (a): full distribution of speedup across all configs
    sp_all         = []
    sp_as          = []   # parallel list of (ne,no) tuples for coloring
    sp_med         = []
    sp_min         = []
    sp_max         = []
    # panel (b): full distribution of |ΔE_GPU-CPU| across all configs
    dE_gpu_cpu_all = []
    dE_as          = []   # parallel list of (ne,no) tuples for coloring
    dE_med         = []
    dE_min         = []
    dE_max         = []
    # panel (d): full distribution of relative VQE improvement over HF
    #            rel = (E_VQE_GPU - E_HF) / E_VQE_GPU   per config
    relhf_all      = []
    relhf_as       = []   # parallel list of (ne,no) tuples for coloring
    relhf_med      = []
    relhf_min      = []
    relhf_max      = []
    # panel (c): best-HF-config values per molecule
    dE_vc_hf       = []
    dE_vg_hf       = []

    for mol in molecules:
        refs   = cpu[mol]["refs"]
        E_hf   = refs.get("E_hf")
        pairs  = pair_runs(cpu[mol]["runs"], gpu[mol]["runs"])
        if not pairs:
            continue

        # Use best-CPU config to validate we have complete data
        best = min(pairs, key=lambda p: p["E_cpu"] if p["E_cpu"] is not None else 0)
        if best["speedup"] is None or best["E_cpu"] is None or best["E_gpu"] is None:
            continue

        # ── panel (a): full speedup distribution across configs ──
        sp_pairs = [(p["ne"], p["no"], p["speedup"])
                    for p in pairs if p["speedup"] is not None]
        sp_list  = [v for _, _, v in sp_pairs]
        if not sp_list:
            continue

        # ── panel (b): |ΔE(GPU-CPU)| distribution over ALL configs ──
        dE_pairs = [
            (p["ne"], p["no"], abs(p["E_gpu"] - p["E_cpu"]) * 1000.0)
            for p in pairs
            if p["E_cpu"] is not None and p["E_gpu"] is not None
        ]
        abs_gaps = [v for _, _, v in dE_pairs]
        if not abs_gaps:
            continue

        mol_labels.append(display(mol))

        # Panel (a) distribution stats + active-space labels for coloring
        sp_all.append(sp_list)
        sp_as.append([(ne, no) for ne, no, _ in sp_pairs])
        sp_med.append(float(np.median(sp_list)))
        sp_min.append(min(sp_list))
        sp_max.append(max(sp_list))

        # Panel (b) distribution stats + active-space labels for coloring
        dE_gpu_cpu_all.append(abs_gaps)
        dE_as.append([(ne, no) for ne, no, _ in dE_pairs])
        dE_med.append(float(np.median(abs_gaps)))
        dE_min.append(min(abs_gaps))
        dE_max.append(max(abs_gaps))

        # ── panel (d): relative VQE improvement over HF (GPU energies) ──
        #   rel = (E_VQE_GPU - E_HF) / E_VQE_GPU  per configuration.
        #   Both numerator and denominator are negative, so rel > 0:
        #   it is the correlation energy captured as a fraction of the
        #   total VQE energy. Dimensionless; one value per config.
        relhf_pairs = []
        if E_hf is not None:
            for p in pairs:
                eg = p["E_gpu"]
                if eg is not None and abs(eg) > 1e-12:
                    relhf_pairs.append((p["ne"], p["no"], (eg - E_hf) / eg))
        if not relhf_pairs:
            relhf_pairs = [(0, 0, 0.0)]
        rel_list = [v for _, _, v in relhf_pairs]
        relhf_all.append(rel_list)
        relhf_as.append([(ne, no) for ne, no, _ in relhf_pairs])
        relhf_med.append(float(np.median(rel_list)))
        relhf_min.append(min(rel_list))
        relhf_max.append(max(rel_list))

        # ── panel (c): BEST HF-improvement config per molecule ──
        # config minimising (E_VQE_CPU - E_HF), i.e. most negative value
        # = largest correlation captured. GPU value taken from SAME config.
        if E_hf is not None:
            best_hf = min(pairs, key=lambda p: (p["E_cpu"] - E_hf))
            dE_vc_hf.append(abs(best_hf["E_cpu"] - E_hf) * 1000)
            dE_vg_hf.append(abs(best_hf["E_gpu"] - E_hf) * 1000)
        else:
            dE_vc_hf.append(0.0)
            dE_vg_hf.append(0.0)

    if not mol_labels:
        print("  SKIP -- no complete data for any molecule.")
        return

    # ── Active-space color map: standard red→violet spectral scale ──
    # Smallest active space (2,3) = red, largest (6,7) = violet, sweeping
    # the visible spectrum (red→orange→yellow→green→blue→violet) so the
    # ordering by active-space size is read directly from hue -- no legend
    # needed. Colors are fixed across all molecules and panels via the
    # canonical list of 9 (Ne, No) pairs; non-standard configs (e.g. (2,2)
    # or (4,6) for Methylene/Pentacene) are silently skipped.
    all_as_configs = CANONICAL_AS
    n_as = max(len(all_as_configs), 1)

    def _as_rgb(idx, n):
        # hue 0 (red) → 0.80 (~288°, violet); high saturation for print
        t   = idx / max(n - 1, 1)
        hue = 0.80 * t
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.90)
        return (round(r, 3), round(g, 3), round(b, 3))

    as_color_map  = {}
    as_color_defs = ""
    for i, cfg in enumerate(all_as_configs):
        cname = f"ascolor{i}"
        r, g, b = _as_rgb(i, n_as)
        as_color_defs += f"\\definecolor{{{cname}}}{{rgb}}{{{r},{g},{b}}}\n"
        as_color_map[cfg] = cname

    sym_list = ",".join(f"{{{m}}}" for m in mol_labels)

    def pts(vals):
        return " ".join(f"({{{m}}},{v:.5f})" for m, v in zip(mol_labels, vals))

    def colored_dot_addplots(labels, as_lists, val_lists, fmt=".5f"):
        """One \\addplot per active-space config, colored red→violet.

        No \\addlegendentry: the red→violet hue itself encodes active-space
        size (explained in the caption), so the detailed legend is dropped.
        """
        by_as = defaultdict(list)
        for m, mol_as, mol_vals in zip(labels, as_lists, val_lists):
            for cfg, v in zip(mol_as, mol_vals):
                by_as[cfg].append(f"({{{m}}},{v:{fmt}})")
        latex = ""
        for cfg in all_as_configs:
            if cfg not in by_as:
                continue
            cname  = as_color_map[cfg]
            coords = " ".join(by_as[cfg])
            latex += (
                f"\\addplot[only marks, mark=*, mark size=1.7pt,"
                f" color={cname}, fill={cname}, fill opacity=0.70,"
                f" draw opacity=0.90, forget plot, error bars/y dir=none]"
                f" coordinates {{{coords}}};\n"
            )
        return latex

    tick_opts = (
        r"  xtick=data," + "\n"
        r"  x tick label style={rotate=45, anchor=east, font=\scriptsize}," + "\n"
        r"  ymajorgrids=true, grid style={dotted,gray!30}," + "\n"
    )

    # ── Panel (a): SPEEDUP DISTRIBUTION ────────────────────────────
    ymax_a = max(sp_max) * 1.15
    ymin_a = -1.2

    # Whisker spans the full min->max range. It is anchored at the median
    # purely so the +error/-error reach max/min; the central marker is
    # HIDDEN (mark=none) -- we no longer draw a median square.
    sp_whisker_coords = " ".join(
        f"({{{m}}},{med:.5f}) += (0,{(mx-med):.5f}) -= (0,{(med-mn):.5f})"
        for m, med, mn, mx in zip(mol_labels, sp_med, sp_min, sp_max)
    )
    panel_a = (
        r"\begin{subfigure}[t]{0.48\textwidth}" + "\n"
        r"\centering\begin{tikzpicture}" + "\n"
        r"\begin{axis}[" + "\n"
        r"  width=\textwidth, height=0.70\textwidth," + "\n"
        r"  ylabel={GPU speedup ($t_{\mathrm{CPU}}/t_{\mathrm{GPU}}$)}," + "\n"
        f"  symbolic x coords={{{sym_list}}}," + "\n"
        f"  xtick={{{sym_list}}}," + "\n"
        r"  enlarge x limits=0.06," + "\n"
        + tick_opts
        + f"  ymin={ymin_a}, ymax={ymax_a:.2f}," + "\n"
        r"  error bars/y dir=both," + "\n"
        r"  error bars/y explicit," + "\n"
        r"  error bars/error bar style={line width=0.9pt, blue!70!black}," + "\n"
        r"  error bars/error mark options={" + "\n"
        r"    line width=0.9pt, mark size=4pt, rotate=90, blue!70!black}," + "\n"
        r"  legend style={at={(0.5,-0.52)}, anchor=north, legend columns=4, font=\tiny}," + "\n"
        r"]" + "\n"
        # Whisker (min->max range), central marker hidden.
        # forget plot keeps the whisker out of the legend counter.
        f"\\addplot+[only marks, mark=none, blue!70!black, forget plot]"
        f" plot coordinates {{{sp_whisker_coords}}};\n"
        # Dots colored by (Ne, No) active-space configuration (red→green).
        + colored_dot_addplots(mol_labels, sp_as, sp_all)
        + r"\end{axis}\end{tikzpicture}" + "\n"
        r"\caption{GPU speedup distribution across all active-space configurations:"
        r" each marker is one $(N_e^{(a)},N_o^{(a)})$ configuration (color-coded"
        r" red $\to$ violet by active-space size); the whisker spans the min/max"
        r" of $t_{\mathrm{CPU}}/t_{\mathrm{GPU}}$ per molecule.}" + "\n"
        r"\end{subfigure}\hfill" + "\n"
    )

    # ── Panel (b): |ΔE(GPU-CPU)| DISTRIBUTION (Methylene excluded) ──
    # Methylene's GPU-CPU spread is far larger than any other molecule and
    # would compress everything else onto the axis floor, so it is dropped
    # from this panel; its range is reported in the caption instead.
    meth_lbl = display("Methylene")
    keep_b   = [i for i, m in enumerate(mol_labels) if m != meth_lbl]
    mol_labels_b = [mol_labels[i]     for i in keep_b]
    dE_as_b      = [dE_as[i]          for i in keep_b]
    dE_all_b     = [dE_gpu_cpu_all[i] for i in keep_b]
    dE_med_b     = [dE_med[i]         for i in keep_b]
    dE_min_b     = [dE_min[i]         for i in keep_b]
    dE_max_b     = [dE_max[i]         for i in keep_b]
    sym_list_b   = ",".join(f"{{{m}}}" for m in mol_labels_b)

    if meth_lbl in mol_labels:
        _mi = mol_labels.index(meth_lbl)
        meth_note = (
            f" Methylene is omitted here for scale: its "
            f"$|\\Delta E_{{\\mathrm{{GPU-CPU}}}}|$ ranges "
            f"{dE_min[_mi]:.2f}--{dE_max[_mi]:.2f}\\,mHa "
            f"(median {dE_med[_mi]:.2f}\\,mHa)."
        )
    else:
        meth_note = ""

    ymax_b = max(max(dE_max_b) * 1.25, 2.0) if dE_max_b else 2.0
    ymin_b = -0.08

    # Whisker spans min->max (central marker hidden); dots = every config.
    dE_whisker_coords = " ".join(
        f"({{{m}}},{med:.5f}) += (0,{(mx-med):.5f}) -= (0,{(med-mn):.5f})"
        for m, med, mn, mx in zip(mol_labels_b, dE_med_b, dE_min_b, dE_max_b)
    )
    panel_b = (
        r"\begin{subfigure}[t]{0.48\textwidth}" + "\n"
        r"\centering\begin{tikzpicture}" + "\n"
        r"\begin{axis}[" + "\n"
        r"  width=\textwidth, height=0.70\textwidth," + "\n"
        r"  ylabel={$|E_{\mathrm{VQE}}^{\mathrm{GPU}} - E_{\mathrm{VQE}}^{\mathrm{CPU}}|$ [mHa]}," + "\n"
        f"  symbolic x coords={{{sym_list_b}}}," + "\n"
        r"  enlarge x limits=0.06," + "\n"
        + tick_opts
        + f"  ymin={ymin_b}, ymax={ymax_b:.4f}," + "\n"
        r"  error bars/y dir=both," + "\n"
        r"  error bars/y explicit," + "\n"
        r"  error bars/error bar style={line width=0.8pt, orange!80!black}," + "\n"
        r"  error bars/error mark options={" + "\n"
        r"    line width=0.8pt, mark size=3pt, rotate=90, orange!80!black}," + "\n"
        r"]" + "\n"
        # Whisker (min->max range), central marker hidden.
        f"\\addplot+[only marks, mark=none, orange!80!black, forget plot]"
        f" plot coordinates {{{dE_whisker_coords}}};\n"
        # Dots colored by (Ne, No) active-space configuration (red->violet).
        + colored_dot_addplots(mol_labels_b, dE_as_b, dE_all_b)
        + r"\addplot[thick, red, dashed, line width=1.1pt, mark=none]"
        + f" coordinates {{({{{mol_labels_b[0]}}},1.6) ({{{mol_labels_b[-1]}}},1.6)}};\n"
        r"\node[anchor=east, font=\tiny, red!80!black, fill=white,"
        r" inner sep=1pt] at (rel axis cs:0.99,0.88)"
        + " {chem.\\ acc.\\ 1.6\\,mHa};\n"
        r"\addplot[gray!50, thin, mark=none]"
        + f" coordinates {{({{{mol_labels_b[0]}}},0) ({{{mol_labels_b[-1]}}},0)}};\n"
        r"\end{axis}\end{tikzpicture}" + "\n"
        r"\caption{CPU--GPU agreement distribution: each marker is one" + "\n"
        r"$(N_e^{(a)},N_o^{(a)})$ configuration; the whisker spans the min/max" + "\n"
        r"of $|\Delta E_{\mathrm{GPU-CPU}}|$ across all" + "\n"
        r"paired active-space configs per molecule. All remaining configurations" + "\n"
        r"lie below the red dashed chemical-accuracy threshold (1.6\,mHa)," + "\n"
        r"confirming GPU introduces no chemically meaningful numerical error."
        + meth_note + "}" + "\n"
        r"\end{subfigure}" + "\n"
    )

    # ── Panel (c): MAX VQE-HF improvement per molecule ─────────────
    panel_c = (
        r"\begin{subfigure}[t]{0.48\textwidth}" + "\n"
        r"\centering\begin{tikzpicture}" + "\n"
        r"\begin{axis}[" + "\n"
        r"  ybar, bar width=5pt," + "\n"
        r"  enlarge x limits=0.06," + "\n"
        r"  width=\textwidth, height=0.70\textwidth," + "\n"
        r"  ylabel={$|E_{\mathrm{VQE}} - E_{\mathrm{HF}}|$ [mHa]}," + "\n"
        r"  ymin=0," + "\n"
        f"  symbolic x coords={{{sym_list}}}," + "\n"
        + tick_opts
        + r"  legend style={at={(0.5,-0.52)}, anchor=north, legend columns=2,"
        r"font=\scriptsize}," + "\n"
        r"]" + "\n"
        r"\addplot[fill=blue!60, draw=blue!80] coordinates {" + pts(dE_vc_hf) + "};\n"
        r"\addplot[fill=red!55,  draw=red!80]  coordinates {" + pts(dE_vg_hf) + "};\n"
        r"\legend{$|E_{\mathrm{VQE}}^{\mathrm{CPU}}-E_{\mathrm{HF}}|$,"
        r"$|E_{\mathrm{VQE}}^{\mathrm{GPU}}-E_{\mathrm{HF}}|$}" + "\n"
        r"\end{axis}\end{tikzpicture}" + "\n"
        r"\caption{Maximum VQE improvement over Hartree--Fock per molecule," + "\n"
        r"$|E_{\mathrm{VQE}}-E_{\mathrm{HF}}|$ [mHa] (taller = more correlation" + "\n"
        r"captured). Each molecule's bar uses the active-space configuration" + "\n"
        r"that captures the most correlation energy; see" + "\n"
        r"Fig.~\ref{fig:correlation_recovery} for the full distribution across" + "\n"
        r"all configurations.}" + "\n"
        r"\end{subfigure}\hfill" + "\n"
    )

    # ── Panel (d): RELATIVE VQE improvement over HF (GPU energies) ──
    # Same strip-plot style as panel (b): one dot per configuration plus
    # a min/max whisker. y = (E_VQE_GPU - E_HF) / E_VQE_GPU, dimensionless.
    # Scale by 1e4: raw values are ~1e-4, this keeps y-axis ticks at 0-6.
    SCALE_D = 1e4
    relhf_all_s = [[v * SCALE_D for v in mol] for mol in relhf_all]
    relhf_med_s = [v * SCALE_D for v in relhf_med]
    relhf_min_s = [v * SCALE_D for v in relhf_min]
    relhf_max_s = [v * SCALE_D for v in relhf_max]
    relhf_flat_max = max((max(v) for v in relhf_all_s), default=0.0)
    relhf_flat_min = min((min(v) for v in relhf_all_s), default=0.0)
    span_d  = max(relhf_flat_max - min(relhf_flat_min, 0.0), 1e-9)
    ymax_d  = relhf_flat_max + span_d * 0.15
    ymin_d  = min(relhf_flat_min, 0.0) - span_d * 0.06

    relhf_whisker_coords = " ".join(
        f"({{{m}}},{med:.5f}) += (0,{(mx-med):.5f}) -= (0,{(med-mn):.5f})"
        for m, med, mn, mx in zip(mol_labels, relhf_med_s, relhf_min_s, relhf_max_s)
    )
    panel_d = (
        r"\begin{subfigure}[t]{0.48\textwidth}" + "\n"
        r"\centering\begin{tikzpicture}" + "\n"
        r"\begin{axis}[" + "\n"
        r"  width=\textwidth, height=0.70\textwidth," + "\n"
        r"  ylabel={$(E_{\mathrm{VQE}}^{\mathrm{GPU}} - E_{\mathrm{HF}})"
        r"\,/\,E_{\mathrm{VQE}}^{\mathrm{GPU}}\;(\times10^{-4})$}," + "\n"
        r"  scaled y ticks=false," + "\n"
        f"  symbolic x coords={{{sym_list}}}," + "\n"
        r"  enlarge x limits=0.06," + "\n"
        + tick_opts
        + f"  ymin={ymin_d:.9f}, ymax={ymax_d:.9f}," + "\n"
        r"  error bars/y dir=both," + "\n"
        r"  error bars/y explicit," + "\n"
        r"  error bars/error bar style={line width=0.8pt, orange!80!black}," + "\n"
        r"  error bars/error mark options={" + "\n"
        r"    line width=0.8pt, mark size=3pt, rotate=90, orange!80!black}," + "\n"
        r"  legend style={at={(0.5,-0.52)}, anchor=north, legend columns=4, font=\tiny}," + "\n"
        r"]" + "\n"
        # Whisker (min->max range), central marker hidden.
        f"\\addplot+[only marks, mark=none, orange!80!black, forget plot]"
        f" plot coordinates {{{relhf_whisker_coords}}};\n"
        # Dots colored by (Ne, No) active-space configuration (red->violet).
        + colored_dot_addplots(mol_labels, relhf_as, relhf_all_s, ".5f")
        + r"\addplot[gray!50, thin, mark=none]"
        + f" coordinates {{({{{mol_labels[0]}}},0) ({{{mol_labels[-1]}}},0)}};\n"
        r"\end{axis}\end{tikzpicture}" + "\n"
        r"\caption{Relative VQE improvement over Hartree--Fock (GPU energies):" + "\n"
        r"each marker is one $(N_e^{(a)},N_o^{(a)})$ configuration; the whisker" + "\n"
        r"spans the min/max of $(E_{\mathrm{VQE}}^{\mathrm{GPU}}-E_{\mathrm{HF}})" + "\n"
        r"/E_{\mathrm{VQE}}^{\mathrm{GPU}}$ per molecule. This is the correlation" + "\n"
        r"energy captured by VQE expressed as a fraction of the total VQE energy" + "\n"
        r"(dimensionless, $>0$); it normalises the absolute improvement in" + "\n"
        r"panel~(c) by the system's total energy.}" + "\n"
        r"\end{subfigure}" + "\n"
    )

    # ── Outer figure caption ──────────────────────────────────────
    body = (
        HEADER_SUBCAP
        + as_color_defs
        + r"\begin{figure*}[htbp]" + "\n"
        + r"\centering" + "\n"
        + r"\captionsetup[subfigure]{labelformat=parens, labelsep=space, font=small}" + "\n\n"
        + panel_a
        + panel_b
        + "\n"
        + r"\vspace{0.35cm}" + "\n\n"
        + panel_c
        + panel_d
        + "\n"
        + r"\caption{GPU-acceleration performance summary."
        + r" (a)~GPU speedup distribution (one dot per active-space"
        + r" configuration, min/max whiskers)"
        + r" of $t_{\mathrm{CPU}}/t_{\mathrm{GPU}}$ across all active-space"
        + r" configurations per molecule."
        + r" (b)~CPU--GPU numerical agreement distribution"
        + r" (one dot per configuration, min/max whiskers):"
        + r" $|\Delta E_{\mathrm{GPU-CPU}}|$ [mHa] across all configurations;"
        + r" dashed red line marks chemical accuracy (1.6\,mHa), which all"
        + r" configurations for all molecules lie below."
        + r" (c)~Maximum VQE improvement over HF:"
        + r" $|E_{\mathrm{VQE}}-E_{\mathrm{HF}}|$ at the active-space config"
        + r" capturing the most correlation energy per molecule"
        + r" (taller = better)."
        + r" (d)~Relative VQE improvement over HF (one dot per configuration,"
        + r" min/max whiskers): $(E_{\mathrm{VQE}}^{\mathrm{GPU}}-E_{\mathrm{HF}})"
        + r"/E_{\mathrm{VQE}}^{\mathrm{GPU}}$ across all configurations per"
        + r" molecule --- the correlation energy captured as a dimensionless"
        + r" fraction of the total VQE energy."
        + r" In panels~(a), (b) and~(d) each dot is one active-space"
        + r" configuration, coloured on a red$\to$violet scale by active-space"
        + r" size (smallest $(2,3)$ red $\to$ largest $(6,7)$ violet); panel~(b)"
        + r" excludes Methylene for scale (see its sub-caption)."
        + r" See Fig.~\ref{fig:accuracy_summary} for CCSD-accuracy comparison"
        + r" and multi-metric normalised view.}" + "\n"
        + r"\label{fig:performance_summary}" + "\n"
        + r"\end{figure*}" + "\n"
        + FOOTER
    )
    write_both(outdir, "fig15_performance_summary",
               body.replace(HEADER, "").replace(HEADER_SUBCAP, "").replace(FOOTER, "").strip())

    # ── Split embed variants: (a,b) and (c,d) as two separate figures ──
    # Same panels/colours as the combined figure, each in its own figure*
    # with a focused caption. Only the embeddable (\input) form is written.
    def _comment_subcaptions(block):
        """LaTeX-comment (%) every sub-figure \\caption{...} so only the main
        figure caption renders. Brace-matched, so multi-line captions are
        fully commented and easy to re-enable."""
        out, in_cap, depth = [], False, 0
        for ln in block.split("\n"):
            if not in_cap and ln.lstrip().startswith(r"\caption{"):
                in_cap, depth = True, 0
            if in_cap:
                depth += ln.count("{") - ln.count("}")
                out.append("% " + ln)
                if depth <= 0:
                    in_cap = False
            else:
                out.append(ln)
        return "\n".join(out)

    def _split_figure(panels, cap_body, label):
        return (
            as_color_defs
            + r"\begin{figure*}[htbp]" + "\n" + r"\centering" + "\n"
            + r"\captionsetup[subfigure]{labelformat=parens, labelsep=space,"
            + r" font=small}" + "\n\n"
            + _comment_subcaptions(panels) + "\n"
            + cap_body + "\n"
            + r"\label{" + label + r"}" + "\n"
            + r"\end{figure*}" + "\n"
        )

    cap_ab = (
        r"\caption{GPU-acceleration performance (part~1)."
        + r" (a)~GPU speedup distribution (one dot per active-space configuration,"
        + r" min/max whiskers) of $t_{\mathrm{CPU}}/t_{\mathrm{GPU}}$ across all"
        + r" configurations per molecule."
        + r" (b)~CPU--GPU numerical agreement distribution:"
        + r" $|\Delta E_{\mathrm{GPU-CPU}}|$ [mHa] across all configurations;"
        + r" dashed red line marks chemical accuracy (1.6\,mHa)."
        + r" Each dot is one active-space configuration, coloured on a"
        + r" red$\to$violet scale by active-space size (smallest $(2,3)$ red"
        + r" $\to$ largest $(6,7)$ violet); panel~(b) excludes Methylene for scale"
        + r" (see its sub-caption).}"
    )
    cap_cd = (
        r"\caption{GPU-acceleration performance (part~2)."
        + r" (c)~Maximum VQE improvement over HF:"
        + r" $|E_{\mathrm{VQE}}-E_{\mathrm{HF}}|$ at the active-space config"
        + r" capturing the most correlation energy per molecule (taller = better)."
        + r" (d)~Relative VQE improvement over HF (one dot per configuration,"
        + r" min/max whiskers): $(E_{\mathrm{VQE}}^{\mathrm{GPU}}-E_{\mathrm{HF}})"
        + r"/E_{\mathrm{VQE}}^{\mathrm{GPU}}$ across all configurations per molecule."
        + r" In panel~(d) each dot is one active-space configuration, coloured on a"
        + r" red$\to$violet scale by active-space size (smallest $(2,3)$ red $\to$"
        + r" largest $(6,7)$ violet).}"
    )
    write(os.path.join(outdir, "fig15_performance_summary_embed_a_b.tex"),
          HEADER_EMBED + _split_figure(panel_a + panel_b, cap_ab,
                                       "fig:performance_summary_ab") + FOOTER_EMBED)
    write(os.path.join(outdir, "fig15_performance_summary_embed_c_d.tex"),
          HEADER_EMBED + _split_figure(panel_c + panel_d, cap_cd,
                                       "fig:performance_summary_cd") + FOOTER_EMBED)


# ══════════════════════════════════════════════════════════════════
#  FIG 22 -- ACCURACY SUMMARY  (2-panel subfigure)
#
#  Panel (a): MIN |VQE−CCSD| per molecule (best-agreement config)
#  Panel (b): Normalised multi-metric bar (median speedup + best
#             |ΔE_CCSD| + best |ΔE_HF|, min-max normalised to [0,1])
#
#  Companion figure to fig15_performance_summary (which holds the
#  speedup + CPU/GPU agreement + VQE-HF correlation panels).
#  Fig 22 focuses on accuracy relative to the CCSD gold standard
#  and a combined cross-metric view across all molecules.
# ══════════════════════════════════════════════════════════════════

def fig22_accuracy_summary(cpu, gpu, molecules, outdir):
    """
    Two-panel accuracy-focused summary figure.

    Panel (a) -- left:
        MIN |VQE-CCSD| residual per molecule: for each molecule,
        the active-space config that comes closest to the CCSD
        reference. Positive y-axis (lower = closer to gold std).

    Panel (b) -- right:
        Min-max normalised multi-metric bar at preferred config:
        median speedup + best |ΔE_CCSD| + best |ΔE_HF| per molecule,
        all min-max scaled to [0,1] for cross-molecule comparison.

    Panels (c)--(e) of the original fig15 live here; panels (a)-(c)
    of the original fig15 remain in fig15_performance_summary.
    """
    print("\n[Fig 22] Accuracy summary -- 2-panel subfigure …")

    # ── Data collection ────────────────────────────────────────────
    mol_labels = []
    sp_med     = []               # for normalised panel only
    dE_vc_hf   = []
    dE_vg_hf   = []
    dE_vc_ccsd = []
    dE_vg_ccsd = []

    for mol in molecules:
        refs   = cpu[mol]["refs"]
        E_hf   = refs.get("E_hf")
        E_ccsd = refs.get("E_ccsd")
        pairs  = pair_runs(cpu[mol]["runs"], gpu[mol]["runs"])
        if not pairs:
            continue

        best = min(pairs, key=lambda p: p["E_cpu"] if p["E_cpu"] is not None else 0)
        if best["speedup"] is None or best["E_cpu"] is None or best["E_gpu"] is None:
            continue

        sp_list = [p["speedup"] for p in pairs if p["speedup"] is not None]
        if not sp_list:
            continue

        mol_labels.append(display(mol))
        sp_med.append(float(np.median(sp_list)))

        # BEST config per metric (same rule as original fig15 panels c, d)
        if E_hf is not None:
            best_hf = min(pairs, key=lambda p: (p["E_cpu"] - E_hf))
            dE_vc_hf.append(abs(best_hf["E_cpu"] - E_hf) * 1000)
            dE_vg_hf.append(abs(best_hf["E_gpu"] - E_hf) * 1000)
        else:
            dE_vc_hf.append(0.0)
            dE_vg_hf.append(0.0)

        if E_ccsd is not None:
            best_cc = min(pairs, key=lambda p: abs(p["E_cpu"] - E_ccsd))
            dE_vc_ccsd.append(abs(best_cc["E_cpu"] - E_ccsd) * 1000)
            dE_vg_ccsd.append(abs(best_cc["E_gpu"] - E_ccsd) * 1000)
        else:
            dE_vc_ccsd.append(0.0)
            dE_vg_ccsd.append(0.0)

    if not mol_labels:
        print("  SKIP -- no complete data for any molecule.")
        return

    # min-max normalise helper
    def minmax(v):
        lo, hi = min(v), max(v)
        if hi == lo:
            return [0.0] * len(v)
        return [(x - lo) / (hi - lo) for x in v]

    sp_norm  = minmax(sp_med)
    dcc_norm = minmax([abs(x) for x in dE_vc_ccsd])
    dhf_norm = minmax([abs(x) for x in dE_vc_hf])

    sym_list = ",".join(f"{{{m}}}" for m in mol_labels)

    def pts(vals):
        return " ".join(f"({{{m}}},{v:.5f})" for m, v in zip(mol_labels, vals))

    tick_opts = (
        r"  xtick=data," + "\n"
        r"  x tick label style={rotate=45, anchor=east, font=\scriptsize}," + "\n"
        r"  ymajorgrids=true, grid style={dotted,gray!30}," + "\n"
    )

    # ── Panel (a): MIN |VQE-CCSD| residual ─────────────────────────
    panel_a = (
        r"\begin{subfigure}[t]{0.48\textwidth}" + "\n"
        r"\centering\begin{tikzpicture}" + "\n"
        r"\begin{axis}[" + "\n"
        r"  ybar, bar width=5pt," + "\n"
        r"  enlarge x limits=0.06," + "\n"
        r"  width=\textwidth, height=0.70\textwidth," + "\n"
        r"  ylabel={$|E_{\mathrm{VQE}} - E_{\mathrm{CCSD}}|$ [mHa]}," + "\n"
        r"  ymin=0," + "\n"
        f"  symbolic x coords={{{sym_list}}}," + "\n"
        + tick_opts
        + r"  legend style={at={(0.5,-0.52)}, anchor=north, legend columns=2,"
        r"font=\scriptsize}," + "\n"
        r"]" + "\n"
        r"\addplot[fill=blue!60, draw=blue!80] coordinates {" + pts(dE_vc_ccsd) + "};\n"
        r"\addplot[fill=red!55,  draw=red!80]  coordinates {" + pts(dE_vg_ccsd) + "};\n"
        r"\legend{$|E_{\mathrm{VQE}}^{\mathrm{CPU}}-E_{\mathrm{CCSD}}|$,"
        r"$|E_{\mathrm{VQE}}^{\mathrm{GPU}}-E_{\mathrm{CCSD}}|$}" + "\n"
        r"\end{axis}\end{tikzpicture}" + "\n"
        r"\caption{Minimum residual VQE error vs CCSD per molecule," + "\n"
        r"$|E_{\mathrm{VQE}}-E_{\mathrm{CCSD}}|$ [mHa] (lower = closer to the" + "\n"
        r"CCSD gold standard). Each molecule's bar uses the active-space" + "\n"
        r"configuration achieving the smallest residual gap; see" + "\n"
        r"Fig.~\ref{fig:energy_error_vs_ccsd} for the full distribution.}" + "\n"
        r"\end{subfigure}\hfill" + "\n"
    )

    # ── Panel (b): Normalised multi-metric ────────────────────────
    panel_b = (
        r"\begin{subfigure}[t]{0.48\textwidth}" + "\n"
        r"\centering\begin{tikzpicture}" + "\n"
        r"\begin{axis}[" + "\n"
        r"  ybar, bar width=4pt," + "\n"
        r"  enlarge x limits=0.06," + "\n"
        r"  width=\textwidth, height=0.70\textwidth," + "\n"
        r"  ylabel={Normalised value $[0,1]$}," + "\n"
        r"  ymin=0, ymax=1.05," + "\n"
        f"  symbolic x coords={{{sym_list}}}," + "\n"
        + tick_opts
        + r"  legend style={at={(0.5,-0.62)}, anchor=north, legend columns=1,"
        r"font=\scriptsize}," + "\n"
        r"]" + "\n"
        r"\addplot[fill=blue!55,        draw=blue!75]  coordinates {" + pts(sp_norm)  + "};\n"
        r"\addplot[fill=green!55,       draw=green!70] coordinates {" + pts(dcc_norm) + "};\n"
        r"\addplot[fill=orange!65!white,draw=orange!80]coordinates {" + pts(dhf_norm) + "};\n"
        r"\legend{Speedup (norm.),"
        r"$|E_{\mathrm{VQE}}^{\mathrm{CPU}}-E_{\mathrm{CCSD}}|$ (norm.),"
        r"$|E_{\mathrm{VQE}}^{\mathrm{CPU}}-E_{\mathrm{HF}}|$ (norm.)}" + "\n"
        r"\end{axis}\end{tikzpicture}" + "\n"
        r"\caption{Multi-metric normalised comparison (min--max scaling):" + "\n"
        r"median speedup (blue), best-case CCSD error (green), best-case HF" + "\n"
        r"improvement (orange) per molecule on a common $[0,1]$ scale.}" + "\n"
        r"\end{subfigure}" + "\n"
    )

    # ── Outer figure caption ──────────────────────────────────────
    body = (
        HEADER_SUBCAP
        + r"\begin{figure*}[htbp]" + "\n"
        + r"\centering" + "\n"
        + r"\captionsetup[subfigure]{labelformat=parens, labelsep=space, font=small}" + "\n\n"
        + panel_a
        + panel_b
        + "\n"
        + r"\caption{VQE accuracy summary vs the CCSD reference."
        + r" (a)~Best-case residual error: $|E_{\mathrm{VQE}}-E_{\mathrm{CCSD}}|$"
        + r" at the active-space config minimising the gap per molecule"
        + r" (lower bars = closer to the CCSD gold standard)."
        + r" (b)~Min--max normalised multi-metric view combining median"
        + r" speedup (blue), best-case CCSD error (green), and best-case HF"
        + r" improvement (orange), enabling cross-molecule comparison on a"
        + r" common $[0,1]$ scale. Companion figure to"
        + r" Fig.~\ref{fig:performance_summary}, which focuses on the"
        + r" GPU-acceleration story (speedup, CPU/GPU agreement, HF improvement)."
        + r" See Figs.~\ref{fig:correlation_recovery} and"
        + r" \ref{fig:energy_error_vs_ccsd} for full distributions across"
        + r" all configurations.}" + "\n"
        + r"\label{fig:accuracy_summary}" + "\n"
        + r"\end{figure*}" + "\n"
        + FOOTER
    )
    write_both(outdir, "fig22_accuracy_summary",
               body.replace(HEADER, "").replace(HEADER_SUBCAP, "").replace(FOOTER, "").strip())


# ══════════════════════════════════════════════════════════════════
#  FIG 23 -- GPU SPEEDUP HISTOGRAM  (single panel, all data points)
#
#  Condenses the per-molecule runtime/speedup panels (Figs 6, 7) into
#  ONE distribution plot. Every (molecule, active-space configuration)
#  contributes a single speedup value t_CPU / t_GPU; these ~120 points
#  are pooled and binned into a histogram.
#    x = GPU speedup,  y = number of configurations.
#  Median marked with a dashed red line; N / mean / max annotated.
#  Read straight from the pkl data via pair_runs() -- no external input.
# ══════════════════════════════════════════════════════════════════

def fig23_speedup_histogram(cpu, gpu, molecules, outdir):
    """
    Single-panel histogram of GPU speedup (t_CPU / t_GPU) pooled over
    EVERY (molecule, active-space configuration) data point.

    All speedups come from pair_runs(), i.e. directly from the loaded
    pkl files -- the same source every other figure uses. Bins are
    fixed-width (BIN_W = 2x) running from 0 up to the next even number
    above the maximum speedup. The bar for bin [e_i, e_{i+1}) is the
    count of configurations whose speedup falls in that interval.
    """
    print("\n[Fig 23] GPU speedup histogram (all configurations) …")

    # ── pool per-configuration speedups across all molecules ──
    # Restrict to the 9 officially-reported canonical active spaces
    # (CANONICAL_AS) so the total is 13 x 9 = 117, not the raw pkl count
    # (which includes non-standard extras like (4,6)/(2,2)).
    canon = set(CANONICAL_AS)
    speedups = [p["speedup"]
                for mol in molecules
                for p in pair_runs(cpu[mol]["runs"], gpu[mol]["runs"])
                if p["speedup"] is not None and p["key"] in canon]
    if not speedups:
        print("  SKIP -- no speedup data.")
        return

    arr  = np.array(speedups, dtype=float)
    n    = int(arr.size)
    med  = float(np.median(arr))
    mean = float(arr.mean())
    vmax = float(arr.max())

    # ── fixed-width linear bins (width 2x), 0 → next even above max ──
    BIN_W     = 2.0
    hi        = float(math.ceil(vmax / BIN_W) * BIN_W)
    edges     = np.arange(0.0, hi + BIN_W, BIN_W)
    counts, _ = np.histogram(arr, bins=edges)
    ymax_c    = int(counts.max()) if len(counts) else 1

    # ybar interval needs N+1 coordinates (height of the last point is
    # ignored; it only supplies the right edge of the final bar).
    coords = (" ".join(f"({edges[i]:.0f},{int(counts[i])})"
                       for i in range(len(counts)))
              + f" ({edges[-1]:.0f},0)")

    xtick = ",".join(str(int(x)) for x in np.arange(0, hi + 1, 5))
    ytop  = ymax_c * 1.18
    yline = ymax_c * 1.08

    body = (
        HEADER
        + r"\begin{figure*}[htbp]" + "\n"
        + r"\centering" + "\n"
        + r"\begin{tikzpicture}" + "\n"
        + r"\begin{axis}[" + "\n"
        + r"  width=14cm, height=7.8cm," + "\n"
        + r"  xlabel={GPU speedup ($t_{\mathrm{CPU}}/t_{\mathrm{GPU}}$)}," + "\n"
        + r"  ylabel={Occurrence}," + "\n"
        + f"  xmin=0, xmax={hi:.0f}, ymin=0, ymax={ytop:.2f}," + "\n"
        + f"  xtick={{{xtick}}}," + "\n"
        + r"  ymajorgrids=true, grid style={dotted,gray!30}," + "\n"
        + r"  tick label style={font=\small}, label style={font=\small}," + "\n"
        + r"  axis on top," + "\n"
        + r"]" + "\n"
        + r"\addplot[ybar interval, fill=blue!55, draw=black!70,"
        + r" fill opacity=0.85] coordinates {" + coords + "};\n"
        + f"\\draw[red, thick, dashed] (axis cs:{med:.4f},0) --"
        + f" (axis cs:{med:.4f},{yline:.2f});\n"
        + r"\end{axis}" + "\n"
        + r"\end{tikzpicture}" + "\n"
        + r"\caption{Distribution of GPU speedup $t_{\mathrm{CPU}}/t_{\mathrm{GPU}}$"
        + r" pooled over all " + str(n) + r" (molecule, active-space"
        + r" configuration) data points (cc-pVDZ). Bin width $2\times$;"
        + r" the dashed red line marks the median"
        + f" (${med:.1f}\\times$; mean ${mean:.1f}\\times$). Most configurations cluster at modest"
        + r" speedups, while a tail of large active spaces reaches"
        + f" ${vmax:.0f}\\times$. This single panel summarises the"
        + r" per-molecule runtime comparisons"
        + r" (Figs.~\ref{fig:speedup_all_No_a},"
        + r"~\ref{fig:ccpvdz_runtime_all_molecules}).}" + "\n"
        + r"\label{fig:speedup_histogram}" + "\n"
        + r"\end{figure*}" + "\n"
        + FOOTER
    )
    write_both(outdir, "fig23_speedup_histogram",
               body.replace(HEADER, "").replace(HEADER_SUBCAP, "").replace(FOOTER, "").strip())


# ══════════════════════════════════════════════════════════════════
#  FIG 24 -- VQE ACCURACY vs CASCI  (single panel, all molecules)
#
#  The genuine VQE-accuracy figure: per molecule, the absolute gap
#      |E_VQE - E_CASCI|  [mHa]
#  to the EXACT diagonalisation of the same active-space Hamiltonian
#  VQE solves (CASCI = FCI within the active space). This isolates the
#  VQE ansatz + optimiser error from active-space truncation, unlike
#  |E_VQE - E_CCSD| (Fig 22a) whose magnitude is dominated by
#  correlation lying OUTSIDE the active space.
#
#  Grouped bars: CPU (blue) vs GPU (red); dashed orange chemical-accuracy
#  line at 1.6 mHa. Writes ONE pair of files
#  (fig24_vqe_casci_accuracy{,_embed}.tex) -> drop-in \input.
# ══════════════════════════════════════════════════════════════════

def fig24_vqe_casci_accuracy(cpu, gpu, molecules, outdir):
    """
    True VQE accuracy relative to the exact active-space reference (CASCI).

    For every molecule a single summary statistic of the per-configuration
    gap |E_VQE - E_CASCI| [mHa] is shown as a grouped bar (CPU blue, GPU
    red). CASCI is the exact ground state of the SAME active-space
    Hamiltonian that VQE solves, so this gap is the genuine VQE
    ansatz/optimiser error -- with active-space truncation removed. This
    is the accuracy reference to use instead of HF (a deliberately low
    bar) or full-system CCSD (whose ~Hartree gap is dominated by
    correlation outside the active space, not by VQE error).

    A dashed orange line marks chemical accuracy (1.6 mHa ~ 1 kcal/mol).

    STAT selects the per-molecule statistic over all configurations:
      "max"    -- worst-case gap (most conservative; default)
      "median" -- typical gap
      "min"    -- best-case gap
    Uses the stored d_casci field where available, else falls back to
    E_VQE - E_CASCI directly. Reads only pkl data via pair_runs().
    """
    print("\n[Fig 24] VQE accuracy vs CASCI (exact active-space reference) …")

    STAT     = "max"      # "max" | "median" | "min"
    CHEM_ACC = 1.6        # mHa  ~ 1 kcal/mol

    def summarise(vals):
        if not vals:
            return None
        if STAT == "median":
            return float(np.median(vals))
        if STAT == "min":
            return min(vals)
        return max(vals)              # default: worst-case

    mol_labels, gap_cpu, gap_gpu = [], [], []

    for mol in molecules:
        pairs = pair_runs(cpu[mol]["runs"], gpu[mol]["runs"])
        if not pairs:
            continue

        gaps_c, gaps_g = [], []
        for p in pairs:
            # CPU gap vs exact active-space CASCI energy (Ha -> mHa)
            dc = p["cpu"].get("d_casci")
            if dc is None and p["E_cpu"] is not None \
                    and p["cpu"].get("E_casci") is not None:
                dc = p["E_cpu"] - p["cpu"]["E_casci"]
            # GPU gap
            dg = p["gpu"].get("d_casci")
            if dg is None and p["E_gpu"] is not None \
                    and p["gpu"].get("E_casci") is not None:
                dg = p["E_gpu"] - p["gpu"]["E_casci"]
            if dc is not None:
                gaps_c.append(abs(dc) * 1000.0)
            if dg is not None:
                gaps_g.append(abs(dg) * 1000.0)

        sc = summarise(gaps_c)
        sg = summarise(gaps_g)
        if sc is None and sg is None:
            continue

        mol_labels.append(display(mol))
        gap_cpu.append(sc if sc is not None else 0.0)
        gap_gpu.append(sg if sg is not None else 0.0)

    if not mol_labels:
        print("  SKIP -- no molecules have CASCI reference data.")
        return

    sym_list = ",".join(f"{{{m}}}" for m in mol_labels)

    def pts(vals):
        return " ".join(f"({{{m}}},{v:.6f})" for m, v in zip(mol_labels, vals))

    # y-range: keep the chemical-accuracy line visible even when every
    # bar sits well below it (the expected, desirable outcome).
    data_max = max(max(gap_cpu, default=0.0), max(gap_gpu, default=0.0))
    ymax_v   = max(data_max, CHEM_ACC) * 1.20

    stat_word = {"max": "Worst-case", "median": "Median",
                 "min": "Best-case"}[STAT]

    # Chemical-accuracy reference line via \pgfplotsextra (avoids the ybar
    # handler rendering a 2-point \addplot as bars). first_sym supplies a
    # valid symbolic x for the axis-cs y anchor; the |- keeps only its y.
    first_sym = f"{{{mol_labels[0]}}}"
    chem_line = (
        r"\pgfplotsextra{%" + "\n"
        f"  \\draw[orange!80!black, dashed, line width=1pt]\n"
        f"    ({{rel axis cs:0,0}}|-{{axis cs:{first_sym},{CHEM_ACC:.2f}}}) --\n"
        f"    ({{rel axis cs:1,0}}|-{{axis cs:{first_sym},{CHEM_ACC:.2f}}});\n"
        r"}%" + "\n"
    )

    body = (
        HEADER
        + r"\begin{figure*}[htbp]" + "\n"
        + r"\centering\resizebox{\textwidth}{!}{%" + "\n"
        + r"\begin{tikzpicture}" + "\n"
        + r"\begin{axis}[" + "\n"
        + r"  ybar, bar width=6pt," + "\n"
        + r"  width=\textwidth, height=0.42\textwidth," + "\n"
        + r"  enlarge x limits=0.05," + "\n"
        + f"  symbolic x coords={{{sym_list}}}," + "\n"
        + r"  xtick=data," + "\n"
        + r"  x tick label style={rotate=45, anchor=east, font=\small}," + "\n"
        + r"  ymajorgrids=true, grid style={dotted,gray!30}," + "\n"
        + r"  ylabel={$|E_{\mathrm{VQE}} - E_{\mathrm{CASCI}}|$ [mHa]}," + "\n"
        + f"  ymin=0, ymax={ymax_v:.4f}," + "\n"
        + r"  tick label style={font=\small}, label style={font=\small}," + "\n"
        + r"  legend style={at={(0.99,0.99)}, anchor=north east,"
        + r" legend columns=1, font=\small}," + "\n"
        + r"]" + "\n"
        + r"\addplot[fill=blue!60, draw=blue!80] coordinates {" + pts(gap_cpu) + "};\n"
        + r"\addlegendentry{$|E_{\mathrm{VQE}}^{\mathrm{CPU}}-E_{\mathrm{CASCI}}|$}" + "\n"
        + r"\addplot[fill=red!55,  draw=red!80]  coordinates {" + pts(gap_gpu) + "};\n"
        + r"\addlegendentry{$|E_{\mathrm{VQE}}^{\mathrm{GPU}}-E_{\mathrm{CASCI}}|$}" + "\n"
        + r"\addlegendimage{orange!80!black, dashed, line width=1pt}" + "\n"
        + r"\addlegendentry{Chem.\ accuracy (1.6\,mHa)}" + "\n"
        + chem_line
        + r"\end{axis}" + "\n"
        + r"\end{tikzpicture}" + "\n"
        + r"}" + "\n"
        + r"\caption{" + stat_word + r" VQE accuracy relative to the exact"
        + r" active-space reference (CASCI) per molecule:"
        + r" $|E_{\mathrm{VQE}}-E_{\mathrm{CASCI}}|$ [mHa] for CPU (blue) and"
        + r" GPU (red), taken as the " + STAT + r" over all active-space"
        + r" configurations $(N_e^{(a)},N_o^{(a)})$. CASCI is the exact"
        + r" diagonalisation of the same active-space Hamiltonian solved by"
        + r" VQE, so this gap measures the genuine VQE ansatz/optimiser error"
        + r" with active-space truncation removed --- unlike"
        + r" $|E_{\mathrm{VQE}}-E_{\mathrm{CCSD}}|$"
        + r" (Fig.~\ref{fig:accuracy_summary}a), which is dominated by"
        + r" correlation outside the active space. The dashed orange line"
        + r" marks chemical accuracy ($1.6\,\mathrm{mHa}\approx"
        + r" 1\,\mathrm{kcal/mol}$); bars far below it confirm VQE reproduces"
        + r" the exact active-space energy on both backends, with CPU and GPU"
        + r" in close agreement.}" + "\n"
        + r"\label{fig:vqe_casci_accuracy}" + "\n"
        + r"\end{figure*}" + "\n"
        + FOOTER
    )
    write_both(outdir, "fig24_vqe_casci_accuracy",
               body.replace(HEADER, "").replace(HEADER_SUBCAP, "").replace(FOOTER, "").strip())


# ══════════════════════════════════════════════════════════════════
#  FIG 25 -- VQE "ACCURACY" vs CCSD  (single panel, all molecules)
#
#  The CCSD counterpart of Fig 24. Per molecule, |E_VQE - E_CCSD| [mHa]
#  against the FULL-SYSTEM CCSD reference, CPU (blue) vs GPU (red).
#
#  IMPORTANT INTERPRETATION (different from Fig 24!):
#  CCSD here correlates the ENTIRE cc-pVDZ system, while VQE correlates
#  only a small active space. The gap is therefore DOMINATED BY
#  ACTIVE-SPACE TRUNCATION (correlation lying outside the active space),
#  NOT by VQE error. Expect bars of 100s--1000s mHa -- two to three
#  orders of magnitude above the 1.6 mHa chemical-accuracy line -- even
#  though VQE solves its own active-space problem essentially exactly
#  (see Fig 24). Plotted on a LOG y-axis so the dynamic range and the
#  chem-accuracy line are both visible. This is a "fraction of total
#  correlation captured" view, not a VQE-accuracy view.
# ══════════════════════════════════════════════════════════════════

def fig25_vqe_ccsd_accuracy(cpu, gpu, molecules, outdir):
    """
    |E_VQE - E_CCSD| [mHa] per molecule (CPU blue, GPU red), log y-axis.

    Mirrors fig24 but uses the full-system CCSD reference. The resulting
    gap is dominated by active-space truncation rather than VQE error,
    so the bars are large (100s--1000s mHa) and sit far above chemical
    accuracy -- the opposite visual of fig24. Use this to show how much
    of the TOTAL (full-system) correlation the active space leaves on
    the table; use fig24 (vs CASCI) for genuine VQE accuracy.

    STAT selects the per-molecule statistic over all configurations:
      "min"    -- best-case (smallest) residual (closest to CCSD; default)
      "median" -- typical residual
      "max"    -- worst-case residual
    Uses the stored d_ccsd field where available, else E_VQE - E_CCSD.
    """
    print("\n[Fig 25] VQE residual vs full-system CCSD …")

    STAT     = "min"      # "min" | "median" | "max"
    CHEM_ACC = 1.6        # mHa  ~ 1 kcal/mol

    def summarise(vals):
        if not vals:
            return None
        if STAT == "median":
            return float(np.median(vals))
        if STAT == "max":
            return max(vals)
        return min(vals)              # default: best-case

    mol_labels, gap_cpu, gap_gpu = [], [], []

    for mol in molecules:
        E_ccsd = cpu[mol]["refs"].get("E_ccsd")
        pairs  = pair_runs(cpu[mol]["runs"], gpu[mol]["runs"])
        if not pairs or E_ccsd is None:
            continue

        gaps_c, gaps_g = [], []
        for p in pairs:
            dc = p["cpu"].get("d_ccsd")
            if dc is None and p["E_cpu"] is not None:
                dc = p["E_cpu"] - E_ccsd
            dg = p["gpu"].get("d_ccsd")
            if dg is None and p["E_gpu"] is not None:
                dg = p["E_gpu"] - E_ccsd
            if dc is not None:
                gaps_c.append(abs(dc) * 1000.0)
            if dg is not None:
                gaps_g.append(abs(dg) * 1000.0)

        sc = summarise(gaps_c)
        sg = summarise(gaps_g)
        if sc is None and sg is None:
            continue

        mol_labels.append(display(mol))
        gap_cpu.append(sc if sc is not None else 0.0)
        gap_gpu.append(sg if sg is not None else 0.0)

    if not mol_labels:
        print("  SKIP -- no molecules have CCSD reference data.")
        return

    sym_list = ",".join(f"{{{m}}}" for m in mol_labels)

    def pts(vals):
        return " ".join(f"({{{m}}},{v:.6f})" for m, v in zip(mol_labels, vals))

    data_max = max(max(gap_cpu, default=0.0), max(gap_gpu, default=0.0))
    data_min = min([v for v in (gap_cpu + gap_gpu) if v > 0], default=1.0)
    # log window: floor below the chem-acc line so it is visible, ceiling
    # above the largest bar.
    ymin_v = min(data_min * 0.5, CHEM_ACC * 0.5)
    ymax_v = max(data_max, CHEM_ACC) * 1.6

    stat_word = {"max": "Worst-case", "median": "Median",
                 "min": "Best-case"}[STAT]

    first_sym = f"{{{mol_labels[0]}}}"
    chem_line = (
        r"\pgfplotsextra{%" + "\n"
        f"  \\draw[orange!80!black, dashed, line width=1pt]\n"
        f"    ({{rel axis cs:0,0}}|-{{axis cs:{first_sym},{CHEM_ACC:.2f}}}) --\n"
        f"    ({{rel axis cs:1,0}}|-{{axis cs:{first_sym},{CHEM_ACC:.2f}}});\n"
        r"}%" + "\n"
    )

    body = (
        HEADER
        + r"\begin{figure*}[htbp]" + "\n"
        + r"\centering\resizebox{\textwidth}{!}{%" + "\n"
        + r"\begin{tikzpicture}" + "\n"
        + r"\begin{axis}[" + "\n"
        + r"  ybar, bar width=6pt," + "\n"
        + r"  width=\textwidth, height=0.42\textwidth," + "\n"
        + r"  enlarge x limits=0.05," + "\n"
        + f"  symbolic x coords={{{sym_list}}}," + "\n"
        + r"  xtick=data," + "\n"
        + r"  x tick label style={rotate=45, anchor=east, font=\small}," + "\n"
        + r"  ymajorgrids=true, grid style={dotted,gray!30}," + "\n"
        + r"  ymode=log," + "\n"
        + r"  ylabel={$|E_{\mathrm{VQE}} - E_{\mathrm{CCSD}}|$ [mHa]}," + "\n"
        + f"  ymin={ymin_v:.4f}, ymax={ymax_v:.4f}," + "\n"
        + r"  tick label style={font=\small}, label style={font=\small}," + "\n"
        + r"  legend style={at={(0.99,0.99)}, anchor=north east,"
        + r" legend columns=1, font=\small}," + "\n"
        + r"]" + "\n"
        + r"\addplot[fill=blue!60, draw=blue!80] coordinates {" + pts(gap_cpu) + "};\n"
        + r"\addlegendentry{$|E_{\mathrm{VQE}}^{\mathrm{CPU}}-E_{\mathrm{CCSD}}|$}" + "\n"
        + r"\addplot[fill=red!55,  draw=red!80]  coordinates {" + pts(gap_gpu) + "};\n"
        + r"\addlegendentry{$|E_{\mathrm{VQE}}^{\mathrm{GPU}}-E_{\mathrm{CCSD}}|$}" + "\n"
        + r"\addlegendimage{orange!80!black, dashed, line width=1pt}" + "\n"
        + r"\addlegendentry{Chem.\ accuracy (1.6\,mHa)}" + "\n"
        + chem_line
        + r"\end{axis}" + "\n"
        + r"\end{tikzpicture}" + "\n"
        + r"}" + "\n"
        + r"\caption{" + stat_word + r" residual of VQE relative to the"
        + r" full-system CCSD reference per molecule:"
        + r" $|E_{\mathrm{VQE}}-E_{\mathrm{CCSD}}|$ [mHa] for CPU (blue) and"
        + r" GPU (red), taken as the " + STAT + r" over all active-space"
        + r" configurations $(N_e^{(a)},N_o^{(a)})$ (log scale)."
        + r" Unlike Fig.~\ref{fig:vqe_casci_accuracy} (vs CASCI), CCSD"
        + r" correlates the \emph{entire} cc-pVDZ system whereas VQE"
        + r" correlates only the active space, so these residuals are"
        + r" dominated by active-space \emph{truncation} (correlation lying"
        + r" outside the active space), not by VQE error --- hence bars two"
        + r" to three orders of magnitude above the chemical-accuracy line"
        + r" even though VQE reproduces the exact active-space energy"
        + r" (Fig.~\ref{fig:vqe_casci_accuracy}). This panel therefore"
        + r" measures how much of the total correlation the chosen active"
        + r" spaces capture, and is best read alongside the correlation-"
        + r"recovery figures (Figs.~\ref{fig:correlation_recovery},"
        + r"~\ref{fig:correlation_recovery_log}).}" + "\n"
        + r"\label{fig:vqe_ccsd_accuracy}" + "\n"
        + r"\end{figure*}" + "\n"
        + FOOTER
    )
    write_both(outdir, "fig25_vqe_ccsd_accuracy",
               body.replace(HEADER, "").replace(HEADER_SUBCAP, "").replace(FOOTER, "").strip())


# ══════════════════════════════════════════════════════════════════
#  FIG 17 -- VQE OPTIMISATION CONVERGENCE CURVES
#           Energy vs function-evaluation index for best config
#           per molecule; CPU blue, GPU red.
#           Also marks the CCSD target as a dashed green h-line.
#           Research value: shows optimizer efficiency and landscape
#           smoothness; noisy curves → barren plateau risk.
# ══════════════════════════════════════════════════════════════════

def fig17_convergence_curves(cpu, gpu, molecules, outdir):
    """
    Per-molecule convergence plot: total VQE energy vs iteration index.
    Uses the (Ne,No) config with the most function evaluations (richest curve).
    CPU: solid blue.  GPU: dashed red.  CCSD target: dotted green h-line.
    """
    print("\n[Fig 17] VQE convergence curves (energy vs iteration) …")
    NCOLS = 4
    nrows = (len(molecules) + NCOLS - 1) // NCOLS + 1

    PREF_NE, PREF_NO = 6, 7

    blocks = []
    for mol in molecules:
        refs   = cpu[mol]["refs"]
        E_ccsd = refs.get("E_ccsd")

        # pick richest convergence curve (most nfev)
        def pick_run(runs):
            pref = [r for r in runs if r["ne"]==PREF_NE and r["no"]==PREF_NO]
            pool = pref if pref else runs
            return max(pool, key=lambda r: len(r.get("econv", [])), default=None)

        cr = pick_run(cpu[mol]["runs"])
        gr = pick_run(gpu[mol]["runs"])
        if cr is None or not cr.get("econv"):
            continue

        ec = cr["econv"]
        eg = gr["econv"] if gr and gr.get("econv") else []

        # subsample to <=100 pts for tex size
        def sub(lst, n=100):
            if len(lst) <= n: return list(lst)
            step = len(lst)//n
            return lst[::step]

        ec_s = sub(ec); eg_s = sub(eg)
        c_coords = " ".join(f"({i},{v:.6f})" for i, v in enumerate(ec_s))
        g_coords = " ".join(f"({i},{v:.6f})" for i, v in enumerate(eg_s))

        all_E = ec_s + eg_s
        if not all_E: continue
        ymin_v = min(all_E); ymax_v = max(all_E)
        rng = max(abs(ymax_v - ymin_v), 1e-6)
        ymin_v -= rng * 0.12; ymax_v += rng * 0.12
        xmax_v = max(len(ec_s), len(eg_s), 1) - 1

        ccsd_line = ""
        if E_ccsd is not None and ymin_v <= E_ccsd <= ymax_v + rng:
            ccsd_line = (
                r"\pgfplotsextra{%" + "\n"
                f"  \\draw[green!60!black, dotted, line width=1.3pt]"
                f" (axis cs:0,{E_ccsd:.6f}) -- (axis cs:{xmax_v},{E_ccsd:.6f});\n"
                r"}%" + "\n"
            )

        label_str = f"({cr['ne']},{cr['no']})"
        col_idx17 = len(blocks) % NCOLS
        ylabel_17 = (
            r"xlabel={Iteration}, ylabel={$E_{\mathrm{VQE}}$ [Ha]},"
            if col_idx17 == 0 else
            r"xlabel={Iteration}, ylabel={},"
        )
        blocks.append(
            f"% === {mol} ===\n"
            f"\\nextgroupplot[\n"
            f"  xmin=0, xmax={xmax_v},\n"
            f"  ymin={ymin_v:.6f}, ymax={ymax_v:.6f},\n"
            f"  scaled y ticks=false,\n"
            f"  {ylabel_17}\n"
            f"  y tick label style={{/pgf/number format/fixed precision=3, font=\\tiny}},\n"
            f"]\n"
            + ccsd_line
            + f"\\addplot[blue, thick, mark=none] coordinates {{{c_coords}}};\n"
            + (f"\\addplot[red, dashed, thick, mark=none] coordinates {{{g_coords}}};\n"
               if g_coords else "")
            + mol_label_node(mol, f"{display(mol)} {label_str}", bold=True)
            + "\n"
        )

    if not blocks:
        print("  SKIP -- no convergence data found.")
        return

    blocks.append(
        r"\nextgroupplot[hide axis, legend style={at={(0.5,0.5)},anchor=center,"
        r"legend columns=1,font=\small,row sep=3pt}]" + "\n"
        r"\addlegendimage{blue, thick, mark=none}" + "\n"
        r"\addlegendentry{VQE (CPU)}" + "\n"
        r"\addlegendimage{red, dashed, thick, mark=none}" + "\n"
        r"\addlegendentry{VQE (GPU)}" + "\n"
        r"\addlegendimage{green!60!black, dotted, line width=1.3pt}" + "\n"
        r"\addlegendentry{$E_{\mathrm{CCSD}}$ target}" + "\n"
    )

    body = (
        HEADER
        + r"\begin{figure*}[htbp]" + "\n"
        + r"\centering\resizebox{1.1\textwidth}{!}{%" + "\n"
        + r"\begin{tikzpicture}" + "\n"
        + r"\begin{groupplot}[" + "\n"
        + f"  group style={{group size={NCOLS} by {nrows},"
        + r"horizontal sep=1.8cm,vertical sep=1.6cm}," + "\n"
        + r"  width=0.27\textwidth, height=0.30\textwidth," + "\n"
        + r"  tick label style={font=\scriptsize}," + "\n"
        + r"  label style={font=\scriptsize}," + "\n"
        + r"  ymajorgrids=true, grid style={dotted,gray!30}," + "\n"
        + "]\n\n"
        + "\n".join(blocks)
        + r"\end{groupplot}" + "\n"
        + r"\end{tikzpicture}" + "\n"
        + r"}" + "\n"
        + r"\caption{VQE optimisation convergence curves for each molecule"
        + r" (preferred $(N_e^{(a)},N_o^{(a)})=(6,7)$, fallback to config"
        + r" with most function evaluations). Solid blue: CPU; dashed red:"
        + r" GPU. Dotted green: $E_{\mathrm{CCSD}}$ target where within"
        + r" plot range. Noisy or non-monotone curves indicate a challenging"
        + r" optimisation landscape; smooth descent indicates well-conditioned"
        + r" UCCSD parametrisation.}" + "\n"
        + r"\label{fig:vqe_convergence}" + "\n"
        + r"\end{figure*}" + "\n"
        + FOOTER
    )
    write_both(outdir, "fig17_convergence_curves", body.replace(HEADER, "").replace(HEADER_SUBCAP, "").replace(FOOTER, "").strip())


# ══════════════════════════════════════════════════════════════════
#  FIG 18 -- RUNTIME BREAKDOWN: quantum vs optimizer vs overhead
#           Stacked bar per (Ne,No) config per molecule (CPU only)
#           Research value: reveals where wall-clock time is spent;
#           GPU should shrink the "quantum" slice most dramatically.
# ══════════════════════════════════════════════════════════════════

def fig18_runtime_breakdown(cpu, gpu, molecules, outdir):
    """
    Stacked bar chart: for each (Ne,No) config, the CPU runtime is split
    into three components:
      - Quantum simulation time  (blue)
      - Classical optimizer time (orange)
      - Overhead (setup, I/O, etc.) (grey)
    GPU total runtime overlaid as a red step line for comparison.
    """
    print("\n[Fig 18] Runtime breakdown (quantum / optimizer / overhead) …")
    NCOLS = 4
    nrows = (len(molecules) + NCOLS - 1) // NCOLS + 1

    blocks = []
    for mol in molecules:
        pairs = pair_runs(cpu[mol]["runs"], gpu[mol]["runs"])
        if not pairs:
            continue
        pairs.sort(key=lambda p: (p["ne"], p["no"]))

        sym_coords = [sym(p["ne"], p["no"]) for p in pairs]
        sym_labels = [p["label"]             for p in pairs]
        sym_list   = ",".join(sym_coords)
        lbl_list   = ",".join(sym_labels)

        q_pts   = []   # quantum
        opt_pts = []   # optimizer
        ov_pts  = []   # overhead
        gpu_pts = []   # GPU total

        for p in pairs:
            sc  = sym(p["ne"], p["no"])
            cr  = p["cpu"]
            rt  = cr.get("runtime") or 0
            rtq = cr.get("rt_q")    or rt
            rto = cr.get("rt_opt")  or 0
            rth = max(rt - rtq - rto, 0)
            q_pts.append(  f"({sc},{max(rtq,0.001):.4f})")
            opt_pts.append(f"({sc},{max(rto,0.001):.4f})")
            ov_pts.append( f"({sc},{max(rth,0.001):.4f})")
            if p["rt_gpu"]:
                gpu_pts.append(f"({sc},{p['rt_gpu']:.4f})")

        col_idx18 = len(blocks) % NCOLS
        ylabel_18 = "" if col_idx18 == 0 else "  ylabel={},\n"
        blocks.append(
            f"% === {mol} ===\n"
            f"\\nextgroupplot[\n"
            f"  symbolic x coords={{{sym_list}}},\n"
            f"  xticklabels={{{lbl_list}}},\n"
            f"  xticklabel style={{font=\\tiny, rotate=45, anchor=east}},\n"
            f"  ymode=log,\n"
            f"{ylabel_18}]\n"
            f"\\addplot+[ybar stacked, fill=blue!60,    draw=blue!80]"
            f" coordinates {{{' '.join(q_pts)}}};\n"
            f"\\addplot+[ybar stacked, fill=orange!70,  draw=orange!90]"
            f" coordinates {{{' '.join(opt_pts)}}};\n"
            f"\\addplot+[ybar stacked, fill=gray!45,    draw=gray!60]"
            f" coordinates {{{' '.join(ov_pts)}}};\n"
            + (f"\\addplot+[red, thick, mark=square*, mark size=1.5pt,"
               f" mark options={{fill=red}}, ybar=0pt, bar width=0pt]"
               f" coordinates {{{' '.join(gpu_pts)}}};\n"
               if gpu_pts else "")
            + mol_label_node(mol, display(mol), bold=True, anchor='south west', pos='0.02,0.02')
            + "\n"
        )

    if not blocks:
        print("  SKIP -- no runtime data.")
        return

    blocks.append(
        r"\nextgroupplot[hide axis, legend style={at={(0.5,0.5)},anchor=center,"
        r"legend columns=1,font=\small,row sep=3pt}]" + "\n"
        r"\addlegendimage{ybar stacked, fill=blue!60,   draw=blue!80}"   + "\n"
        r"\addlegendentry{Quantum simulation [s]}" + "\n"
        r"\addlegendimage{ybar stacked, fill=orange!70, draw=orange!90}" + "\n"
        r"\addlegendentry{Classical optimizer [s]}" + "\n"
        r"\addlegendimage{ybar stacked, fill=gray!45,   draw=gray!60}"   + "\n"
        r"\addlegendentry{Overhead [s]}" + "\n"
        r"\addlegendimage{red, thick, mark=square*}" + "\n"
        r"\addlegendentry{GPU total [s]}" + "\n"
    )

    body = (
        HEADER
        + r"\begin{figure*}[htbp]" + "\n"
        + r"\centering\resizebox{1.1\textwidth}{!}{%" + "\n"
        + r"\begin{tikzpicture}" + "\n"
        + r"\begin{groupplot}[" + "\n"
        + f"  group style={{group size={NCOLS} by {nrows},"
        + r"horizontal sep=1.8cm,vertical sep=1.5cm}," + "\n"
        + r"  width=0.27\textwidth, height=0.32\textwidth," + "\n"
        + r"  ybar stacked, xtick=data," + "\n"
        + r"  tick label style={font=\scriptsize}," + "\n"
        + r"  label style={font=\scriptsize}," + "\n"
        + r"  ymajorgrids=true, grid style={dotted,gray!30}," + "\n"
        + r"  ylabel={Runtime [s] (log)}," + "\n"
        + "]\n\n"
        + "\n".join(blocks)
        + r"\end{groupplot}" + "\n"
        + r"\end{tikzpicture}" + "\n"
        + r"}" + "\n"
        + r"\caption{CPU runtime decomposition per active-space configuration"
        + r" $(N_e^{(a)},N_o^{(a)})$: quantum simulation (blue),"
        + r" classical COBYLA optimizer (orange), and overhead (grey)."
        + r" GPU total runtime shown as red markers."
        + r" GPU acceleration primarily compresses the quantum-simulation slice;"
        + r" the optimizer overhead is hardware-independent.}" + "\n"
        + r"\label{fig:runtime_breakdown}" + "\n"
        + r"\end{figure*}" + "\n"
        + FOOTER
    )
    write_both(outdir, "fig18_runtime_breakdown", body.replace(HEADER, "").replace(HEADER_SUBCAP, "").replace(FOOTER, "").strip())


# ══════════════════════════════════════════════════════════════════
#  FIG 19 -- CCSD T1 / T2 DIAGNOSTICS  (multireference character)
#           T1 norm > 0.02 → single-reference methods (HF/CCSD)
#           become unreliable; VQE may be the only trustworthy method.
#           Research value: justifies VQE use case molecule-by-molecule.
# ══════════════════════════════════════════════════════════════════

def fig19_ccsd_diagnostics(cpu, gpu, molecules, outdir):
    """
    Bar chart of CCSD T1 and T2 norms per molecule.
    T1 norm > 0.02 (dashed red line) signals significant multireference
    character → single-reference methods are unreliable → VQE justified.
    Sorted by T1 norm descending so most challenging molecules are leftmost.
    """
    print("\n[Fig 19] CCSD T1/T2 diagnostic (multireference character) …")

    mol_labels, t1_vals, t2_vals = [], [], []
    for mol in molecules:
        refs = cpu[mol]["refs"]
        t1   = refs.get("t1_norm")
        t2   = refs.get("t2_norm")
        if t1 is None:
            continue
        mol_labels.append(display(mol))
        t1_vals.append(t1)
        t2_vals.append(t2 if t2 is not None else 0.0)

    if not mol_labels:
        print("  SKIP -- no T1/T2 data in pkl refs.")
        return

    # sort by T1 descending (most multireference first)
    order = sorted(range(len(mol_labels)), key=lambda i: -t1_vals[i])
    mol_labels = [mol_labels[i] for i in order]
    t1_vals    = [t1_vals[i]    for i in order]
    t2_vals    = [t2_vals[i]    for i in order]

    sym_list = ",".join(f"{{{m}}}" for m in mol_labels)

    def pts(vals):
        return " ".join(f"({{{m}}},{v:.6f})" for m, v in zip(mol_labels, vals))

    # T1 threshold line at 0.02 via pgfplotsextra
    first = f"{{{mol_labels[0]}}}"
    last  = f"{{{mol_labels[-1]}}}"
    t1_thresh_line = (
        r"\pgfplotsextra{%" + "\n"
        f"  \\draw[red!80!black, dashed, line width=1.2pt]\n"
        f"    ({{axis cs:{first},0.02}}-|{{rel axis cs:0,0}}) --\n"
        f"    ({{axis cs:{last}, 0.02}}-|{{rel axis cs:1,0}});\n"
        r"}%" + "\n"
        r"\node[anchor=south west, font=\tiny, red!80!black] at"
        r" (rel axis cs:0.01,0.01) {$T_1 = 0.02$ (MR threshold)};" + "\n"
    )

    body = (
        HEADER
        + r"\begin{figure}[htbp]" + "\n"
        + r"\centering" + "\n"
        + r"\resizebox{\textwidth}{!}{%" + "\n"
        + r"\begin{tikzpicture}" + "\n"
        + r"\begin{axis}[" + "\n"
        + r"  ybar, bar width=8pt," + "\n"
        + r"  width=\textwidth, height=0.45\textwidth," + "\n"
        + r"  enlarge x limits=0.05," + "\n"
        + f"  symbolic x coords={{{sym_list}}}," + "\n"
        + r"  xtick=data," + "\n"
        + r"  x tick label style={rotate=45, anchor=east, font=\small}," + "\n"
        + r"  ymajorgrids=true, grid style={dotted,gray!30}," + "\n"
        + r"  ylabel={CCSD norm}," + "\n"
        + r"  ymin=0," + "\n"
        + r"  legend style={at={(0.99,0.99)}, anchor=north east,"
        + r"legend columns=1, font=\small}," + "\n"
        + "]\n"
        + t1_thresh_line
        + r"\addplot[fill=blue!60,    draw=blue!80]    coordinates {" + pts(t1_vals) + "};\n"
        + r"\addplot[fill=orange!65,  draw=orange!80]  coordinates {" + pts(t2_vals) + "};\n"
        + r"\legend{$\|T_1\|$ norm, $\|T_2\|$ norm}" + "\n"
        + r"\end{axis}" + "\n"
        + r"\end{tikzpicture}" + "\n"
        + r"}" + "\n"
        + r"\caption{CCSD $T_1$ (blue) and $T_2$ (orange) amplitude norms per"
        + r" molecule, sorted by $T_1$ norm (descending)."
        + r" The dashed red line marks $\|T_1\| = 0.02$, the widely used"
        + r" threshold above which single-reference methods (HF, CCSD) become"
        + r" unreliable due to significant multireference character."
        + r" Molecules above this threshold are the strongest candidates for"
        + r" quantum VQE treatment, since classical coupled-cluster theory loses"
        + r" accuracy while VQE's variational ansatz is unaffected by"
        + r" multireference effects.}" + "\n"
        + r"\label{fig:ccsd_diagnostics}" + "\n"
        + r"\end{figure}" + "\n"
        + FOOTER
    )
    write_both(outdir, "fig19_ccsd_diagnostics", body.replace(HEADER, "").replace(HEADER_SUBCAP, "").replace(FOOTER, "").strip())


# ══════════════════════════════════════════════════════════════════
#  FIG 20 -- CIRCUIT COMPLEXITY vs QUBITS
#           Scatter: n_params (UCCSD parameters) and ham_terms
#           (Hamiltonian Pauli terms) vs qubit count.
#           Research value: quantifies quantum resource scaling;
#           shows how circuit depth and measurement cost scale
#           with system size -- key for hardware feasibility.
# ══════════════════════════════════════════════════════════════════

def fig20_circuit_complexity(cpu, gpu, molecules, outdir):
    """
    Two-panel figure:
    Left:  UCCSD circuit parameters vs qubit count (scatter, per molecule)
    Right: Hamiltonian Pauli terms vs qubit count (scatter, per molecule)
    Both panels include a dashed power-law guide line.
    """
    print("\n[Fig 20] Circuit complexity: n_params + ham_terms vs qubits …")

    # Marker SHAPE encodes the molecule, matching the pgfplots scatter plots
    # (make_scatter_plots.py TIKZ_MARKS, assigned in MOL_ORDER) so the two
    # figure families use identical molecule symbols.
    # NB: use OPEN oplus/otimes (not oplus*/otimes*) -- the filled '*'
    # variants hide the inner +/x glyph and render as plain filled circles,
    # colliding with '*' (Methylene). Open versions keep +/x visible.
    TIKZ_MARKS = ["*", "square*", "triangle*", "diamond*", "pentagon*", "o",
                  "square", "triangle", "diamond", "pentagon", "oplus",
                  "otimes", "star"]
    mol_mark = {m: TIKZ_MARKS[i % len(TIKZ_MARKS)] for i, m in enumerate(MOL_ORDER)}
    def _shape(mol):
        return (mol_mark.get(mol, "*"), "")

    # collect points; each carries its molecule and (Ne,No) config
    param_pts, ham_pts = [], []     # (qubits, value, mol, (ne,no))
    configs = set()
    for mol in molecules:
        for r in cpu[mol]["runs"]:
            q, npr, ht = r["qubits"], r["n_params"], r["ham_terms"]
            cfg = (r["ne"], r["no"])
            if q and npr:
                param_pts.append((q, npr, mol, cfg)); configs.add(cfg)
            if q and ht:
                ham_pts.append((q, ht, mol, cfg));   configs.add(cfg)

    if not param_pts:
        print("  SKIP -- no circuit complexity data.")
        return

    # Marker COLOUR encodes active-space size on a red->violet scale,
    # ordered by (Ne,No) so the smallest AS is red and the largest violet.
    cfg_order = sorted(configs)
    n_cfg = max(len(cfg_order), 1)

    def _as_rgb(i, n):
        t = i / max(n - 1, 1)
        r, g, b = colorsys.hsv_to_rgb(0.80 * t, 0.85, 0.90)   # red -> violet
        return round(r, 3), round(g, 3), round(b, 3)

    color_defs, cfg_color = "", {}
    for i, cfg in enumerate(cfg_order):
        cname = f"cfgcol{i}"
        r, g, b = _as_rgb(i, n_cfg)
        color_defs += f"\\definecolor{{{cname}}}{{rgb}}{{{r},{g},{b}}}\n"
        cfg_color[cfg] = cname

    # one (forget-plot) marker per point: shape=molecule, colour=config
    def data_block(pts):
        out = []
        for q, v, mol, cfg in sorted(pts, key=lambda p: (p[2], p[3])):
            mk, mo = _shape(mol)
            opt = f", {mo}" if mo else ""
            out.append(
                f"\\addplot[only marks, color={cfg_color[cfg]}, mark={mk}{opt},"
                f" mark size=2.4pt, forget plot] coordinates {{({q},{v})}};\n"
            )
        return "".join(out)

    param_block = data_block(param_pts)
    ham_block   = data_block(ham_pts)

    # SINGLE legend: molecule shapes only (drawn neutral/black), on ax2.
    def _legend_entry(mol):
        mk, mo = _shape(mol)
        opt = f", {mo}" if mo else ""
        return (f"\\addlegendimage{{only marks, mark={mk}{opt}, black}}\n"
                f"\\addlegendentry{{{display(mol)}}}\n")
    legend_block = "".join(_legend_entry(mol) for mol in molecules)

    body = (
        HEADER
        + color_defs
        + r"\begin{figure}[htbp]" + "\n"
        + r"\centering" + "\n"
        + r"\begin{tikzpicture}" + "\n"
        # Left panel: n_params (no legend)
        + r"\begin{axis}[" + "\n"
        + r"  name=ax1," + "\n"
        + r"  width=0.48\textwidth, height=0.40\textwidth," + "\n"
        + r"  xlabel={Qubit count}," + "\n"
        + r"  ylabel={UCCSD parameters $N_{\mathrm{params}}$}," + "\n"
        + r"  xmode=log, ymode=log," + "\n"
        + r"  grid=both, grid style={dotted,gray!30}," + "\n"
        + r"  tick label style={font=\small}, label style={font=\small}," + "\n"
        + r"  title={(a) Circuit parameters}," + "\n"
        + "]\n"
        + param_block
        + r"\end{axis}" + "\n"
        # Right panel: ham_terms (holds the single shared molecule legend)
        + r"\begin{axis}[" + "\n"
        + r"  name=ax2, at={(ax1.south east)}, anchor=south west," + "\n"
        + r"  xshift=1.2cm," + "\n"
        + r"  width=0.48\textwidth, height=0.40\textwidth," + "\n"
        + r"  xlabel={Qubit count}," + "\n"
        + r"  ylabel={Hamiltonian Pauli terms}," + "\n"
        + r"  xmode=log, ymode=log," + "\n"
        + r"  grid=both, grid style={dotted,gray!30}," + "\n"
        + r"  tick label style={font=\small}, label style={font=\small}," + "\n"
        + r"  legend style={font=\tiny, at={(1.02,1)}, anchor=north west,"
        + r"legend columns=1}," + "\n"
        + r"  title={(b) Hamiltonian complexity}," + "\n"
        + "]\n"
        + ham_block
        + legend_block
        + r"\end{axis}" + "\n"
        + r"\end{tikzpicture}" + "\n"
        + r"\caption{Quantum circuit complexity scaling with qubit count."
        + r" (a)~UCCSD variational parameters $N_{\mathrm{params}}$ and"
        + r" (b)~non-constant qubit-Hamiltonian Pauli terms, both vs qubit"
        + r" count (log--log). Each point is one active-space configuration:"
        + r" marker \emph{shape} identifies the molecule (legend), while marker"
        + r" \emph{colour} encodes the active-space size on a red$\to$violet"
        + r" scale (smallest $(2,3)$ red $\to$ largest $(6,7)$ violet)."
        + r" In panel~(a) $N_{\mathrm{params}}$ depends only on the active space"
        + r" $(N_e^{(a)},N_o^{(a)})$, so all molecules coincide at each"
        + r" configuration. Super-linear scaling of both quantities with qubit"
        + r" count quantifies the quantum resource overhead and motivates GPU"
        + r" acceleration for larger active spaces.}" + "\n"
        + r"\label{fig:circuit_complexity}" + "\n"
        + r"\end{figure}" + "\n"
        + FOOTER
    )
    write_both(outdir, "fig20_circuit_complexity", body.replace(HEADER, "").replace(HEADER_SUBCAP, "").replace(FOOTER, "").strip())

# ══════════════════════════════════════════════════════════════════
#  FIG 21 -- CORRELATION ENERGY RECOVERY (LOG SCALE VERSION)
#           Same η = (E_VQE - E_HF) / (E_CCSD - E_HF) × 100 % as Fig 12,
#           but plotted on a logarithmic y-axis so the full dynamic
#           range across molecule sizes is visible.
#
#           Fig 12 (linear scale) compresses recovery values for large
#           molecules (acenes, nucleobases) into the 0-5% band where
#           they're nearly invisible.  Fig 21 (log scale) spreads them
#           out so every molecule's recovery is readable, and the CCSD
#           100% target line remains visible as a reference ceiling.
# ══════════════════════════════════════════════════════════════════

def fig21_correlation_recovery_log(cpu, gpu, molecules, outdir):
    """
    Same data as Fig 12 but with ymode=log so every molecule's
    recovery fraction is distinguishable, regardless of absolute
    size. Per-molecule subplot: η as percentage of CCSD correlation
    recovered by VQE.  Grouped bars per (Ne,No) config; CPU blue,
    GPU red.  Dashed green line at 100% marks the CCSD target.

    Key differences vs fig12:
    - ymode=log          -- spreads the 0.1 -- 100 % range visually
    - ymin=0.3, ymax=200 -- fixed log window shared across panels so
                            all molecules are on the same scale
    - log origin y=-infty -- bars extend from the visible axis floor
                            (pgfplots-standard for log-scale ybars)
    - Caption adds the size/recovery-fraction relationship insight
      that the linear plot could not convey.
    """
    print("\n[Fig 21] Correlation energy recovery (log scale) …")
    NCOLS = 4
    nrows = (len(molecules) + NCOLS) // NCOLS

    # Fixed log window -- same across all panels for fair comparison.
    YMIN = 0.3     # 0.3% lower bound (below any data point we care about)
    YMAX = 200.0   # 200% upper bound (leaves headroom above 100% target)

    blocks = []
    for mol in molecules:
        refs   = cpu[mol]["refs"]
        E_hf   = refs.get("E_hf")
        E_ccsd = refs.get("E_ccsd")
        pairs  = pair_runs(cpu[mol]["runs"], gpu[mol]["runs"])
        if not pairs or E_hf is None or E_ccsd is None:
            continue

        denom = E_ccsd - E_hf   # < 0
        if abs(denom) < 1e-10:
            continue

        pairs.sort(key=lambda p: (p["ne"], p["no"]))
        sym_coords = [sym(p["ne"], p["no"]) for p in pairs]
        sym_labels = [p["label"]             for p in pairs]
        sym_list   = ",".join(sym_coords)
        lbl_list   = ",".join(sym_labels)

        vc_pts, vg_pts = [], []
        for p in pairs:
            sc    = sym(p["ne"], p["no"])
            eta_c = corr_recovery(p["E_cpu"], E_hf, E_ccsd)
            eta_g = corr_recovery(p["E_gpu"], E_hf, E_ccsd)
            # Clamp to YMIN to avoid log(0) errors when eta is negative
            # or tiny (happens occasionally due to numerical noise for
            # configs where E_VQE > E_HF by a hair). These points will
            # render at the axis floor and are visually negligible.
            if eta_c is not None:
                v = max(eta_c * 100, YMIN * 1.01)
                vc_pts.append(f"({sc},{v:.4f})")
            if eta_g is not None:
                v = max(eta_g * 100, YMIN * 1.01)
                vg_pts.append(f"({sc},{v:.4f})")

        # 100% reference line drawn with explicit coordinates
        first_sym = sym_coords[0]
        last_sym  = sym_coords[-1]
        ref_line = (
            r"\addplot[no marks, green!60!black, dashed, line width=1.2pt]"
            + f" coordinates {{({first_sym},100) ({last_sym},100)}};\n"
        )

        col_idx21 = len(blocks) % NCOLS
        ylabel_21 = "" if col_idx21 == 0 else "  ylabel={},\n"
        blocks.append(
            f"% === {mol} ===\n"
            f"\\nextgroupplot[\n  symbolic x coords={{{sym_list}}},\n"
            f"  xticklabels={{{lbl_list}}},\n"
            f"  xticklabel style={{font=\\tiny,xshift=4.5pt,yshift=-3pt}},\n"
            f"  ymin={YMIN}, ymax={YMAX},\n"
            f"  log origin y=-infty,\n"
            f"{ylabel_21}]\n"
            + (f"\\addplot+[ybar, bar shift=-2pt, fill=blue!60, draw=blue!80]"
               f" coordinates {{{' '.join(vc_pts)}}};\n"
               if vc_pts else "")
            + (f"\\addplot+[ybar, bar shift=+2pt, fill=red!55,  draw=red!80]"
               f"  coordinates {{{' '.join(vg_pts)}}};\n"
               if vg_pts else "")
            + ref_line
            + mol_label_node(mol, display(mol), anchor='north east', pos='0.98,0.98')
            + "\n"
        )

    if not blocks:
        print("  SKIP -- no molecules have both E_hf and E_ccsd.")
        return

    blocks.append(
        r"\nextgroupplot[hide axis, legend style={at={(0.5,0.5)}, anchor=center}]" + "\n"
        r"\addlegendimage{ybar, bar shift=-2pt, fill=blue!60, draw=blue!80}" + "\n"
        r"\addlegendentry{$\eta_{\mathrm{CPU}}$ [\%]}" + "\n"
        r"\addlegendimage{ybar, bar shift=+2pt, fill=red!55,  draw=red!80}" + "\n"
        r"\addlegendentry{$\eta_{\mathrm{GPU}}$ [\%]}" + "\n"
        r"\addlegendimage{no marks, green!60!black, dashed, line width=1.2pt}" + "\n"
        r"\addlegendentry{100\% (CCSD target)}" + "\n"
    )

    body = (
        HEADER
        + r"\begin{figure*}[htbp]" + "\n"
        + r"\centering\resizebox{1.1\textwidth}{!}{%" + "\n"
        + r"\begin{tikzpicture}" + "\n"
        + r"\begin{groupplot}[" + "\n"
        + f"  group style={{group size={NCOLS} by {nrows},"
        + r"horizontal sep=1.4cm,vertical sep=1cm}," + "\n"
        + r"  width=0.27\textwidth, height=0.34\textwidth," + "\n"
        + r"  ybar, bar width=3pt, xtick=data," + "\n"
        + r"  x tick label style={rotate=45, anchor=east}," + "\n"
        + r"  tick label style={font=\scriptsize}, label style={font=\scriptsize}," + "\n"
        + r"  ymajorgrids=true, grid style={dotted,gray!30}," + "\n"
        + r"  ymode=log," + "\n"
        + r"  ylabel={Correlation recovery $\eta$ [\%]}," + "\n"
        + "]\n\n"
        + "\n".join(blocks)
        + r"\end{groupplot}" + "\n"
        + r"\end{tikzpicture}" + "\n"
        + r"}" + "\n"
        + r"\caption{Percentage of CCSD correlation energy recovered by VQE"
        + r" (logarithmic scale): "
        + r"$\eta = (E_{\mathrm{VQE}} - E_{\mathrm{HF}}) / "
        + r"(E_{\mathrm{CCSD}} - E_{\mathrm{HF}}) \times 100\,\%$. "
        + r"CPU and GPU shown in blue and red, respectively; dashed green"
        + r" line marks the 100\,\% CCSD target. Same data as"
        + r" Fig.~\ref{fig:correlation_recovery} but plotted logarithmically"
        + r" so the full dynamic range of recovery fractions is visible."
        + r" Recovery correlates inversely with molecule size: smaller"
        + r" systems (Methylene, Ethylene) recover $\sim$10--20\,\% of"
        + r" their correlation budget within the 14-qubit active space,"
        + r" while larger systems (acenes, nucleobases) recover"
        + r" $\sim$1--3\,\%. This reflects the intrinsic limitation of"
        + r" fixed-size active-space VQE: static correlation within the"
        + r" active space is captured, but dynamic correlation from the"
        + r" inactive orbital manifold requires active-space expansion.}" + "\n"
        + r"\label{fig:correlation_recovery_log}" + "\n"
        + r"\end{figure*}" + "\n"
        + FOOTER
    )
    write_both(outdir, "fig21_correlation_recovery_log",
               body.replace(HEADER, "").replace(HEADER_SUBCAP, "").replace(FOOTER, "").strip())
# ══════════════════════════════════════════════════════════════════
#  NEW INSIGHT FIGURES  A - H
# ══════════════════════════════════════════════════════════════════

TIKZ_COLORS_CYCLE = [
    "blue!80!black", "red!80!black", "teal!80!black", "orange!90!black",
    "violet!80!black", "brown!70!black", "cyan!70!black", "magenta!70!black",
    "olive!80!black",
]

def figA_vqe_casci_gap(cpu, gpu, molecules, outdir):
    """
    Fig A: |E_VQE - E_CASCI| per active-space config per molecule.
    Shows ansatz (UCCSD) quality within each active space.
    Zero gap = VQE reached the exact active-space solution.
    Non-zero = optimizer ran out of evaluations, NOT an ansatz limitation.
    """
    print("\n[Fig A] VQE-CASCI gap (ansatz quality) ...")
    NCOLS = 4
    CHEM_ACC_mHa = 1.6

    for mol in molecules:
        runs = cpu[mol]["runs"]
        E_hf = cpu[mol]["refs"]["E_hf"]
        if not runs:
            print(f"  SKIP {mol} -- no runs")
            continue

        # Sort configs by qubit count
        sorted_runs = sorted(runs, key=lambda r: (r["qubits"], r["ne"], r["no"]))

        # One bar per config: |d_casci| * 1000 mHa
        sym_list = ",".join(f"{r['ne']}-{r['no']}" for r in sorted_runs)
        lbl_list = ",".join(r["label"] for r in sorted_runs)

        gaps = []
        for r in sorted_runs:
            d = r.get("d_casci")
            gaps.append(abs(d * 1000) if d is not None else 0.0)

        pts_conv  = " ".join(f"({r['ne']}-{r['no']},{g:.6f})"
                             for r, g in zip(sorted_runs, gaps) if r.get("success_any"))
        pts_nconv = " ".join(f"({r['ne']}-{r['no']},{g:.6f})"
                             for r, g in zip(sorted_runs, gaps) if not r.get("success_any"))

        gap_max = max(gaps) * 1.2 if max(gaps) > 0 else 0.1
        ymax = max(gap_max, CHEM_ACC_mHa * 1.5)

        body = (
            HEADER
            + r"\begin{figure}[htbp]" + "\n"
            + r"\centering\resizebox{0.7\textwidth}{!}{%" + "\n"
            + r"\begin{tikzpicture}" + "\n"
            + r"\begin{axis}[" + "\n"
            + f"  title={{{display(mol)} -- VQE vs CASCI gap}}," + "\n"
            + f"  symbolic x coords={{{sym_list}}}," + "\n"
            + f"  xticklabels={{{lbl_list}}}," + "\n"
            + r"  xtick=data, x tick label style={rotate=45, anchor=east, font=\small}," + "\n"
            + r"  ylabel={$|E_{\mathrm{VQE}} - E_{\mathrm{CASCI}}|$ [mHa]}," + "\n"
            + r"  xlabel={Active-space config $(N_e^{(a)}, N_o^{(a)})$}," + "\n"
            + f"  ymin=0, ymax={ymax:.4f}," + "\n"
            + r"  ybar, bar width=12pt," + "\n"
            + r"  ymajorgrids=true, grid style={dotted,gray!30}," + "\n"
            + r"  tick label style={font=\small}," + "\n"
            + r"  label style={font=\small}," + "\n"
            + r"  width=0.85\textwidth, height=0.45\textwidth," + "\n"
            + r"]" + "\n"
            # chemical accuracy line
            + f"\\draw[dashed, orange!80!black, thick] ({{rel axis cs:0,0}}|-{{axis cs:{sorted_runs[0]['ne']}-{sorted_runs[0]['no']},{CHEM_ACC_mHa:.2f}}}) -- ({{rel axis cs:1,0}}|-{{axis cs:{sorted_runs[0]['ne']}-{sorted_runs[0]['no']},{CHEM_ACC_mHa:.2f}}});\n"
            + f"\\node[anchor=north east, font=\\scriptsize, orange!80!black] at (rel axis cs:0.98,0.90) {{Chemical accuracy ({CHEM_ACC_mHa} mHa)}};\n"
        )
        if pts_conv:
            body += (
                f"\\addplot+[ybar, fill=teal!60, draw=teal!80] coordinates {{{pts_conv}}};\n"
                + "\\addlegendentry{Converged};\n"
            )
        if pts_nconv:
            body += (
                f"\\addplot+[ybar, fill=orange!60, draw=orange!80] coordinates {{{pts_nconv}}};\n"
                + "\\addlegendentry{Not converged};\n"
            )
        body += (
            r"\end{axis}" + "\n"
            + r"\end{tikzpicture}" + "\n"
            + r"}" + "\n"
            + f"\\caption{{VQE--CASCI gap for {display(mol)}: $|E_{{\\mathrm{{VQE}}}}-E_{{\\mathrm{{CASCI}}}}|$ [mHa] per active-space configuration. "
            + r"Teal = converged; orange = hit evaluation limit. "
            + r"The dashed line marks chemical accuracy (1.6 mHa). "
            + r"Near-zero gaps confirm UCCSD is expressive enough; non-zero gaps reflect optimizer budget, not ansatz error.}"
            + "\n"
            + f"\\label{{fig:vqe_casci_gap_{mol.lower()}}}" + "\n"
            + r"\end{figure}" + "\n"
            + FOOTER
        )
        write_both(outdir, f"figA_vqe_casci_gap_{mol.lower()}", body.replace(HEADER, "").replace(FOOTER, "").strip())
    print("  done.")


def figB_mo_orbital_spectrum(cpu, molecules, outdir):
    """
    Fig B: MO orbital energy spectrum with active-space windows.
    Shows the subset of molecular orbitals included in each active-space config
    and where they sit relative to HF HOMO/LUMO.
    """
    print("\n[Fig B] MO orbital energy spectrum ...")
    for mol in molecules:
        refs = cpu[mol]["refs"]
        runs = cpu[mol]["runs"]
        mo_energy = refs.get("mo_energy", [])
        nocc = refs.get("nocc", 0)
        if not mo_energy or not runs:
            print(f"  SKIP {mol}")
            continue

        mo_energy = list(mo_energy)
        sorted_runs = sorted(runs, key=lambda r: (r["qubits"], r["ne"], r["no"]))

        # Find range of active orbitals across all configs
        all_ncore = [r["ncore"] for r in sorted_runs]
        all_last  = [r["ncore"] + r["no"] - 1 for r in sorted_runs]
        orb_lo = max(0, min(all_ncore) - 2)
        orb_hi = min(len(mo_energy) - 1, max(all_last) + 2)
        orb_range = list(range(orb_lo, orb_hi + 1))

        # Build coordinates for occupied and virtual in range
        occ_pts  = " ".join(f"({i},{mo_energy[i]:.6f})"
                            for i in orb_range if i < nocc)
        virt_pts = " ".join(f"({i},{mo_energy[i]:.6f})"
                            for i in orb_range if i >= nocc)

        # Active space window bars — one \addplot fill per config
        # Use x fill between orbital indices, y range = energy span
        window_blocks = []
        colors_w = ["blue!15", "red!12", "teal!14", "orange!12",
                    "violet!12", "brown!12", "cyan!12", "magenta!12", "olive!12"]
        for idx, r in enumerate(sorted_runs):
            nc   = r["ncore"]
            norb = r["no"]
            x1, x2 = nc - 0.4, nc + norb - 0.6
            y1 = mo_energy[nc] - 0.05
            y2 = mo_energy[min(nc + norb - 1, len(mo_energy)-1)] + 0.05
            col = colors_w[idx % len(colors_w)]
            window_blocks.append(
                f"\\addplot[draw=none, fill={col}] "
                f"coordinates {{({x1},{y1}) ({x2},{y1}) ({x2},{y2}) ({x1},{y2})}} \\closedcycle;\n"
                f"\\node[anchor=south, font=\\tiny, rotate=90] "
                f"at (axis cs:{(x1+x2)/2:.2f},{y2}) "
                f"{{{r['label']}}};\n"
            )

        E_homo = mo_energy[nocc - 1]
        E_lumo = mo_energy[nocc]
        gap_eV = (E_lumo - E_homo) * 27.211

        xmin = orb_lo - 0.5
        xmax = orb_hi + 0.5
        ymin = mo_energy[orb_lo] - 0.1
        ymax = mo_energy[orb_hi] + 0.2

        body = (
            HEADER
            + r"\begin{figure}[htbp]" + "\n"
            + r"\centering\resizebox{0.8\textwidth}{!}{%" + "\n"
            + r"\begin{tikzpicture}" + "\n"
            + r"\begin{axis}[" + "\n"
            + f"  title={{{display(mol)} -- MO energy spectrum \\& active-space windows}}," + "\n"
            + f"  xmin={xmin:.1f}, xmax={xmax:.1f}," + "\n"
            + f"  ymin={ymin:.4f}, ymax={ymax:.4f}," + "\n"
            + r"  xlabel={MO index}," + "\n"
            + r"  ylabel={MO energy [Ha]}," + "\n"
            + r"  ymajorgrids=true, grid style={dotted,gray!30}," + "\n"
            + r"  tick label style={font=\small}, label style={font=\small}," + "\n"
            + r"  width=0.9\textwidth, height=0.48\textwidth," + "\n"
            + r"]" + "\n"
        )
        for wb in window_blocks:
            body += wb
        if occ_pts:
            body += (
                f"\\addplot+[only marks, mark=-, mark size=5pt, blue!70!black, thick, mark options={{solid}}] "
                f"coordinates {{{occ_pts}}};\n"
                + "\\addlegendentry{Occupied MOs};\n"
            )
        if virt_pts:
            body += (
                f"\\addplot+[only marks, mark=-, mark size=5pt, red!70!black, thick, mark options={{solid}}] "
                f"coordinates {{{virt_pts}}};\n"
                + "\\addlegendentry{Virtual MOs};\n"
            )
        # HOMO/LUMO boundary
        body += (
            f"\\draw[dashed, black!50, thick] (axis cs:{nocc - 0.5},{ymin}) -- "
            f"(axis cs:{nocc - 0.5},{ymax});\n"
            + f"\\node[anchor=north west, font=\\scriptsize, rotate=90] "
            + f"at (axis cs:{nocc - 0.5},{(ymin+ymax)/2:.4f}) {{HOMO/LUMO (gap {gap_eV:.2f} eV)}};\n"
        )
        body += (
            r"\end{axis}" + "\n"
            + r"\end{tikzpicture}" + "\n"
            + r"}" + "\n"
            + f"\\caption{{MO energy spectrum for {display(mol)} (cc-pVDZ) in the orbital range spanning all active-space configurations. "
            + r"Blue ticks = occupied MOs; red ticks = virtual MOs. Coloured bands mark the orbital window "
            + r"activated by each $(N_e^{(a)}, N_o^{(a)})$ configuration. "
            + f"The dashed vertical line separates HOMO and LUMO (H--L gap = {gap_eV:.2f} eV).}}"
            + "\n"
            + f"\\label{{fig:mo_spectrum_{mol.lower()}}}" + "\n"
            + r"\end{figure}" + "\n"
            + FOOTER
        )
        write_both(outdir, f"figB_mo_spectrum_{mol.lower()}", body.replace(HEADER, "").replace(FOOTER, "").strip())
    print("  done.")


def figC_convergence_all_configs(cpu, molecules, outdir):
    """
    Fig C: VQE energy convergence traces for ALL active-space configs per molecule.
    Shows landscape difficulty as a function of active-space size:
    small configs converge immediately (CCSD warm-start perfect);
    large configs require hundreds of evaluations.
    """
    print("\n[Fig C] Convergence traces -- all configs per molecule ...")
    NCOLS = 4

    for mol in molecules:
        runs = cpu[mol]["runs"]
        E_hf = cpu[mol]["refs"]["E_hf"]
        if not runs:
            continue

        sorted_runs = sorted(runs, key=lambda r: (r["qubits"], r["ne"], r["no"]))
        blocks = []

        for idx, r in enumerate(sorted_runs):
            col_idx = idx % NCOLS
            ylabel_c = "" if col_idx == 0 else "  ylabel={},\n"

            ec = r.get("econv", [])
            if not ec:
                continue

            nfev_plot = r["nfev_actual"]
            n_pts = len(ec)
            step = max(1, nfev_plot // n_pts)
            xs = [i * step for i in range(n_pts)]

            # Convert to mHa above final energy for visual clarity
            E_final = r["E_vqe"]
            pts = " ".join(f"({xs[i]},{(ec[i]-E_final)*1000:.6f})" for i in range(n_pts))

            col = TIKZ_COLORS_CYCLE[idx % len(TIKZ_COLORS_CYCLE)]
            conv_str = r"converged" if r.get("success_any") else r"not conv."

            blocks.append(
                f"% === {mol} ({r['label']}) ===\n"
                + f"\\nextgroupplot[\n"
                + f"  title={{{r['label']} ({r['qubits']}q, {r['n_params']}p)}},"
                + f"  title style={{font=\\scriptsize}},"
                + f"\n  xmin=0, xmax={nfev_plot},\n"
                + f"  xlabel={{f. evals}}, xlabel style={{font=\\tiny}},\n"
                + f"  {ylabel_c}]\n"
                + f"\\addplot+[{col}, thick, no marks] coordinates {{{pts}}};\n"
                + f"\\draw[dashed, gray!60] (axis cs:0,0) -- (axis cs:{nfev_plot},0);\n"
                + f"\\node[anchor=north east, font=\\tiny] at (rel axis cs:0.98,0.98)"
                + f" {{{r['label']} {conv_str}}};\n\n"
            )

        if not blocks:
            continue

        nrows = math.ceil(len(sorted_runs) / NCOLS)
        body = (
            HEADER
            + r"\begin{figure*}[htbp]" + "\n"
            + r"\centering\resizebox{1.1\textwidth}{!}{%" + "\n"
            + r"\begin{tikzpicture}" + "\n"
            + f"\\begin{{groupplot}}[\n"
            + f"  group style={{group size={NCOLS} by {nrows},"
            + r"horizontal sep=1.4cm,vertical sep=1.4cm}," + "\n"
            + r"  width=0.27\textwidth, height=0.30\textwidth," + "\n"
            + r"  ylabel={$\Delta E$ from final [mHa]}," + "\n"
            + r"  tick label style={font=\tiny}, label style={font=\scriptsize}," + "\n"
            + r"  ymajorgrids=true, grid style={dotted,gray!30}," + "\n"
            + r"]" + "\n"
            + "\n".join(blocks)
            + r"\end{groupplot}" + "\n"
            + r"\end{tikzpicture}" + "\n"
            + r"}" + "\n"
            + f"\\caption{{VQE convergence traces for all active-space configurations of {display(mol)}. "
            + r"$y$-axis: energy above the final VQE value in mHa (lower = better); $x$-axis: function evaluations. "
            + r"Configs where the CCSD warm-start already sits at the minimum appear flat at $\Delta E=0$ from the first evaluation. "
            + r"Larger active spaces (more qubits / parameters) require many evaluations and may not formally converge.}"
            + "\n"
            + f"\\label{{fig:conv_all_{mol.lower()}}}" + "\n"
            + r"\end{figure*}" + "\n"
            + FOOTER
        )
        write_both(outdir, f"figC_convergence_all_{mol.lower()}", body.replace(HEADER, "").replace(FOOTER, "").strip())
    print("  done.")


def figD_quantum_time_distribution(cpu, gpu, molecules, outdir):
    """
    Fig D: Quantum circuit simulation time per evaluation (mean +/- std, min-max).
    Uses PGFPlots error bars. Shows how simulation cost grows with qubit count
    and whether GPU compression is stable (low std = predictable speedup).
    """
    print("\n[Fig D] Quantum time distribution ...")
    NCOLS = 4

    for mol in molecules:
        cpu_runs = sorted(cpu[mol]["runs"], key=lambda r: (r["qubits"], r["ne"], r["no"]))
        gpu_runs_map = {r["key"]: r for r in gpu.get(mol, {}).get("runs", [])}

        if not cpu_runs:
            continue

        sym_list = ",".join(f"c{r['ne']}-{r['no']}" for r in cpu_runs)
        lbl_list = ",".join(r["label"] for r in cpu_runs)

        # CPU: mean, std, min, max
        cpu_mean = " ".join(f"(c{r['ne']}-{r['no']},{r['qt_mean']:.6f})"
                            for r in cpu_runs if r["qt_mean"])
        cpu_err  = " ".join(f"(c{r['ne']}-{r['no']},{r['qt_std']:.6f})"
                            for r in cpu_runs if r["qt_std"])

        # GPU
        gpu_pts  = []
        gpu_err_pts = []
        gpu_sym  = []
        for r in cpu_runs:
            gr = gpu_runs_map.get(r["key"])
            if gr and gr.get("qt_mean"):
                gpu_pts.append(f"(c{r['ne']}-{r['no']},{gr['qt_mean']:.6f})")
                gpu_err_pts.append(f"(c{r['ne']}-{r['no']},{gr['qt_std']:.6f})")
        gpu_mean = " ".join(gpu_pts)

        ymax = max(
            (r["qt_mean"] or 0) + (r["qt_std"] or 0)
            for r in cpu_runs if r["qt_mean"]
        ) * 1.3

        body = (
            HEADER
            + r"\begin{figure}[htbp]" + "\n"
            + r"\centering\resizebox{0.85\textwidth}{!}{%" + "\n"
            + r"\begin{tikzpicture}" + "\n"
            + r"\begin{axis}[" + "\n"
            + f"  title={{{display(mol)} -- quantum eval time per circuit}}," + "\n"
            + f"  symbolic x coords={{{sym_list}}}," + "\n"
            + f"  xticklabels={{{lbl_list}}}," + "\n"
            + r"  xtick=data, x tick label style={rotate=45, anchor=east, font=\small}," + "\n"
            + r"  ylabel={Time per eval [s]}," + "\n"
            + r"  xlabel={Active-space config $(N_e^{(a)}, N_o^{(a)})$}," + "\n"
            + r"  ymode=log," + "\n"
            + r"  ymajorgrids=true, grid style={dotted,gray!30}," + "\n"
            + r"  tick label style={font=\small}, label style={font=\small}," + "\n"
            + r"  width=0.85\textwidth, height=0.48\textwidth," + "\n"
            + r"  error bars/y dir=both, error bars/y explicit," + "\n"
            + r"]" + "\n"
        )
        if cpu_mean:
            body += (
                f"\\addplot+[blue!70, thick, mark=*, mark size=3pt, error bars/.cd, "
                f"y dir=both, y explicit] coordinates {{{cpu_mean}}};\n"
                + "\\addlegendentry{CPU mean};\n"
            )
        if gpu_mean:
            body += (
                f"\\addplot+[red!70, thick, mark=square*, mark size=3pt] "
                f"coordinates {{{gpu_mean}}};\n"
                + "\\addlegendentry{GPU mean};\n"
            )
        body += (
            r"\end{axis}" + "\n"
            + r"\end{tikzpicture}" + "\n"
            + r"}" + "\n"
            + f"\\caption{{Per-evaluation quantum circuit simulation time for {display(mol)}. "
            + r"Blue = CPU; red = GPU. Log scale on $y$-axis. "
            + r"Time grows super-linearly with qubit count (active-space size), "
            + r"confirming that GPU acceleration has the greatest impact on the largest, "
            + r"most chemically important configurations.}}"
            + "\n"
            + f"\\label{{fig:qt_dist_{mol.lower()}}}" + "\n"
            + r"\end{figure}" + "\n"
            + FOOTER
        )
        write_both(outdir, f"figD_qt_dist_{mol.lower()}", body.replace(HEADER, "").replace(FOOTER, "").strip())
    print("  done.")


def figE_parameter_spectrum(cpu, molecules, outdir):
    """
    Fig E: UCCSD theta_opt magnitude spectrum (sorted, log scale).
    For each active-space config of each molecule, plots sorted |theta_i|
    vs parameter rank. Shows parameter sparsity and how many amplitudes
    are physically significant vs near-zero.
    """
    print("\n[Fig E] UCCSD parameter spectrum ...")

    for mol in molecules:
        runs = cpu[mol]["runs"]
        if not runs:
            continue

        sorted_runs = sorted(runs, key=lambda r: (r["qubits"], r["ne"], r["no"]))

        body = (
            HEADER
            + r"\begin{figure}[htbp]" + "\n"
            + r"\centering\resizebox{0.82\textwidth}{!}{%" + "\n"
            + r"\begin{tikzpicture}" + "\n"
            + r"\begin{axis}[" + "\n"
            + f"  title={{{display(mol)} -- UCCSD parameter magnitude spectrum}}," + "\n"
            + r"  xlabel={Parameter rank (sorted by $|\theta|$ descending)}," + "\n"
            + r"  ylabel={$|\theta_{\mathrm{opt}}|$}," + "\n"
            + r"  ymode=log, ymin=1e-5, ymax=1.0," + "\n"
            + r"  xmin=0," + "\n"
            + r"  ymajorgrids=true, grid style={dotted,gray!30}," + "\n"
            + r"  tick label style={font=\small}, label style={font=\small}," + "\n"
            + r"  legend pos=north east, legend style={font=\tiny}," + "\n"
            + r"  width=0.88\textwidth, height=0.50\textwidth," + "\n"
            + r"]" + "\n"
            + r"\draw[dashed, gray!60, thick] ({rel axis cs:0,0}|-{axis cs:0,0.01}) -- ({rel axis cs:1,0}|-{axis cs:0,0.01});" + "\n"
            + r"\node[anchor=north east, font=\tiny, gray!60] at (rel axis cs:0.98,0.25) {sparsity threshold 0.01};" + "\n"
        )
        for idx, r in enumerate(sorted_runs):
            th = sorted([abs(x) for x in r.get("theta_opt", [])], reverse=True)
            if not th:
                continue
            # subsample to <=80 pts for readability
            step = max(1, len(th) // 80)
            pts  = " ".join(f"({i},{th[min(i*step, len(th)-1)]:.6f})"
                            for i in range(min(80, len(th))))
            col = TIKZ_COLORS_CYCLE[idx % len(TIKZ_COLORS_CYCLE)]
            sp_frac = sum(1 for x in th if x < 0.01) / len(th) * 100
            body += (
                f"\\addplot+[{col}, thick, mark=none] coordinates {{{pts}}};\n"
                + f"\\addlegendentry{{{r['label']} ({len(th)}p, {sp_frac:.0f}\\% sparse)}};\n"
            )
        body += (
            r"\end{axis}" + "\n"
            + r"\end{tikzpicture}" + "\n"
            + r"}" + "\n"
            + f"\\caption{{UCCSD optimised parameter magnitudes $|\\theta_i|$ sorted in descending order for {display(mol)}. "
            + r"Each line corresponds to one active-space configuration (number of parameters in legend). "
            + r"The dashed line at $|\theta|=0.01$ separates physically significant from near-zero amplitudes. "
            + r"High sparsity ($>$60\% below threshold) indicates most UCCSD gates contribute negligibly, "
            + r"motivating parameter pruning and circuit compression.}}"
            + "\n"
            + f"\\label{{fig:param_spectrum_{mol.lower()}}}" + "\n"
            + r"\end{figure}" + "\n"
            + FOOTER
        )
        write_both(outdir, f"figE_param_spectrum_{mol.lower()}", body.replace(HEADER, "").replace(FOOTER, "").strip())
    print("  done.")


def figF_cycle_improvement(cpu, molecules, outdir):
    """
    Fig F: Cycle-by-cycle energy improvement (mHa) for challenging configs.
    Shows diminishing returns of COBYLA restarts.
    Only plots configs with > 3 cycles.
    """
    print("\n[Fig F] Cycle-by-cycle improvement ...")

    for mol in molecules:
        runs = cpu[mol]["runs"]
        hard_runs = [r for r in runs if r.get("cycles", 0) > 3]
        if not hard_runs:
            continue

        hard_runs = sorted(hard_runs, key=lambda r: r["qubits"])
        ncols = min(len(hard_runs), 2)
        nrows = math.ceil(len(hard_runs) / ncols)

        blocks = []
        for idx, r in enumerate(hard_runs):
            col_idx = idx % ncols
            ylabel_f = "" if col_idx == 0 else "  ylabel={},\n"

            cdE = r.get("cycle_dE", [])
            crt = r.get("cycle_rt", [])
            n_c = len(cdE)
            if n_c == 0:
                continue

            sym = ",".join(f"C{i+1}" for i in range(n_c))
            pts_pos = " ".join(f"(C{i+1},{cdE[i]:.6f})" for i in range(n_c) if cdE[i] >= 0)
            pts_neg = " ".join(f"(C{i+1},{abs(cdE[i]):.6f})" for i in range(n_c) if cdE[i] < 0)
            ymax = max(abs(x) for x in cdE) * 1.3

            blocks.append(
                f"% === {r['label']} ===\n"
                + f"\\nextgroupplot[\n"
                + f"  title={{{r['label']} ({r['qubits']}q, {n_c} cycles)}},\n"
                + f"  title style={{font=\\scriptsize}},\n"
                + f"  symbolic x coords={{{sym}}},\n"
                + f"  xticklabels={{{sym}}},\n"
                + f"  xtick=data, x tick label style={{font=\\tiny}},\n"
                + f"  ymin=0, ymax={ymax:.4f},\n"
                + f"  {ylabel_f}]\n"
            )
            if pts_pos:
                blocks.append(
                    f"\\addplot+[ybar, fill=teal!60, draw=teal!80] coordinates {{{pts_pos}}};\n"
                    + "\\addlegendentry{Improvement};\n"
                )
            if pts_neg:
                blocks.append(
                    f"\\addplot+[ybar, fill=red!40, draw=red!60] coordinates {{{pts_neg}}};\n"
                    + "\\addlegendentry{Regression};\n"
                )
            avg_t = sum(crt) / len(crt) if crt else 0
            blocks.append(
                f"\\node[anchor=north east, font=\\tiny] at (rel axis cs:0.98,0.98) "
                f"{{~{avg_t:.0f}s/cycle}};\n\n"
            )

        if not blocks:
            continue

        body = (
            HEADER
            + r"\begin{figure}[htbp]" + "\n"
            + r"\centering\resizebox{0.85\textwidth}{!}{%" + "\n"
            + r"\begin{tikzpicture}" + "\n"
            + f"\\begin{{groupplot}}[\n"
            + f"  group style={{group size={ncols} by {nrows},"
            + r"horizontal sep=1.6cm,vertical sep=1.2cm}," + "\n"
            + r"  width=0.45\textwidth, height=0.40\textwidth," + "\n"
            + r"  ybar, bar width=12pt," + "\n"
            + r"  ylabel={$\Delta E$ per cycle [mHa]}," + "\n"
            + r"  xlabel={COBYLA restart cycle}," + "\n"
            + r"  ymajorgrids=true, grid style={dotted,gray!30}," + "\n"
            + r"  tick label style={font=\small}, label style={font=\scriptsize}," + "\n"
            + r"]" + "\n"
            + "".join(blocks)
            + r"\end{groupplot}" + "\n"
            + r"\end{tikzpicture}" + "\n"
            + r"}" + "\n"
            + f"\\caption{{Cycle-by-cycle energy improvement (mHa) for multi-cycle configurations of {display(mol)}. "
            + r"Each bar shows the energy gain $\Delta E$ achieved in one COBYLA restart cycle (600 evaluations). "
            + r"Teal bars = improvement; red bars = slight regression (numerical noise). "
            + r"The rapidly diminishing improvement demonstrates the landscape hardness and motivates "
            + r"adaptive early-stopping criteria.}}"
            + "\n"
            + f"\\label{{fig:cycle_improve_{mol.lower()}}}" + "\n"
            + r"\end{figure}" + "\n"
            + FOOTER
        )
        write_both(outdir, f"figF_cycle_improve_{mol.lower()}", body.replace(HEADER, "").replace(FOOTER, "").strip())
    print("  done.")


def figG_hamiltonian_sparsity(cpu, molecules, outdir):
    """
    Fig G: Hamiltonian sparsity -- n_Pauli_terms / 4^n_qubits vs n_qubits.
    Two-panel figure: left = absolute n_terms, right = fill fraction.
    All molecules shown together (scatter, colored by molecule).
    Demonstrates super-linear qubit overhead but extreme Hamiltonian sparsity.
    """
    print("\n[Fig G] Hamiltonian sparsity ...")

    # Gather all data points across molecules
    mol_colors = TIKZ_COLORS_CYCLE

    terms_pts = {}   # mol -> list of (qubits, n_terms)
    fill_pts  = {}   # mol -> list of (qubits, fill%)
    for i, mol in enumerate(molecules):
        t_list, f_list = [], []
        for r in cpu[mol]["runs"]:
            nq = r["qubits"]
            nt = r["ham_terms"]
            fill = nt / (4 ** nq) * 100 if nq > 0 else 0
            t_list.append(f"({nq},{nt})")
            f_list.append(f"({nq},{fill:.6f})")
        terms_pts[mol] = " ".join(t_list)
        fill_pts[mol]  = " ".join(f_list)

    body = (
        HEADER
        + r"\begin{figure}[htbp]" + "\n"
        + r"\centering\resizebox{1.0\textwidth}{!}{%" + "\n"
        + r"\begin{tikzpicture}" + "\n"
        + r"\begin{groupplot}[" + "\n"
        + r"  group style={group size=2 by 1, horizontal sep=1.8cm}," + "\n"
        + r"  width=0.52\textwidth, height=0.42\textwidth," + "\n"
        + r"  xmode=log, ymode=log," + "\n"
        + r"  ymajorgrids=true, xmajorgrids=true, grid style={dotted,gray!30}," + "\n"
        + r"  tick label style={font=\small}, label style={font=\small}," + "\n"
        + r"]" + "\n"
        # Panel 1: n_terms
        + r"\nextgroupplot[" + "\n"
        + r"  title={Non-constant Pauli terms}," + "\n"
        + r"  xlabel={Qubit count $n_q$}," + "\n"
        + r"  ylabel={$N_{\mathrm{Pauli}}$}," + "\n"
        + r"]" + "\n"
    )
    for i, mol in enumerate(molecules):
        col = mol_colors[i % len(mol_colors)]
        disp = display(mol)
        body += (
            f"\\addplot+[only marks, mark=*, {col}, mark size=3pt] "
            f"coordinates {{{terms_pts[mol]}}};\n"
            + f"\\addlegendentry{{{disp}}};\n"
        )
    body += (
        # Panel 2: fill fraction
        r"\nextgroupplot[" + "\n"
        + r"  title={Hamiltonian fill fraction $N_{\mathrm{Pauli}}/4^{n_q}$}," + "\n"
        + r"  xlabel={Qubit count $n_q$}," + "\n"
        + r"  ylabel={Fill fraction [\%]}," + "\n"
        + r"]" + "\n"
    )
    for i, mol in enumerate(molecules):
        col = mol_colors[i % len(mol_colors)]
        body += (
            f"\\addplot+[only marks, mark=*, {col}, mark size=3pt] "
            f"coordinates {{{fill_pts[mol]}}};\n"
        )
    body += (
        r"\end{groupplot}" + "\n"
        + r"\end{tikzpicture}" + "\n"
        + r"}" + "\n"
        + r"\caption{Hamiltonian sparsity across all molecules and active-space configurations. "
        + r"\textit{Left}: absolute number of non-constant Pauli terms $N_{\mathrm{Pauli}}$ vs qubit count (log--log). "
        + r"\textit{Right}: fill fraction $N_{\mathrm{Pauli}}/4^{n_q}$ vs qubit count. "
        + r"While $N_{\mathrm{Pauli}}$ grows polynomially with qubits, the fill fraction "
        + r"drops from $\sim$3\% at 6 qubits to $<$0.001\% at 14 qubits --- "
        + r"demonstrating that molecular Hamiltonians are extremely sparse "
        + r"despite the exponentially growing Hilbert space.}"
        + "\n"
        + r"\label{fig:ham_sparsity}" + "\n"
        + r"\end{figure}" + "\n"
        + FOOTER
    )
    write_both(outdir, "figG_hamiltonian_sparsity", body.replace(HEADER, "").replace(FOOTER, "").strip())
    print("  done.")


def figH_optimizer_efficiency(cpu, molecules, outdir):
    """
    Fig H: Optimizer efficiency -- fraction of function evaluations
    that strictly improve the energy, per config per molecule.
    Shows that COBYLA spends 95%+ of time exploring (not improving),
    and that the CCSD warm-start already places converged runs at the minimum
    (0% improvement means the starting point IS the optimum).
    """
    print("\n[Fig H] Optimizer efficiency ...")
    NCOLS = 4

    for mol in molecules:
        runs = cpu[mol]["runs"]
        if not runs:
            continue

        sorted_runs = sorted(runs, key=lambda r: (r["qubits"], r["ne"], r["no"]))
        sym_list = ",".join(f"{r['ne']}-{r['no']}" for r in sorted_runs)
        lbl_list = ",".join(r["label"] for r in sorted_runs)

        pts_eff = []
        for r in sorted_runs:
            nfev = r.get("nfev_actual", r.get("nfev", 1))
            nim  = r.get("n_improve", 0)
            eff  = nim / max(nfev - 1, 1) * 100
            pts_eff.append(f"({r['ne']}-{r['no']},{eff:.4f})")

        body = (
            HEADER
            + r"\begin{figure}[htbp]" + "\n"
            + r"\centering\resizebox{0.78\textwidth}{!}{%" + "\n"
            + r"\begin{tikzpicture}" + "\n"
            + r"\begin{axis}[" + "\n"
            + f"  title={{{display(mol)} -- COBYLA optimizer efficiency}}," + "\n"
            + f"  symbolic x coords={{{sym_list}}}," + "\n"
            + f"  xticklabels={{{lbl_list}}}," + "\n"
            + r"  xtick=data, x tick label style={rotate=45, anchor=east, font=\small}," + "\n"
            + r"  ylabel={Improving evaluations [\%]}," + "\n"
            + r"  xlabel={Active-space config $(N_e^{(a)}, N_o^{(a)})$}," + "\n"
            + r"  ymin=0, ymax=12," + "\n"
            + r"  ybar, bar width=12pt," + "\n"
            + r"  ymajorgrids=true, grid style={dotted,gray!30}," + "\n"
            + r"  tick label style={font=\small}, label style={font=\small}," + "\n"
            + r"  width=0.85\textwidth, height=0.45\textwidth," + "\n"
            + r"]" + "\n"
            + f"\\addplot+[ybar, fill=violet!50, draw=violet!80] coordinates {{{' '.join(pts_eff)}}};\n"
            + r"\node[anchor=north west, font=\scriptsize, text width=4cm] at (rel axis cs:0.02,0.98)"
            + r" {0\% = CCSD warm-start already at minimum};"
            + "\n"
            + r"\end{axis}" + "\n"
            + r"\end{tikzpicture}" + "\n"
            + r"}" + "\n"
            + f"\\caption{{COBYLA optimizer efficiency for {display(mol)}: "
            + r"percentage of function evaluations that strictly decrease the energy. "
            + r"Converged runs (0\%) indicate the CCSD-sliced warm-start already places "
            + r"the optimizer at the variational minimum --- all evaluations are spent "
            + r"confirming convergence, not descending. "
            + r"Non-converged large active spaces reach only $\sim$4--5\% efficiency, "
            + r"motivating adaptive step sizes and better gradient estimators.}}"
            + "\n"
            + f"\\label{{fig:opt_eff_{mol.lower()}}}" + "\n"
            + r"\end{figure}" + "\n"
            + FOOTER
        )
        write_both(outdir, f"figH_opt_efficiency_{mol.lower()}", body.replace(HEADER, "").replace(FOOTER, "").strip())
    print("  done.")


# ══════════════════════════════════════════════════════════════════
#  CLI & MAIN
# ══════════════════════════════════════════════════════════════════

def cli():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Generate LaTeX/PGFPlots .tex files from VQE benchmark pkl data.")
    p.add_argument("--cpu_dir", default="results/pkl_results/cpu_pkl_results",
                   help="Directory with CPU .pkl files  [cpu_pkl_results]")
    p.add_argument("--gpu_dir", default="results/pkl_results/gpu_pkl_results",
                   help="Directory with GPU .pkl files  [gpu_pkl_results]")
    p.add_argument("--out",     default="tex_out",
                   help="Output directory for .tex files  [tex_out]")
    p.add_argument("--figs",   nargs="*",
                   default=["1","2","3","4","5","6","7","8","9",
                            "10","11","12","13","14","15","16",
                            "17","18","19","20","21","22","23","24","25",
                            "A","B","C","D","E","F","G","H"],
                   help="Which figures to generate  [1..20, A..H]")
    p.add_argument("--no_table", action="store_true",
                   help="Skip terminal summary table")
    return p.parse_args()


def main():
    args = cli()
    figs = set(args.figs)
    os.makedirs(args.out, exist_ok=True)

    print("=" * 66)
    print("  VQE LaTeX Generator -- CPU vs GPU + HF/VQE/CCSD Benchmarking")
    print("=" * 66)
    print(f"  cpu_dir : {args.cpu_dir}")
    print(f"  gpu_dir : {args.gpu_dir}")
    print(f"  output  : {args.out}")
    # print(f"  figures : {sorted(figs, key=lambda x: int(x))}")

    cpu, gpu, molecules = load_data(args.cpu_dir, args.gpu_dir)

    if not molecules:
        sys.exit("\nERROR: no matched molecules found. "
                 "Check --cpu_dir / --gpu_dir paths.")

    print("\n── Writing provenance README ───────────────────────────────")
    write_readme(args.cpu_dir, args.gpu_dir, args.out, cpu, gpu, molecules, args)

    if not args.no_table:
        print_summary(cpu, gpu, molecules)

    print("\n── Generating LaTeX files ─────────────────────────────────")
    dispatch = {
        # ── GPU acceleration ──────────────────────────────────────
        "1" : fig1_energy_vs_norb,
        "2" : fig2_energy_per_config,
        "3" : fig3_runtime_per_config,
        "4" : fig4_speedup_per_config,
        "5" : fig5_runtime_vs_qubits,
        "6" : fig6_speedup_vs_norb,
        "7" : fig7_runtime_vs_norb,
        "8" : fig8_speedup_heatmap,
        "9" : fig9_speedup_combined,
        # ── HF / VQE / CCSD accuracy analysis ────────────────────
        "10": fig10_energy_comparison,       # 4-way energy bar (HF,VQE_CPU,VQE_GPU,CCSD)
        "11": fig11_energy_error_vs_ccsd,    # |E_VQE - E_CCSD| + HF baseline
        "12": fig12_correlation_recovery,    # η = correlation recovery %
        "13": fig13_energy_landscape,        # HF/CCSD lines + VQE scatter vs No
        "14": fig14_vqe_ccsd_delta,          # Signed ΔE (mHa) + chem accuracy line
        "15": fig15_performance_summary,     # 3-panel subfigure: speedup/error/normalised
        "16": fig16_vqe_vs_hf_only,          # VQE-HF bars only, no CCSD reference
        # ── New research analyses (from pkl internals) ────────────
        "17": fig17_convergence_curves,      # VQE energy vs iteration convergence
        "18": fig18_runtime_breakdown,       # Stacked: quantum/optimizer/overhead
        "19": fig19_ccsd_diagnostics,        # T1/T2 norms -- multireference character
        "20": fig20_circuit_complexity,      # n_params + ham_terms vs qubits scatter
        "21": fig21_correlation_recovery_log, # log-scale version of fig12
        "22": fig22_accuracy_summary,         # CCSD accuracy + normalised panels
        "23": fig23_speedup_histogram,        # pooled GPU-speedup histogram (all configs)
        "24": fig24_vqe_casci_accuracy,       # |E_VQE - E_CASCI| true accuracy (CPU vs GPU)
        "25": fig25_vqe_ccsd_accuracy,        # |E_VQE - E_CCSD| residual (truncation-dominated)

    }
    num_keys = [k for k in dispatch if k.isdigit()]
    str_keys = [k for k in dispatch if not k.isdigit()]
    for key in sorted(num_keys, key=int) + sorted(str_keys):
        if key in figs:
            dispatch[key](cpu, gpu, molecules, args.out)

    export_csv(cpu, gpu, molecules, args.out)
    export_preamble(args.out)

    print("\n" + "=" * 66)
    print(f"  Done -- files written to '{args.out}/'")
    print()
    print("  Two versions of every figure are generated:")
    print("    figN_xxx.tex        standalone -- compile with pdflatex directly")
    print("    figN_xxx_embed.tex  embeddable -- paste or \\input{} into your doc")
    print()
    print("  To include figures in your main .tex document:")
    print("  1. In your preamble (once):")
    print("       \\input{tex_out/vqe_preamble.tex}")
    print("  2. For each figure:")
    print("       \\input{tex_out/figN_xxx_embed.tex}")
    print("=" * 66)


if __name__ == "__main__":
    main()