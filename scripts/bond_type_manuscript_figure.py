"""Manuscript figure: energy order vs bond type, two molecules side by side (double-column width).

For each molecule and each size (default (6e,8o) and (8e,10o)) two columns: the energy-ordered
space and the best bond-type space of that size. Every orbital level is split into pieces by its
bond shares (squared overlaps with IBOs / LIVVOs), coloured by bond family; the active orbitals
are framed; levels that no column uses are left out (vertical ellipsis). Below each column the
CASCI correlation energy and the selection ratio (bold when below the threshold).

    python scripts/bond_type_manuscript_figure.py --molecules Ethylene Cytosine --sizes 6,8 8,10
Writes analysis/figures/bond_type_selection/manuscript/bond_type_selection.pdf/.png and a captions file.
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vqe_cudaq.bond_type_search import ALL, csv_paths, ensure_sizes, typed_orbitals  # noqa: E402
from vqe_cudaq.chosen_spaces import bond_type_space, bond_type_spaces, orbital_name  # noqa: E402
from vqe_cudaq.hamiltonian import run_scf  # noqa: E402
from vqe_cudaq.molecules import molecules  # noqa: E402
from vqe_cudaq.orbital_selection import BOND_SHARE_MIN, NON_VALENCE  # noqa: E402

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
                     "axes.linewidth": 0.8, "pdf.fonttype": 42})

# Okabe-Ito colours, one per bond family (bond and antibond share the colour; arrows mark occupied)
FAMILY_COLOUR = {"π": "#0072B2", "lone pair": "#CC79A7", "X–X σ": "#009E73", "X–H σ": "#D55E00",
                 "outside valence": "#8C8C8C", "other": "#DADADA"}
FAMILY_LABEL = {"π": "π / π*", "lone pair": "lone pair", "X–X σ": "X–X σ / σ*", "X–H σ": "X–H σ / σ*",
                "outside valence": "outside valence (virtual)", "other": f"other shares (each < {BOND_SHARE_MIN})"}
DISPLAY = {"NH2-": "NH₂⁻"}
SCALE = 1.0                     # font and line scale (set by --scale)


def family(bond):
    if bond == NON_VALENCE:
        return "outside valence"
    if "lone pair" in bond:
        return "lone pair"
    if "π" in bond:
        return "π"
    return "X–H σ" if "–H" in bond else "X–X σ"


def pieces(table, k):
    """(family, share) of MO k: shares >= BOND_SHARE_MIN summed by family, largest first; the rest as 'other'."""
    sums = {}
    for bond, w in table.loc[k, "bond_shares"].items():
        if w >= BOND_SHARE_MIN:
            sums[family(bond)] = sums.get(family(bond), 0) + w
    out = sorted(sums.items(), key=lambda t: -t[1])
    rest = 1 - sum(w for _, w in out)
    return out + ([("other", rest)] if rest > 0.005 else [])


def pretty(name):
    return name.replace("-", "−")


def energy_order_space(folder, molecule, nele, norb):
    main = pd.read_csv(csv_paths(folder, molecule)[0], encoding="utf-8-sig")
    row = main[(main["occupied_types"] == ALL) & (main["virtual_types"] == ALL)
               & (main["nele"] == nele) & (main["norb"] == norb)]
    return None if row.empty else ([int(k) for k in row["active"].iloc[0].split()], float(row["E_corr_mHa"].iloc[0]))


def molecule_data(name, basis, folder, sizes):
    mf = run_scf(molecules[name], basis)
    nocc, nmo = mf.mol.nelectron // 2, mf.mo_coeff.shape[1]
    table = typed_orbitals(mf)
    ensure_sizes(folder, name, mf, sizes)                     # searched only if missing
    every = bond_type_spaces(folder, name)
    e_best = float(every["E_corr_mHa"].min())
    corr = dict(zip(every["active"], every["E_corr_mHa"]))
    columns, used_sizes = [], []
    for nele, norb in sizes:
        eo = energy_order_space(folder, name, nele, norb)
        if eo is None:
            continue
        bt = bond_type_space(folder, name, nele, norb, nocc, nmo)
        used_sizes.append((nele, norb))
        columns.append(dict(rule="energy", active=eo[0], E_corr=eo[1]))
        columns.append(dict(rule="bond type", active=bt["active_orbitals"],
                            E_corr=corr[" ".join(map(str, bt["active_orbitals"]))], label=bt["label"]))
    return dict(name=name, table=table, nocc=nocc, ncore=table.attrs["ncore"], columns=columns,
                sizes=used_sizes, e_best=e_best)


def layout(d):
    """y position of every shown level, and where levels are left out."""
    nocc, ncore = d["nocc"], d["ncore"]
    used = sorted(set().union(*[c["active"] for c in d["columns"]]))
    deepest = min(used)
    shown = list(range(deepest, nocc)) + [k for k in used if k >= nocc]
    y, pos, breaks = {}, 0.0, []
    if deepest > ncore:                                       # occupied valence levels left out below
        breaks.append(-0.65)
    for i, k in enumerate(shown):
        if i and k >= nocc and shown[i - 1] >= nocc and k - shown[i - 1] > 1:
            pos += 0.8
            breaks.append(pos - 0.9)
        elif i and k == nocc:
            pos += 0.55
        y[k] = pos
        pos += 1.0
    return shown, y, breaks


def panel(ax, d, letter, threshold):
    table, nocc = d["table"], d["nocc"]
    shown, y, breaks = layout(d)
    top = max(y.values())
    half, frame = 0.34, 0.42
    xs = []
    for c, col in enumerate(d["columns"]):
        x = c + 0.3 * (c // 2)                                # a small gap between the sizes
        xs.append(x)
        for k in shown:
            start = x - half
            for fam, w in pieces(table, k):
                ax.plot([start, start + 2 * half * w], [y[k]] * 2, color=FAMILY_COLOUR[fam], lw=1.9 * SCALE,
                        solid_capstyle="butt", zorder=2)
                start += 2 * half * w
            if k < nocc:
                for dx, up in ((-0.1, True), (0.1, False)):
                    a, b = (y[k] - 0.3, y[k] + 0.3) if up else (y[k] + 0.3, y[k] - 0.3)
                    ax.annotate("", xy=(x + dx, b), xytext=(x + dx, a), zorder=3,
                                arrowprops=dict(arrowstyle="-|>", lw=0.55 * SCALE, color="#1f2937", mutation_scale=4.5 * SCALE,
                                                shrinkA=0, shrinkB=0))
        for yb in breaks:
            ax.text(x, yb, "⋮", ha="center", va="center", fontsize=6.5 * SCALE, color="#555555")
        chosen = sorted(col["active"])
        runs, start_k = [], chosen[0]
        for a, b in zip(chosen, chosen[1:] + [None]):
            if b is None or b - a > 1:                        # a skipped orbital splits the frame
                runs.append((start_k, a))
                start_k = b
        for lo, hi in runs:
            ax.add_patch(Rectangle((x - frame, y[lo] - 0.42), 2 * frame, y[hi] - y[lo] + 0.84, fill=False,
                                   lw=0.85, edgecolor="black", zorder=4))
        ratio = (col["E_corr"] - d["e_best"]) / abs(d["e_best"])
        ax.text(x, top + 0.8, col["rule"], ha="center", va="bottom", fontsize=7.5 * SCALE)
        ax.text(x, -1.25, f"{col['E_corr']:.1f}", ha="center", va="top", fontsize=7.5 * SCALE)
        ax.text(x, -2.25, f"{ratio:.3f}", ha="center", va="top", fontsize=7.5 * SCALE,
                fontweight="bold" if ratio < threshold else "normal")
    for i, (e, o) in enumerate(d["sizes"]):
        x0, x1 = xs[2 * i], xs[2 * i + 1]
        ax.text((x0 + x1) / 2, top + 1.95, f"({e}e,{o}o)", ha="center", va="bottom", fontsize=8.5 * SCALE)
        ax.plot([x0 - 0.36, x1 + 0.36], [top + 1.8] * 2, color="black", lw=0.6)
    ax.text(-0.6, -1.25, r"$E_\mathrm{corr}$ (mHa)", ha="right", va="top", fontsize=7.5 * SCALE)
    ax.text(-0.6, -2.25, "ratio", ha="right", va="top", fontsize=7.5 * SCALE)
    ax.set_yticks([y[k] for k in shown], [pretty(orbital_name(k, nocc)) for k in shown], fontsize=7 * SCALE)
    ax.tick_params(axis="y", length=2, pad=2)
    ax.set_xticks([])
    ax.set_xlim(-0.55, xs[-1] + 0.55)
    ax.set_ylim(-0.9, top + 0.7)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    return {fam for k in shown for fam, _ in pieces(table, k)}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--molecules", nargs="+", default=["Ethylene", "Cytosine"])
    p.add_argument("--sizes", nargs="+", default=["6,8", "8,10"])
    p.add_argument("--basis", default="cc-pVDZ")
    p.add_argument("--bond-type-dir", default=os.path.join("analysis", "bond_type_casci_hpc"))
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--width", type=float, default=None,
                   help="figure width in inches (default 7.0 side by side, 6.3 stacked)")
    p.add_argument("--stack", action="store_true",
                   help="molecules one below the other (single-column pages, e.g. the Supporting Information)")
    p.add_argument("--scale", type=float, default=None, help="font and line scale (default 1.0, stacked 1.15)")
    p.add_argument("--name", default=None, help="output file name without extension")
    p.add_argument("--outdir", default=os.path.join("analysis", "figures", "bond_type_selection", "manuscript"))
    args = p.parse_args()

    sizes = [tuple(int(v) for v in s.split(",")) for s in args.sizes]
    data = [molecule_data(m, args.basis, args.bond_type_dir, sizes) for m in args.molecules]
    global SCALE
    SCALE = args.scale or (1.15 if args.stack else 1.0)
    width = args.width or (6.3 if args.stack else 7.0)
    n = len(data)
    families = set()
    if args.stack:                                       # one molecule per row, full width
        legend_h, gap = 0.5 * SCALE, 0.12
        heights = [1.0 * SCALE + 0.15 * SCALE * len(layout(d)[0]) for d in data]
        height = legend_h + sum(heights) + gap * (n - 1)
        fig = plt.figure(figsize=(width, height))
        top = height - legend_h
        for i, (d, h) in enumerate(zip(data, heights)):
            low = top - h
            ax_bottom, ax_top = low + 0.42 * SCALE, top - 0.55 * SCALE
            ax = fig.add_axes((0.16, ax_bottom / height, 0.82, (ax_top - ax_bottom) / height))
            families |= panel(ax, d, "abcdefgh"[i], args.threshold)
            fig.text(0.01, (top - 0.12 * SCALE) / height, f"({'abcdefgh'[i]})  {DISPLAY.get(d['name'], d['name'])}",
                     ha="left", va="center", fontsize=9.5 * SCALE, fontweight="bold")
            top = low - gap
    else:
        rows = max(len(layout(d)[0]) for d in data)
        height = 0.95 + 0.165 * rows
        fig = plt.figure(figsize=(width, height))
        for i, d in enumerate(data):
            left = 0.085 + i / n
            bottom, top_margin = 0.42 / height, 0.95 / height
            ax = fig.add_axes((left, bottom, 1 / n - 0.105, 1 - bottom - top_margin))
            families |= panel(ax, d, "abcdefgh"[i], args.threshold)
            fig.text(i / n + 0.01, 1 - 0.47 / height, f"({'abcdefgh'[i]})  {DISPLAY.get(d['name'], d['name'])}",
                     ha="left", va="center", fontsize=9.5, fontweight="bold")

    order = ["π", "lone pair", "X–X σ", "X–H σ", "outside valence", "other"]
    handles = [Line2D([0], [0], color=FAMILY_COLOUR[f], lw=2.2 * SCALE) for f in order if f in families]
    labels = [FAMILY_LABEL[f] for f in order if f in families]
    handles.append(Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", lw=0.85))
    labels.append("active orbitals")
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), fontsize=7.5 * SCALE, frameon=False,
               handlelength=2.0, columnspacing=1.4, handletextpad=0.5, bbox_to_anchor=(0.5, 1.0))

    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.join(args.outdir, args.name or ("bond_type_selection_si" if args.stack else "bond_type_selection"))
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=600)
    plt.close(fig)

    def numbers(d):
        parts = []
        for i, (e, o) in enumerate(d["sizes"]):
            a, b = d["columns"][2 * i], d["columns"][2 * i + 1]
            parts.append(f"({e}e,{o}o) {a['E_corr']:.1f} vs {b['E_corr']:.1f} mHa")
        return f"{DISPLAY.get(d['name'], d['name'])}: " + ", ".join(parts) + f"; E_best = {d['e_best']:.1f} mHa"
    caption = (
        "Active-space selection by bond type. For each molecule and each active-space size (N_e^(a),N_o^(a)), "
        "the energy-ordered space (left) and the best bond-type space of the same size (right) are compared. "
        "Horizontal segments are the canonical Hartree–Fock molecular orbitals (RHF/" + args.basis + "; core "
        "orbitals frozen and not shown), from the lowest (bottom) to the highest orbital shown; arrows mark the "
        "two electrons of each occupied orbital. Each segment is split by the share of each bond family in the "
        "orbital: the squared overlap with the intrinsic bond orbitals (occupied) or the localized valence "
        "virtual orbitals (virtual), summed over bonds of one family; 'outside valence' is the part of a virtual "
        "orbital outside the minimal valence space, and shares below " + f"{BOND_SHARE_MIN}" + " are shown together. "
        "Black rectangles delimit the active orbitals; in the bond-type spaces they may skip orbitals. Orbitals "
        "used by no space are omitted (⋮). Below each column: the CASCI correlation energy E_corr = E_CASCI − E_HF "
        "and the selection ratio (E_corr − E_best)/|E_best|, where E_best is the most negative E_corr of all "
        "active spaces of the molecule; spaces with a ratio below " + f"{args.threshold:g}" + " (bold) are "
        "selected. " + "; ".join(numbers(d) for d in data) + ".")
    with open(base + "_caption.txt", "w") as fh:
        fh.write(caption + "\n")
    print(f"{base}.pdf/.png ({width:.1f} x {height:.2f} in), {base}_caption.txt")


if __name__ == "__main__":
    main()
