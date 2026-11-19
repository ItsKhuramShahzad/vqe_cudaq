"""Paper version of the bond-type active-space figure (single-column width, Fig. 3 style).

Two files per molecule, plus suggested captions:
  <molecule>_active_space_levels.pdf   (a) energy order vs bond type at two sizes, levels coloured
                                        by bond type (solid occupied, dashed virtual), chosen
                                        orbitals framed; virtual levels that no column uses are
                                        left out (marked with a vertical ellipsis)
  <molecule>_active_space_orbitals.pdf (b) canonical MOs next to the localized orbitals they are
                                        made of, with the share of each bond type
  <molecule>_captions.txt

    python scripts/bond_type_paper_figure.py --molecule Ethylene --size 6,8
Colours: Okabe-Ito (colour-blind safe). Needs scikit-image, like scripts/hf_orbitals.py.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import hf_orbitals as hfo  # noqa: E402
from bond_type_figure import energy_order_space, pieces, pretty, size_of  # noqa: E402
from vqe_cudaq.bond_type_search import ensure_sizes, typed_orbitals  # noqa: E402
from vqe_cudaq.chosen_spaces import bond_type_space, bond_type_spaces, orbital_index, orbital_name  # noqa: E402
from vqe_cudaq.hamiltonian import run_scf  # noqa: E402
from vqe_cudaq.molecules import molecules  # noqa: E402
from vqe_cudaq.orbital_selection import (NON_VALENCE, PLANAR_LIMIT, _localized, _pi_scores,  # noqa: E402
                                         antibond_orbitals, molecular_plane)

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
                     "axes.linewidth": 0.8, "pdf.fonttype": 42})

# Okabe-Ito colours by kind of bond; occupied solid, virtual dashed
KIND_COLOUR = {"pi": "#0072B2", "xx": "#009E73", "xh": "#D55E00", "lone": "#CC79A7",
               "nb": "#E69F00", "non-valence": "#8C8C8C"}


def bond_kind(t):
    if t == NON_VALENCE:
        return "non-valence"
    k = hfo.kind(t).rstrip("*")
    return "nb" if k == "nb" else k


def colour_of(t):
    return KIND_COLOUR[bond_kind(t)]


def style_of(t, occupied):
    return "-" if occupied else ((0, (1.0, 1.2)) if t == NON_VALENCE else (0, (3.2, 1.6)))


def pair_name(t):
    """'C–C π' and 'C–C π*' share one legend entry: 'C–C π / π*'."""
    if t == NON_VALENCE:
        return "outside valence (virtual)"
    base = t.rstrip("*")
    tail = base.split(" ")[-1]
    return f"{base} / {tail}*" if "lone pair" not in base else base


# ── (a) ───────────────────────────────────────────────────────────────────────

def levels_figure(path, table, nocc, ncore, columns, e_best, sizes):
    used = sorted(set().union(*[c["active"] for c in columns]))
    virt = [k for k in used if k >= nocc]
    shown = list(range(ncore, nocc)) + virt
    y, pos, breaks = {}, 0.0, []
    for i, k in enumerate(shown):
        if i and k >= nocc and shown[i - 1] >= nocc and k - shown[i - 1] > 1:
            pos += 0.8
            breaks.append(pos - 0.9)                        # levels left out here (middle of the gap)
        elif i and k == nocc:
            pos += 0.55                                     # HOMO-LUMO gap
        y[k] = pos
        pos += 1.0
    top = max(y.values())

    n = len(columns)
    fig = plt.figure(figsize=(3.45, 4.35))
    ax = fig.add_axes((0.17, 0.13, 0.81, 0.66))
    step, half, frame = 1.0, 0.33, 0.42
    for c, col in enumerate(columns):
        x = step * c + (0.25 if c >= 2 else 0)              # a small gap between the two sizes
        col["x"] = x
        for k in shown:
            t = table.loc[k, "type"]
            ax.plot([x - half, x + half], [y[k]] * 2, color=colour_of(t), lw=1.7,
                    linestyle=style_of(t, k < nocc), dash_capstyle="butt", solid_capstyle="butt", zorder=2)
            if k < nocc:
                for dx, up in ((-0.1, True), (0.1, False)):
                    a, b = (y[k] - 0.3, y[k] + 0.3) if up else (y[k] + 0.3, y[k] - 0.3)
                    ax.annotate("", xy=(x + dx, b), xytext=(x + dx, a), zorder=3,
                                arrowprops=dict(arrowstyle="-|>", lw=0.6, color=colour_of(table.loc[k, "type"]),
                                                mutation_scale=5, shrinkA=0, shrinkB=0))
        for yb in breaks:
            ax.text(x, yb, "⋮", ha="center", va="center", fontsize=6.5, color="#555555")
        chosen = sorted(col["active"], key=lambda k: y[k])
        runs, start = [], chosen[0]
        for a, b in zip(chosen, chosen[1:] + [None]):
            if b is None or b - a > 1:                     # a skipped orbital (shown or not) splits the frame
                runs.append((start, a))
                start = b
        for lo, hi in runs:
            ax.add_patch(Rectangle((x - frame, y[lo] - 0.42), 2 * frame, y[hi] - y[lo] + 0.84, fill=False,
                                   lw=0.9, edgecolor="black", zorder=4))
        ratio = (col["E_corr"] - e_best) / abs(e_best)
        ax.text(x, top + 0.9, col["rule"], ha="center", va="bottom", fontsize=7.5)
        ax.text(x, -0.9, f"{col['E_corr']:.1f}", ha="center", va="top", fontsize=7.5)
        ax.text(x, -1.85, f"{ratio:.3f}", ha="center", va="top", fontsize=7.5,
                fontweight="bold" if ratio < 0.25 else "normal")
    for i, (e, o) in enumerate(sizes):                      # one size above each pair of columns
        x0, x1 = columns[2 * i]["x"], columns[2 * i + 1]["x"]
        ax.text((x0 + x1) / 2, top + 2.0, f"({e},{o})", ha="center", va="bottom", fontsize=8.5)
        ax.plot([x0 - 0.35, x1 + 0.35], [top + 1.85] * 2, color="black", lw=0.6)
    ax.text(-0.62, -0.9, r"$E_\mathrm{corr}$ (mHa)", ha="right", va="top", fontsize=7.5)
    ax.text(-0.62, -1.85, "ratio", ha="right", va="top", fontsize=7.5)
    ax.set_yticks([y[k] for k in shown], [pretty(orbital_name(k, nocc)) for k in shown], fontsize=7)
    ax.tick_params(axis="y", length=2, pad=2)
    ax.set_xticks([])
    ax.set_xlim(-0.55, columns[-1]["x"] + 0.55)
    ax.set_ylim(-0.7, top + 0.7)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)

    # legend on top: bond types (colour), occupied/virtual (line style), active space (frame)
    types = [table.loc[k, "type"] for k in shown]
    order = ["pi", "lone", "xx", "xh", "nb", "non-valence"]
    entries = {}
    for t in sorted(types, key=lambda t: (order.index(bond_kind(t)), t)):
        entries.setdefault(pair_name(t), t)
    handles = [Line2D([0], [0], color=colour_of(t), lw=1.7) for t in entries.values()]
    labels = list(entries)
    handles += [Line2D([0], [0], color="black", lw=1.2), Line2D([0], [0], color="black", lw=1.2, linestyle=(0, (3.2, 1.6))),
                Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", lw=0.9)]
    labels += ["occupied", "virtual", "active space"]
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=7, frameon=False, handlelength=2.0,
               columnspacing=1.0, handletextpad=0.5, bbox_to_anchor=(0.55, 1.0))
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=600)
    plt.close(fig)


# ── (b) ───────────────────────────────────────────────────────────────────────

def picture(fig, rect, mol, grid, psi, iso, bonds):
    ax = fig.add_axes(rect, projection="3d", computed_zorder=False)
    surface = hfo.isosurface(ax, psi, min(iso, 0.6 * np.abs(psi).max()), grid, np.array([-0.4, -0.5, 0.77]))
    for i, j in bonds:
        ax.plot(*np.c_[grid.atoms[i], grid.atoms[j]], color="#334155", lw=0.8, zorder=2)
    for a in range(mol.natm):
        s = mol.atom_symbol(a)
        ax.scatter(*grid.atoms[a], s=4 if s == "H" else 9, color=hfo.ATOM_COLOUR[s], depthshade=False, zorder=3)
    middle = (grid.atoms.max(0) + grid.atoms.min(0)) / 2
    half = max(max(grid.atoms.max(0) - grid.atoms.min(0)) / 2 + 1.0, np.abs(surface - middle).max())
    ax.set(xlim=(middle[0] - half, middle[0] + half), ylim=(middle[1] - half, middle[1] + half),
           zlim=(middle[2] - half, middle[2] + half))
    ax.set_box_aspect((1, 1, 1), zoom=1.5)
    ax.view_init(elev=48, azim=-68)
    ax.set_axis_off()
    ax.set_rasterized(True)


def orbitals_figure(path, mf, table, nocc, ncore, examples, iso, spacing, padding):
    mol, nmo = mf.mol, mf.mo_coeff.shape[1]
    planar = molecular_plane(mol)[2] <= PLANAR_LIMIT
    normal = molecular_plane(mol)[1]
    pi = _pi_scores(mol, mf.mo_coeff, normal) if planar else np.zeros(nmo)
    _, bnames, boverlap, ibo, _ = _localized(mf, ncore, pi)
    anames, aoverlap, livvo, _ = antibond_orbitals(mf, normal, planar)
    loc = dict(bond_names=np.asarray(bnames), bond_overlap=boverlap, ibo=ibo,
               anti_names=np.asarray(anames), anti_overlap=aoverlap, livvo=livvo)
    grid = hfo.Grid(mol, spacing, padding)
    bonds = hfo.find_bonds(mol)

    rows = len(examples)
    fig = plt.figure(figsize=(3.45, 1.12 * rows + 0.25))
    h = 1.0 / (rows + 0.25)
    w, left = 0.24, 0.04
    for r, k in enumerate(examples):
        y0 = 1 - (r + 1) * h
        t = table.loc[k, "type"]
        picture(fig, (left, y0, w, h * 0.78), mol, grid, grid.orbital(mf.mo_coeff[:, k]), iso, bonds)
        fig.text(left + w / 2, y0 + h * 0.80, f"{pretty(orbital_name(k, nocc))} → {t}", ha="center", va="bottom",
                 fontsize=7.5, color=colour_of(t), fontweight="bold")
        x = left + w
        for i, (bond, share, count, coeff) in enumerate(pieces(k, table, nocc, ncore, loc)):
            fig.text(x + 0.03, y0 + h * 0.39, "=" if i == 0 else "+", fontsize=11, ha="center", va="center")
            picture(fig, (x + 0.06, y0, w, h * 0.78), mol, grid, grid.orbital(coeff), iso, bonds)
            label = f"{share:.2f} {bond}" + (f" (1 of {count})" if count > 1 else "")
            fig.text(x + 0.06 + w / 2, y0 + h * 0.80, label, ha="center", va="bottom", fontsize=7,
                     color=colour_of(bond))
            x += 0.06 + w
    fig.savefig(path + ".pdf", dpi=600)
    fig.savefig(path + ".png", dpi=600)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--molecule", default="Ethylene")
    p.add_argument("--basis", default="cc-pVDZ")
    p.add_argument("--bond-type-dir", default=os.path.join("analysis", "bond_type_casci_hpc"))
    p.add_argument("--size", default="6,8", help="first pair of columns; the second pair is the size of E_best")
    p.add_argument("--examples", nargs="*", default=None, help="MOs for (b), e.g. HOMO HOMO-2 LUMO")
    p.add_argument("--iso", type=float, default=0.04)
    p.add_argument("--spacing", type=float, default=0.15)
    p.add_argument("--padding", type=float, default=4.0)
    p.add_argument("--outdir", default=os.path.join("analysis", "figures", "bond_type_selection", "paper"))
    args = p.parse_args()

    name = args.molecule
    mf = run_scf(molecules[name], args.basis)
    nocc, nmo = mf.mol.nelectron // 2, mf.mo_coeff.shape[1]
    table = typed_orbitals(mf)
    ncore = table.attrs["ncore"]
    size = size_of(args.size)
    ensure_sizes(args.bond_type_dir, name, mf, [size])
    every = bond_type_spaces(args.bond_type_dir, name)
    e_best = float(every["E_corr_mHa"].min())
    best_size = (int(every.iloc[0]["nele"]), int(every.iloc[0]["norb"]))
    corr = dict(zip(every["active"], every["E_corr_mHa"]))
    columns, sizes = [], []
    for sz in dict.fromkeys([size, best_size]):
        eo = energy_order_space(args.bond_type_dir, name, *sz)
        bt = bond_type_space(args.bond_type_dir, name, *sz, nocc, nmo)
        if eo is None:
            continue
        sizes.append(sz)
        columns.append(dict(rule="energy", active=eo[0], E_corr=eo[1]))
        columns.append(dict(rule="bond type", active=bt["active_orbitals"],
                            E_corr=corr[" ".join(map(str, bt["active_orbitals"]))], label=bt["label"]))

    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.join(args.outdir, name)
    levels_figure(base + "_active_space_levels", table, nocc, ncore, columns, e_best, sizes)

    if args.examples:
        examples = [orbital_index(e, nocc) for e in args.examples]
    else:
        best = columns[1]["active"]
        second = lambda k: sorted(table.loc[k, "bond_shares"].values())[-2] if len(table.loc[k, "bond_shares"]) > 1 else 0
        examples = list(dict.fromkeys([nocc - 1, max((k for k in best if k < nocc), key=second), nocc]))
    orbitals_figure(base + "_active_space_orbitals", mf, table, nocc, ncore, examples, args.iso,
                    args.spacing, args.padding)

    # suggested captions, with the numbers of this molecule
    pairs = "; ".join(f"({e}e,{o}o): energy order {columns[2 * i]['E_corr']:.1f} mHa, bond type "
                      f"{columns[2 * i + 1]['E_corr']:.1f} mHa" for i, (e, o) in enumerate(sizes))
    caption_a = (
        f"Active-space selection by bond type for {name} (RHF/{args.basis}). For each size "
        f"(N_e^(a),N_o^(a)), the energy-ordered space (left) and the best bond-type space (right) are shown. "
        f"Horizontal segments are the molecular orbitals (core not shown), coloured by the bond type with the largest "
        f"share (solid: occupied, dashed: virtual); dotted grey: virtual orbitals with no antibond share above 0.1, made "
        f"mainly of the basis functions beyond the minimal (valence) basis; arrows mark "
        f"spin-up and spin-down electrons; black rectangles delimit the active orbitals, which may skip levels in "
        f"the bond-type spaces. Virtual orbitals not used by any space are omitted (⋮). Below each column: the "
        f"CASCI correlation energy E_corr = E_CASCI − E_HF and the selection ratio (E_corr − E_best)/|E_best|, "
        f"where E_best = {e_best:.1f} mHa is the best E_corr of all spaces of the molecule; a space is selected "
        f"when the ratio is below 0.25 (bold). {pairs}.")
    caption_b = (
        f"Origin of the bond types in {name}: canonical molecular orbitals (left) written as their largest "
        f"contributions of localized orbitals, intrinsic bond orbitals (IBOs) for occupied and valence virtual "
        f"orbitals for empty orbitals. The number is the share of each bond type (squared overlap, summed over "
        f"equivalent bonds); the largest share gives the type of the orbital. Isosurfaces ±{args.iso} a.u.; red and "
        f"blue: positive and negative sign of the orbital (each localized orbital drawn with the sign that matches "
        f"its MO).")
    with open(base + "_captions.txt", "w") as fh:
        fh.write("(a) " + caption_a + "\n\n(b) " + caption_b + "\n")
    print(f"{base}_active_space_levels.pdf, {base}_active_space_orbitals.pdf, {base}_captions.txt")


if __name__ == "__main__":
    main()
