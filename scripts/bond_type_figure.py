"""Figure: how the active orbitals are chosen by bond type.

(a) Level diagram in the style of the paper's active-space figure: the valence occupied and the
    needed virtual orbitals, coloured by bond type, for three spaces: the energy-ordered space of
    one size (one block), the best bond-type space of the same size (chosen orbitals framed, levels
    may be skipped) and the best bond-type space of a smaller size. E_corr and the selection ratio
    are written under each column.
(b) Where the types come from: canonical MOs next to the localized orbitals they are made of
    (intrinsic bond orbitals for occupied MOs, valence virtual orbitals for empty ones), with the
    share of each bond type; the largest share gives the type.

    python scripts/bond_type_figure.py --molecule Ethylene --size 6,8 --compare-size 2,2
Needs scikit-image (marching cubes), like scripts/hf_orbitals.py.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import hf_orbitals as hfo  # noqa: E402
from vqe_cudaq.bond_type_search import ALL, csv_paths, ensure_sizes, typed_orbitals  # noqa: E402
from vqe_cudaq.chosen_spaces import bond_type_space, bond_type_spaces, orbital_name  # noqa: E402
from vqe_cudaq.hamiltonian import run_scf  # noqa: E402
from vqe_cudaq.molecules import molecules  # noqa: E402
from vqe_cudaq.orbital_selection import (NON_VALENCE, PLANAR_LIMIT, _localized, _pi_scores,  # noqa: E402
                                         antibond_orbitals, molecular_plane)


def pretty(name):
    return name.replace("-", "−")


def size_of(text):
    e, o = (int(v) for v in text.split(","))
    return e, o


def energy_order_space(folder, molecule, nele, norb):
    """Orbitals and E_corr of the energy-ordered space (type sets "all"/"all") of one size."""
    main = pd.read_csv(csv_paths(folder, molecule)[0], encoding="utf-8-sig")
    row = main[(main["occupied_types"] == ALL) & (main["virtual_types"] == ALL)
               & (main["nele"] == nele) & (main["norb"] == norb)]
    if row.empty:
        return None
    return [int(k) for k in row["active"].iloc[0].split()], float(row["E_corr_mHa"].iloc[0])


# ── (a) level diagram ─────────────────────────────────────────────────────────

SPREAD = 1.45                   # distance between the columns of the level diagram


def level_diagram(ax, table, nocc, ncore, columns, colours, e_best):
    top = max(max(c["active"]) for c in columns)
    shown = list(range(ncore, nocc)) + list(range(nocc, top + 1))
    gap = 0.8                                                    # space between occupied and virtual
    y = {k: i + (gap if k >= nocc else 0) for i, k in enumerate(shown)}
    half, frame = 0.34, 0.44
    for c, col in enumerate(columns):
        x = SPREAD * c
        for k in shown:
            colour = colours.get(table.loc[k, "type"], "#9ca3af")
            ax.plot([x - half, x + half], [y[k], y[k]], color=colour, lw=2.4, solid_capstyle="butt", zorder=2)
            if k < nocc:                                         # two electrons
                for dx, up in ((-0.12, True), (0.12, False)):
                    y0, y1 = (y[k] - 0.36, y[k] + 0.36) if up else (y[k] + 0.36, y[k] - 0.36)
                    ax.annotate("", xy=(x + dx, y1), xytext=(x + dx, y0), zorder=3,
                                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#111827", mutation_scale=7,
                                                shrinkA=0, shrinkB=0))
        # frame the chosen orbitals; neighbouring chosen levels share one frame
        chosen = sorted(y[k] for k in col["active"])
        runs, start = [], chosen[0]
        for a, b in zip(chosen, chosen[1:] + [None]):
            if b is None or b - a > 1.01:
                runs.append((start, a))
                start = b
        for lo, hi in runs:
            ax.add_patch(Rectangle((x - frame, lo - 0.42), 2 * frame, hi - lo + 0.84, fill=False,
                                   lw=1.3, edgecolor="black", zorder=4))
        ratio = (col["E_corr"] - e_best) / abs(e_best)
        ax.text(x, max(y.values()) + 1.3, col["title"], ha="center", va="bottom", fontsize=8.5,
                fontweight="bold", linespacing=1.25)
        ax.text(x, -1.4, f"$E_\\mathrm{{corr}}$ = {col['E_corr']:.1f} mHa\nratio = {ratio:.3f}"
                + ("  (passes)" if ratio < 0.25 else ""), ha="center", va="top", fontsize=7.5, linespacing=1.35)
    ax.set_yticks([y[k] for k in shown], [pretty(orbital_name(k, nocc)) for k in shown], fontsize=6.5)
    ax.set_xticks([])
    ax.set_xlim(-0.75, SPREAD * (len(columns) - 1) + 0.75)
    ax.set_ylim(-0.9, max(y.values()) + 0.9)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_ylabel("Molecular orbital (lowest at the bottom; core not shown)", fontsize=8)
    return [table.loc[k, "type"] for k in shown]


# ── (b) orbital pictures ──────────────────────────────────────────────────────

def draw_orbital(fig, rect, mol, grid, psi, iso, bonds, title, colour):
    ax = fig.add_axes(rect, projection="3d", computed_zorder=False)
    light = np.array([-0.4, -0.5, 0.77])
    surface = hfo.isosurface(ax, psi, min(iso, 0.6 * np.abs(psi).max()), grid, light)
    for i, j in bonds:
        ax.plot(*np.c_[grid.atoms[i], grid.atoms[j]], color="#334155", lw=1.0, zorder=2)
    for a in range(mol.natm):
        s = mol.atom_symbol(a)
        ax.scatter(*grid.atoms[a], s=8 if s == "H" else 16, color=hfo.ATOM_COLOUR[s], depthshade=False, zorder=3)
    middle = (grid.atoms.max(0) + grid.atoms.min(0)) / 2
    extent = grid.atoms.max(0) - grid.atoms.min(0)
    half = max(max(extent) / 2 + 1.2, np.abs(surface - middle).max() + 0.1)
    ax.set(xlim=(middle[0] - half, middle[0] + half), ylim=(middle[1] - half, middle[1] + half),
           zlim=(middle[2] - half, middle[2] + half))
    ax.set_box_aspect((1, 1, 1), zoom=1.45)
    ax.view_init(elev=48, azim=-68)
    ax.set_axis_off()
    ax.set_rasterized(True)
    x, y, w, h = rect
    fig.text(x + w / 2, y + h + 0.002, title, ha="center", va="bottom", fontsize=7, color=colour,
             fontweight="bold", linespacing=1.2)


def pieces(k, table, nocc, ncore, loc):
    """The bond types of MO k with share >= 0.1, largest first: (name, share, count, coefficients)."""
    shares = {n: w for n, w in table.loc[k, "bond_shares"].items() if n != NON_VALENCE and w >= 0.1}
    if k < nocc:
        names, overlap, coeffs, column = loc["bond_names"], loc["bond_overlap"], loc["ibo"], k - ncore
    else:
        names, overlap, coeffs, column = loc["anti_names"], loc["anti_overlap"], loc["livvo"], k - nocc
    out = []
    for name, share in sorted(shares.items(), key=lambda t: -t[1])[:2]:
        members = np.flatnonzero(names == name)
        best = members[np.argmax(overlap[members, column] ** 2)]
        sign = np.sign(overlap[best, column]) or 1.0          # same colours as the MO
        out.append((name, share, len(members), sign * coeffs[:, best]))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--molecule", default="Ethylene")
    p.add_argument("--basis", default="cc-pVDZ")
    p.add_argument("--bond-type-dir", default=os.path.join("analysis", "bond_type_casci_hpc"))
    p.add_argument("--size", default="6,8", help="size of the energy-order and best bond-type columns")
    p.add_argument("--compare-size", default="best",
                   help="third column: 'best' = the best space of all sizes (E_best), or a size such as 2,2")
    p.add_argument("--examples", nargs="*", default=None,
                   help="MOs for part (b), e.g. HOMO HOMO-2 LUMO (default: HOMO, the most mixed chosen MO, LUMO)")
    p.add_argument("--iso", type=float, default=0.04)
    p.add_argument("--spacing", type=float, default=0.2)
    p.add_argument("--padding", type=float, default=4.0)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--outdir", default=os.path.join("analysis", "figures", "bond_type_selection"))
    args = p.parse_args()

    name = args.molecule
    mf = run_scf(molecules[name], args.basis)
    mol, nocc = mf.mol, mf.mol.nelectron // 2
    nmo = mf.mo_coeff.shape[1]
    table = typed_orbitals(mf)
    ncore = table.attrs["ncore"]
    size = size_of(args.size)
    compare = None if args.compare_size == "best" else size_of(args.compare_size)
    ensure_sizes(args.bond_type_dir, name, mf, [size] + ([compare] if compare else []))   # search if missing

    every = bond_type_spaces(args.bond_type_dir, name)
    e_best = float(every["E_corr_mHa"].min())
    best = bond_type_space(args.bond_type_dir, name, *size, nocc, nmo)
    if compare is None:                                           # the E_best space
        top = every.iloc[0]
        compare = (int(top["nele"]), int(top["norb"]))
        small = bond_type_space(args.bond_type_dir, name, *compare, nocc, nmo)
        third = f"Best of all sizes\n({compare[0]}e,{compare[1]}o)"
    else:
        small = bond_type_space(args.bond_type_dir, name, *compare, nocc, nmo)
        third = f"Bond type, best\n({compare[0]}e,{compare[1]}o)"
    corr = dict(zip(every["active"], every["E_corr_mHa"]))
    key = lambda s: " ".join(map(str, s["active_orbitals"]))
    columns = []
    for sz, space, title in ((size, best, f"Bond type, best\n({size[0]}e,{size[1]}o)"),
                             (compare, small, third.replace("Best of all sizes", "Bond type, best\nof all sizes"))):
        eo = energy_order_space(args.bond_type_dir, name, *sz)
        if eo:                                                # the usual choice of the same size
            columns.append(dict(title=f"Energy order\n({sz[0]}e,{sz[1]}o)", active=eo[0], E_corr=eo[1]))
        columns.append(dict(title=title, active=space["active_orbitals"], E_corr=corr[key(space)]))

    colours = hfo.palette(set(table["type"]) - {"core"})
    fig = plt.figure(figsize=(7.2, 10.2))
    ax = fig.add_axes((0.13, 0.43, 0.60, 0.50))
    types_shown = level_diagram(ax, table, nocc, ncore, columns, colours, e_best)
    fig.text(0.02, 0.975, "(a)", fontsize=11, fontweight="bold")
    # legend of the types and of the frames
    legend = hfo.legend_order(set(types_shown))
    for i, t in enumerate(legend):
        yy = 0.88 - 0.028 * i
        fig.lines.append(plt.Line2D([0.755, 0.79], [yy, yy], transform=fig.transFigure, color=colours[t], lw=2.4))
        fig.text(0.80, yy, t, va="center", fontsize=7.5)
    yy = 0.88 - 0.028 * len(legend) - 0.01
    fig.patches.append(Rectangle((0.755, yy - 0.008), 0.035, 0.016, transform=fig.transFigure, fill=False,
                                 edgecolor="black", lw=1.2))
    fig.text(0.80, yy, "active orbitals", va="center", fontsize=7.5)
    skipped = [k for k in range(min(best["active_orbitals"]), nocc) if k not in best["active_orbitals"]]
    def lines(ks, per=3):
        names = [pretty(orbital_name(k, nocc)) for k in ks]
        return "\n  ".join(" ".join(names[i:i + per]) for i in range(0, len(names), per))
    occ_names = lines([k for k in best["active_orbitals"] if k < nocc])
    vir_names = lines([k for k in best["active_orbitals"] if k >= nocc])
    note = (f"Best ({size[0]}e,{size[1]}o) by bond type:\n  {occ_names}\n  {vir_names}"
            + (f"\n  ({', '.join(pretty(orbital_name(k, nocc)) for k in skipped)} skipped)" if skipped else "")
            + f"\n\n$E_\\mathrm{{best}}$ = {e_best:.1f} mHa\n(best $E_\\mathrm{{corr}}$ of all sizes)"
            + "\n\nratio = ($E_\\mathrm{corr} - E_\\mathrm{best}$) / |$E_\\mathrm{best}$|\npasses when ratio < 0.25")
    fig.text(0.755, yy - 0.04, note, va="top", fontsize=6.8, linespacing=1.4)

    # ── (b) ──
    fig.text(0.02, 0.385, "(b)", fontsize=11, fontweight="bold")
    fig.text(0.08, 0.385, "Where a type comes from: a canonical MO and the localized orbitals it is made of "
             "(IBOs for occupied, LIVVOs for virtual MOs)", fontsize=7.5, va="baseline")
    planar = molecular_plane(mol)[2] <= PLANAR_LIMIT
    normal = molecular_plane(mol)[1]
    pi = _pi_scores(mol, mf.mo_coeff, normal) if planar else np.zeros(nmo)
    _, bnames, boverlap, ibo, _ = _localized(mf, ncore, pi)
    anames, aoverlap, livvo, _ = antibond_orbitals(mf, normal, planar)
    loc = dict(bond_names=np.asarray(bnames), bond_overlap=boverlap, ibo=ibo,
               anti_names=np.asarray(anames), anti_overlap=aoverlap, livvo=livvo)
    if args.examples:
        from vqe_cudaq.chosen_spaces import orbital_index
        examples = [orbital_index(e, nocc) for e in args.examples]
    else:
        occ_chosen = [k for k in best["active_orbitals"] if k < nocc]
        second = lambda k: sorted(table.loc[k, "bond_shares"].values())[-2] if len(table.loc[k, "bond_shares"]) > 1 else 0
        mixed = max(occ_chosen, key=second)
        examples = list(dict.fromkeys([nocc - 1, mixed, nocc]))
    grid = hfo.Grid(mol, args.spacing, args.padding)
    bonds = hfo.find_bonds(mol)
    row_h, pic_w = 0.10, 0.17
    for r, k in enumerate(examples):
        y0 = 0.262 - r * (row_h + 0.022)
        parts = pieces(k, table, nocc, ncore, loc)
        mo_title = f"{pretty(orbital_name(k, nocc))} (MO)"
        draw_orbital(fig, (0.04, y0, pic_w, row_h), mol, grid, grid.orbital(mf.mo_coeff[:, k]), args.iso,
                     bonds, mo_title, "#111827")
        x = 0.04 + pic_w
        for i, (bond, share, count, coeff) in enumerate(parts):
            fig.text(x + 0.012, y0 + row_h / 2, "=" if i == 0 else "+", fontsize=13, ha="center", va="center")
            label = f"{share:.2f} × {bond}" + (f"\n(one of {count} {'IBOs' if k < nocc else 'LIVVOs'})" if count > 1 else "")
            draw_orbital(fig, (x + 0.025, y0, pic_w, row_h), mol, grid, grid.orbital(coeff), args.iso,
                         bonds, label, colours.get(bond, "#111827"))
            x += 0.025 + pic_w
        rest = 1 - sum(s for _, s, _, _ in parts)
        what = "outside valence" if k >= nocc else "other bonds"
        fig.text(x + 0.01, y0 + row_h / 2,
                 f"→ type: {table.loc[k, 'type']}" + (f"\n({what} {rest:.2f})" if rest > 0.02 else ""),
                 fontsize=8, va="center", color=colours.get(table.loc[k, "type"], "#111827"), fontweight="bold",
                 linespacing=1.3)
    fig.text(0.04, 0.004, f"{name}, RHF/{args.basis}; isosurfaces ±{args.iso} a.u., red +, blue −: sign of the orbital.\n"
             f"Share = squared overlap with the localized orbitals of one bond type, summed over equal bonds; "
             f"largest share = type.", fontsize=6.3, color="#475569", linespacing=1.3)

    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.join(args.outdir, f"{name}_bond_type_selection")
    fig.savefig(base + ".pdf", dpi=args.dpi)
    fig.savefig(base + ".png", dpi=args.dpi)
    print(f"{base}.pdf / .png  ({', '.join(c['title'].replace(chr(10), ' ') for c in columns)}; "
          f"examples {', '.join(orbital_name(k, nocc) for k in examples)})")


if __name__ == "__main__":
    main()
