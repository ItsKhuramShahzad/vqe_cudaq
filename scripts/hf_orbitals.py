"""Hartree-Fock level diagram and orbital isosurfaces of each molecule, core orbitals excluded.

For each molecule, one PDF: the level diagram (each line split by its bond shares), five tables
that lead from the canonical orbitals to their bond shares (atoms, IBOs and IAOs, IBOs inside each
orbital, the result), and one isosurface per orbital (all occupied valence orbitals plus the first
virtual ones). Also CSV tables, a Molden file with every orbital and, with --cube, one Gaussian
cube file per plotted orbital. Needs scikit-image for the isosurfaces.

    python scripts/hf_orbitals.py --molecule Benzene
    python scripts/hf_orbitals.py --all --n-virtual 10 --cube
"""

import argparse
import textwrap
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pyscf.dft import numint
from pyscf.tools import cubegen, molden
from skimage.measure import marching_cubes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vqe_cudaq.hamiltonian import run_scf
from vqe_cudaq.molecules import molecules
from vqe_cudaq.orbital_selection import (BOND_SHARE_MIN, NON_VALENCE, bond_summary, classify_orbitals,
                                         localized_orbitals, select_active_orbitals)

HARTREE_TO_EV = 27.211386245988
BOHR = 0.52917721067
POSITIVE, NEGATIVE = "#d1495b", "#2e6fb7"
CLOSE_LEVELS_EV = 0.3
# bond types are named by their atoms ("C–H σ", "C–O π*"); each kind has a family of shades
SHADES = {
    "pi": ["#1d4ed8", "#4f46e5", "#0369a1", "#1e3a8a"],
    "xx": ["#15803d", "#0f766e", "#4d7c0f", "#166534"],
    "xh": ["#dc2626", "#ea580c", "#b91c1c"],
    "lone": ["#9333ea", "#c026d3", "#7e22ce", "#a855f7"],
    "pi*": ["#38bdf8", "#818cf8", "#7dd3fc", "#93c5fd"],
    "xx*": ["#4ade80", "#2dd4bf", "#a3e635", "#86efac"],
    "xh*": ["#f87171", "#fb923c", "#fca5a5"],
    "nb*": ["#d8b4fe", "#f0abfc"],
    "non-valence": ["#9ca3af"],
}
KIND_ORDER = list(SHADES)
OTHER = f"other (each < {BOND_SHARE_MIN})"
OTHER_COLOUR = "#e5e7eb"
DISPLAY_NAME = {"NH2-": "Amide anion"}
ATOM_COLOUR = {"H": "#94a3b8", "C": "#1e293b", "N": "#7c3aed", "O": "#ea580c"}


def nice_name(name):
    return name.replace("-", "−")


def kind(bond):
    if bond == NON_VALENCE:
        return "non-valence"
    if "lone pair" in bond:
        return "lone"
    if "non-bonding" in bond:
        return "nb*"
    star = "*" if bond.endswith("*") else ""
    if "π" in bond:
        return "pi" + star
    return ("xh" if "–H" in bond else "xx") + star


def palette(bonds):
    """A colour for every bond name of one molecule: the kind gives the family, the name the shade."""
    colours = {}
    for k in KIND_ORDER:
        for i, b in enumerate(sorted(b for b in set(bonds) if kind(b) == k)):
            colours[b] = SHADES[k][i % len(SHADES[k])]
    return colours


def legend_order(bonds):
    return sorted(set(bonds), key=lambda b: (KIND_ORDER.index(kind(b)), b))


def formula(mol):
    """Molecular formula as matplotlib math text: C and H first when there is carbon, charge as superscript."""
    counts = {}
    for a in range(mol.natm):
        counts[mol.atom_symbol(a)] = counts.get(mol.atom_symbol(a), 0) + 1
    heavy = sorted(s for s in counts if s not in ("C", "H"))
    order = (["C", "H"] + heavy) if "C" in counts else (heavy + ["H"])
    text = "".join(s + (f"_{{{counts[s]}}}" if counts[s] > 1 else "") for s in order if s in counts)
    if mol.charge:
        size = f"{abs(mol.charge)}" if abs(mol.charge) > 1 else ""
        text += f"^{{{size}{'+' if mol.charge > 0 else '-'}}}"
    return rf"$\mathrm{{{text}}}$"


def hf_summary(name, mf, table):
    """Heading of the level diagram: molecule, method, and the main numbers of the HF ground state."""
    mol, nocc = mf.mol, table.attrs["nocc"]
    homo, lumo = mf.mo_energy[nocc - 1], mf.mo_energy[nocc]
    basis = mol.basis if isinstance(mol.basis, str) else "mixed basis"
    line1 = f"{DISPLAY_NAME.get(name, name)} ({formula(mol)}), RHF/{basis}"
    charge = "neutral" if mol.charge == 0 else f"charge {mol.charge:+d}"
    line2 = (f"{mol.nelectron} electrons, {charge}, singlet ground state; "
             f"{mol.nao} basis functions = {mol.nao} MOs")
    line3 = (f"$E_{{\\rm HF}}$ = {mf.e_tot:.6f} Ha, $E_{{\\rm nuc}}$ = {mol.energy_nuc():.6f} Ha, "
             f"HOMO–LUMO gap = {(lumo - homo) * HARTREE_TO_EV:.2f} eV")
    if mol.charge == 0:
        line3 += f", dipole = {np.linalg.norm(mf.dip_moment(unit='Debye', verbose=0)):.2f} D"
    n_vir = int(min(VIRTUALS, mol.nao - nocc))
    ncore = table.attrs["ncore"]
    line4 = (f"Shown: {nocc - ncore} occupied valence and first {n_vir} virtual orbitals "
             f"({ncore} core orbital{'s' if ncore != 1 else ''} not shown)")
    return "\n".join([line1, line2, line3, line4])


DEGENERATE_DRAW_EV = 0.15   # a level closer than this to the previous one goes to the other column
LABEL_GAP_IN = 0.13         # vertical space one name label needs (inches)
LINE = {0: (1.30, 2.00), 1: (2.30, 3.00)}    # x range of the lines in the left and right column


def _spread(ys, sep, low, high):
    """Label heights near ``ys`` (sorted), at least ``sep`` apart, inside [low, high]."""
    out = list(ys)
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1] + sep)
    if out and out[-1] > high:
        out[-1] = high
        for i in range(len(out) - 2, -1, -1):
            out[i] = min(out[i], out[i + 1] - sep)
    return out


def _columns(energies):
    """Left column, except a level close to the previous one, which goes to the other column."""
    columns = []
    for i, e in enumerate(energies):
        columns.append(1 - columns[-1] if i and e - energies[i - 1] < DEGENERATE_DRAW_EV else 0)
    return columns


def _panel_height(energies, default_in):
    """Inches a panel needs so that the labels of each column fit near their lines."""
    columns = _columns(energies)
    densest = 1
    for c in (0, 1):
        e = np.sort(energies[np.array(columns) == c])
        for i in range(len(e)):
            densest = max(densest, int(((e >= e[i]) & (e < e[i] + 1.0)).sum()))
    span = energies.max() - energies.min() + 1.2
    return max(default_in, 1.6, span * densest * LABEL_GAP_IN * 1.15)


def _levels(ax, rows, colours, marked, height_in, used, used_other):
    """One panel of levels: each line split by bond share, its name outside the line."""
    energies = rows["energy"].to_numpy() * HARTREE_TO_EV
    low, high = energies.min() - 0.6, energies.max() + 0.6
    ax.set_ylim(low, high)
    columns = _columns(energies)
    sep = LABEL_GAP_IN * (high - low) / height_in
    for c in (0, 1):
        members = [i for i, col in enumerate(columns) if col == c]
        labels_y = _spread([energies[i] for i in members], sep, low + sep / 2, high - sep / 2)
        x0, x1 = LINE[c]
        for i, y in zip(members, labels_y):
            r, e = rows.iloc[i], energies[i]
            start = x0
            pieces = bond_summary(r["bond_shares"])
            for bond, w in pieces:                  # one piece per bond type, length = its share
                ax.hlines(e, start, start + (x1 - x0) * w, color=colours[bond], lw=2.2)
                start += (x1 - x0) * w
            if x1 - start > 0.005:
                ax.hlines(e, start, x1, color=OTHER_COLOUR, lw=2.2)
                used_other.append(True)
            used.extend(b for b, _ in pieces)
            if r["index"] in marked:
                ax.plot(x1 + 0.06, e, "<", ms=4.5, color="black")
            # labels outside: left of the left column, right of the right column
            edge, tx, ha = (x0 - 0.04, x0 - 0.12, "right") if c == 0 else (x1 + 0.12, x1 + 0.20, "left")
            if abs(y - e) > 1e-9:
                ax.plot([edge, tx + (0.02 if c == 0 else -0.02)], [e, y], color="#9ca3af", lw=0.5)
            ax.text(tx, y, f"{nice_name(r['name'])}  {e:.4f}", va="center", ha=ha, fontsize=6.5, color="#1f2937")


def level_diagram(name, table, marked, path, mf, colours, pdf=None):
    shown = table[~table["core"]]
    nocc = table.attrs["nocc"]
    shown = shown[shown["index"] < nocc + VIRTUALS]
    occ, vir = shown[shown["occupied"]], shown[~shown["occupied"]]
    e_occ, e_vir = (occ["energy"].to_numpy() * HARTREE_TO_EV, vir["energy"].to_numpy() * HARTREE_TO_EV)
    span = lambda e: e.max() - e.min() + 1.2
    base = 7.2 / (span(e_occ) + span(e_vir))        # inches per eV when nothing is crowded
    h_vir = _panel_height(e_vir, span(e_vir) * base)
    h_occ = _panel_height(e_occ, span(e_occ) * base)
    head, foot, gap = 1.9, 0.5, 0.65
    fig_h = head + h_vir + gap + h_occ + foot
    fig = plt.figure(figsize=(11.5, fig_h))
    ax_v = fig.add_axes((0.08, (foot + h_occ + gap) / fig_h, 0.58, h_vir / fig_h))
    ax_o = fig.add_axes((0.08, foot / fig_h, 0.58, h_occ / fig_h))
    used, used_other = [], []
    _levels(ax_v, vir, colours, marked, h_vir, used, used_other)
    _levels(ax_o, occ, colours, marked, h_occ, used, used_other)
    for ax, title in ((ax_v, "virtual orbitals"), (ax_o, "occupied valence orbitals")):
        ax.set_xlim(0, 4.2)
        ax.set_xticks([])
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_title(title, loc="left", fontsize=9, color="#6b7280")
    for ax, y in ((ax_v, 0), (ax_o, 1)):            # the HOMO-LUMO gap is cut out of the axis
        ax.plot([-0.012, 0.012], [y - 0.012, y + 0.012], transform=ax.transAxes, color="black", lw=1, clip_on=False)
    homo, lumo = e_occ.max(), e_vir.min()
    fig.text(0.10, (foot + h_occ + 0.45) / fig_h, f"axis broken: HOMO–LUMO gap {lumo - homo:.2f} eV",
             fontsize=8.5, color="#6b7280", va="center")
    fig.text(0.02, (foot + h_occ + gap / 2) / fig_h, "Hartree–Fock orbital energy (eV)", rotation=90, va="center")
    handles = [plt.Line2D([], [], color=colours[b], lw=3, label=b) for b in legend_order(used)]
    if used_other:
        handles.append(plt.Line2D([], [], color=OTHER_COLOUR, lw=3, label=OTHER))
    handles.append(plt.Line2D([], [], marker="<", ms=5, color="black", ls="",
                              label="in the current (6e,7o) space"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.70, (foot + h_occ + gap + h_vir) / fig_h),
               frameon=False, fontsize=10, title=f"bond shares above {BOND_SHARE_MIN}\n(line length = share)",
               title_fontsize=10)
    fig.text(0.08, 1 - 0.25 / fig_h, hf_summary(name, mf, table), fontsize=11, va="top", linespacing=1.4)
    fig.savefig(path, dpi=DPI)
    if pdf is not None:
        pdf.savefig(fig)                            # vector page, as tall as the diagram needs
    plt.close(fig)


# ── "clear" level diagram: one hue per bond, its antibond in a light shade of the same hue ──
FAMILY_HUES = {
    "pi": ["#1d4ed8", "#0e7490", "#4338ca", "#1e40af"],
    "xx": ["#15803d", "#0f766e", "#65a30d", "#166534"],
    "xh": ["#dc2626", "#ea580c", "#be123c"],
    "lone": ["#7e22ce", "#c026d3", "#9333ea", "#a21caf"],
}
DEGENERATE_EV = 0.005       # levels this close are drawn as one line with both names


def _pair(bond):
    """Bond and antibond share a key: "C–O π" and "C–O π*"; "O lone pair" and "O non-bonding"."""
    key = bond.rstrip("*")
    if "lone pair" in key or "non-bonding" in key:
        key = key.split()[0] + " lone pair/non-bonding"
    return key


def _lighter(colour, amount=0.55):
    r, g, b = to_rgb(colour)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def clear_palette(bonds):
    keys = {}
    for b in bonds:
        if b == NON_VALENCE:
            continue
        family = kind(b).rstrip("*")
        family = "lone" if family == "nb" else family
        keys.setdefault(family, set()).add(_pair(b))
    hue = {}
    for family, ks in keys.items():
        for i, k in enumerate(sorted(ks)):
            hue[k] = FAMILY_HUES[family][i % len(FAMILY_HUES[family])]
    colours = {NON_VALENCE: "#9ca3af"}
    for b in bonds:
        if b != NON_VALENCE:
            colours[b] = _lighter(hue[_pair(b)]) if b.endswith("*") or "non-bonding" in b else hue[_pair(b)]
    return colours


def _groups(rows):
    """Consecutive levels within DEGENERATE_EV become one line: (energy, names, row, indices)."""
    out = []
    for _, r in rows.iterrows():
        e = r["energy"] * HARTREE_TO_EV
        if out and e - out[-1][0] < DEGENERATE_EV:
            out[-1][1].append(nice_name(r["name"]))
            out[-1][3].append(int(r["index"]))
        else:
            out.append([e, [nice_name(r["name"])], r, [int(r["index"])]])
    return out


def _clear_panel(ax, rows, colours, marked, height_in, used, used_other):
    """Names on the left, the line split by share, the shares written on the right.

    Levels closer than DEGENERATE_DRAW_EV (not degenerate) would touch, so each gets half the
    line length, side by side.
    """
    groups = _groups(rows)
    energies = np.array([g[0] for g in groups])
    low, high = energies.min() - 0.6, energies.max() + 0.6
    ax.set_ylim(low, high)
    sep = LABEL_GAP_IN * (high - low) / height_in
    ys = _spread(list(energies), sep, low + sep / 2, high - sep / 2)
    close = [i > 0 and energies[i] - energies[i - 1] < DEGENERATE_DRAW_EV for i in range(len(energies))]
    slots, slot = [], None                       # None = full line, 0 = left half, 1 = right half
    for i in range(len(energies)):
        if close[i]:
            slot = 1 - slot if slot is not None else 1
            if slots[-1] is None:
                slots[-1] = 0
        else:
            slot = None
        slots.append(slot)
    X0, X1 = 3.7, 7.0                           # the full line
    mid = (X0 + X1) / 2
    for (e, names, r, idx), y, s in zip(groups, ys, slots):
        x0, x1 = {None: (X0, X1), 0: (X0, mid - 0.06), 1: (mid + 0.06, X1)}[s]
        start = x0
        pieces = bond_summary(r["bond_shares"])
        for bond, w in pieces:
            ax.hlines(e, start, start + (x1 - x0) * w, color=colours[bond], lw=4.5)
            start += (x1 - x0) * w
        if x1 - start > 0.005:
            ax.hlines(e, start, x1, color=OTHER_COLOUR, lw=4.5)
            used_other.append(True)
        used.extend(b for b, _ in pieces)
        if any(k in marked for k in idx):
            ax.plot(X1 + 0.12, e, "<", ms=6, color="black")
        label = " / ".join(names) + (" (deg.)" if len(names) > 1 else "")
        shares = "   ".join(f"{b} {w:.2f}" for b, w in pieces)
        ax.plot([X0 - 0.22, x0 - 0.03], [y, e], color="#9ca3af", lw=0.6)
        ax.plot([X1 + 0.3, X1 + 0.45], [e, y], color="#9ca3af", lw=0.6)
        if s == 1:                               # the right half: also join its left end to the label line
            ax.plot([X0 - 0.03, x0 - 0.03], [e, e], color="#e5e7eb", lw=0.6)
        ax.text(X0 - 0.26, y, f"{label}   {e:.4f} eV", ha="right", va="center", fontsize=8.5, color="#111827")
        ax.text(X1 + 0.5, y, shares, ha="left", va="center", fontsize=8.5, color="#374151")


def level_diagram_clear(name, table, marked, path, mf, colours, pdf=None):
    shown = table[~table["core"]]
    nocc = table.attrs["nocc"]
    shown = shown[shown["index"] < nocc + VIRTUALS]
    occ, vir = shown[shown["occupied"]], shown[~shown["occupied"]]
    e_occ, e_vir = occ["energy"].to_numpy() * HARTREE_TO_EV, vir["energy"].to_numpy() * HARTREE_TO_EV
    span = lambda e: e.max() - e.min() + 1.2
    base = 7.0 / (span(e_occ) + span(e_vir))
    def need(e, default):                            # labels sit in one column, so count all levels
        dens = max(int(((e >= x) & (e < x + 1.0)).sum()) for x in e)
        return max(default, 1.8, span(e) * dens * LABEL_GAP_IN * 1.25)
    h_vir, h_occ = need(e_vir, span(e_vir) * base), need(e_occ, span(e_occ) * base)
    used = []
    for _, r in shown.iterrows():
        used.extend(b for b, _ in bond_summary(r["bond_shares"]))
    bonds = [b for b in legend_order(used) if not b.endswith("*") and b != NON_VALENCE]
    antis = [b for b in legend_order(used) if b.endswith("*")]
    legend_rows = max(len(bonds), len(antis), 3)
    head, legend_h, foot, gap = 1.6, 0.55 + 0.27 * legend_rows, 0.45, 0.7
    fig_h = head + legend_h + h_vir + gap + h_occ + foot
    fig = plt.figure(figsize=(14.0, fig_h))
    ax_v = fig.add_axes((0.07, (foot + h_occ + gap) / fig_h, 0.90, h_vir / fig_h))
    ax_o = fig.add_axes((0.07, foot / fig_h, 0.90, h_occ / fig_h))
    used_other = []
    _clear_panel(ax_v, vir, colours, marked, h_vir, [], used_other)
    _clear_panel(ax_o, occ, colours, marked, h_occ, [], used_other)
    for ax, title in ((ax_v, "virtual orbitals"), (ax_o, "occupied valence orbitals")):
        ax.set_xlim(0, 13.5)
        ax.set_xticks([])
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_title(title, loc="left", fontsize=10, color="#6b7280")
    for ax, y in ((ax_v, 0), (ax_o, 1)):
        ax.plot([-0.01, 0.01], [y - 0.012, y + 0.012], transform=ax.transAxes, color="black", lw=1, clip_on=False)
    fig.text(0.09, (foot + h_occ + 0.48) / fig_h,
             f"axis broken: HOMO–LUMO gap {e_vir.min() - e_occ.max():.4f} eV", fontsize=9, color="#6b7280", va="center")
    fig.text(0.015, (foot + h_occ + gap / 2) / fig_h, "Hartree–Fock orbital energy (eV)", rotation=90, va="center",
             fontsize=11)
    top = 1 - head / fig_h
    swatch = lambda b: plt.Line2D([], [], color=colours[b], lw=5, label=b)
    extra = [plt.Line2D([], [], color=colours[NON_VALENCE], lw=5, label=NON_VALENCE + " (virtual only)")]
    if used_other:
        extra.append(plt.Line2D([], [], color=OTHER_COLOUR, lw=5, label=OTHER))
    extra.append(plt.Line2D([], [], marker="<", ms=6, color="black", ls="", label="in the current (6e,7o) space"))
    for x, title, handles in ((0.07, "bonds (occupied)", [swatch(b) for b in bonds]),
                              (0.33, "antibonds (virtual)", [swatch(b) for b in antis]),
                              (0.59, "also shown", extra)):
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x, top), frameon=False, fontsize=10,
                   title=title, title_fontsize=10, alignment="left")
    fig.text(0.07, 1 - 0.2 / fig_h, hf_summary(name, mf, table) + f"\nLine length = bond share; shares above "
             f"{BOND_SHARE_MIN} are written on the right; deg. = degenerate.", fontsize=11, va="top", linespacing=1.4)
    fig.savefig(path, dpi=DPI)
    if pdf is not None:
        pdf.savefig(fig)
    plt.close(fig)


def molecule_frame(mol):
    """Rows are the molecule's principal axes: long axis, short axis, plane normal."""
    xyz = mol.atom_coords() * BOHR
    centre = xyz.mean(0)
    axes = np.linalg.svd(xyz - centre)[2]
    return centre, axes, (xyz - centre) @ axes.T


class Grid:
    def __init__(self, mol, spacing, padding):
        self.centre, self.axes, self.atoms = molecule_frame(mol)
        low, high = self.atoms.min(0) - padding, self.atoms.max(0) + padding
        self.ticks = [np.arange(a, b + spacing, spacing) for a, b in zip(low, high)]
        self.spacing, self.shape, self.low = spacing, tuple(len(t) for t in self.ticks), low
        mesh = np.stack(np.meshgrid(*self.ticks, indexing="ij"), -1).reshape(-1, 3)
        points = (self.centre + mesh @ self.axes) / BOHR       # back to the molecule's own frame
        self.ao = np.vstack([numint.eval_ao(mol, points[i:i + 50000]) for i in range(0, len(points), 50000)])

    def orbital(self, coefficients):
        return (self.ao @ coefficients).reshape(self.shape)

    def touches_box(self, psi, level):
        """True if the isosurface would be cut off by the edge of the box."""
        faces = [psi[0], psi[-1], psi[:, 0], psi[:, -1], psi[:, :, 0], psi[:, :, -1]]
        return max(np.abs(face).max() for face in faces) >= level


def isosurface(ax, psi, level, grid, light):
    """Both signs in one collection, so that their faces are depth-sorted together."""
    triangles, colours = [], []
    for sign, colour in ((+1, POSITIVE), (-1, NEGATIVE)):
        field = sign * psi
        if field.max() <= level:
            continue
        verts, faces, _, _ = marching_cubes(field, level, spacing=(grid.spacing,) * 3)
        tri = (verts + grid.low)[faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        brightness = 0.5 + 0.5 * np.abs(normals @ light)
        triangles.append(tri)
        colours.append(np.c_[np.outer(brightness, to_rgb(colour)), np.full(len(tri), 0.95)])
    ax.add_collection3d(Poly3DCollection(np.vstack(triangles), facecolors=np.vstack(colours),
                                         edgecolors="none", zorder=1))
    return np.vstack(triangles).reshape(-1, 3)


def panel(mol, grid, psi, title, colour, level, path, bonds, subtitle=""):
    extent = grid.atoms.max(0) - grid.atoms.min(0)
    long_molecule = extent[0] > 2.2 * max(extent[1], 1e-6)
    fig = plt.figure(figsize=(4.4, 2.75) if long_molecule else (3.2, 2.9))
    ax = fig.add_axes((-0.1, -0.12, 1.2, 1.12), projection="3d", computed_zorder=False)
    light = np.array([-0.4, -0.5, 0.77])
    used = min(level, 0.6 * np.abs(psi).max())
    surface = isosurface(ax, psi, used, grid, light)
    for i, j in bonds:
        ax.plot(*np.c_[grid.atoms[i], grid.atoms[j]], color="#334155", lw=1.2, zorder=2)
    for a in range(mol.natm):
        symbol = mol.atom_symbol(a)
        ax.scatter(*grid.atoms[a], s=14 if symbol == "H" else 26, color=ATOM_COLOUR[symbol], depthshade=False, zorder=3)
    middle = (grid.atoms.max(0) + grid.atoms.min(0)) / 2
    # diffuse orbitals reach far from the atoms: widen the view so that they fit
    half = max(max(extent) / 2 + 1.5, np.abs(surface - middle).max() + 0.2)
    ax.set(xlim=(middle[0] - half, middle[0] + half), ylim=(middle[1] - half, middle[1] + half),
           zlim=(middle[2] - half, middle[2] + half))
    ax.set_box_aspect((1, 1, 1), zoom=1.75 if long_molecule else 1.5)
    ax.view_init(elev=48, azim=-68)
    ax.set_axis_off()
    fig.text(0.5, 0.975, title, ha="center", va="top", fontsize=9.5, color=colour, fontweight="bold")
    fig.text(0.5, 0.905, subtitle, ha="center", va="top", fontsize=7, color="#1f2937", linespacing=1.15)
    if used < level:
        fig.text(0.5, 0.02, f"iso ±{used:.3f}", ha="center", fontsize=7.5, color="#475569")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def find_bonds(mol):
    xyz = mol.atom_coords() * BOHR
    limit = {"H": 0.45, "C": 0.85, "N": 0.8, "O": 0.8}
    return [(i, j) for i in range(mol.natm) for j in range(i)
            if np.linalg.norm(xyz[i] - xyz[j]) < limit[mol.atom_symbol(i)] + limit[mol.atom_symbol(j)]
            and not (mol.atom_symbol(i) == "H" and mol.atom_symbol(j) == "H")]


def page_of_panels(pdf, name, mol, images, columns=5, rows=4):
    for start in range(0, len(images), columns * rows):
        fig, axes = plt.subplots(rows, columns, figsize=(11.69, 8.27))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, image in zip(axes.ravel(), images[start:start + columns * rows]):
            ax.imshow(plt.imread(image), interpolation="none")     # embedded at full resolution
        fig.suptitle(f"{DISPLAY_NAME.get(name, name)} ({formula(mol)}): orbitals, isosurface ±{ISO} a.u.; "
                     "red and blue are the two signs",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97), pad=0.3)
        pdf.savefig(fig, dpi=170)
        plt.close(fig)


MATRIX_COLUMNS = 14        # number columns per page of a matrix table
MATRIX_ROWS = 28           # rows per page
IAO_MATRIX_LIMIT = 16      # Table 2 as a matrix up to this many IAO columns, else as a list


def _style(tab, widths, header_rows=1):
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    for (r, c), cell in tab.get_celld().items():
        cell.PAD = 0.006 / widths[c]                # same small margin in every column
        cell.set_edgecolor("#d1d5db")
        cell.set_linewidth(0.4)
        if r < header_rows:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#f3f4f6")


PENDING = []               # tables waiting to be placed on pages
PAGE_W, PAGE_H = 11.69, 8.27
ROW_IN = 0.205             # height of one table row (inches)


CHAR_W = 0.0064            # width of one character at font size 8, as a fraction of the table width
CAPTION_CHARS = 140        # caption line length that fits the page at font size 8.5


def _page(pdf, heading, caption, columns, rows, widths, align="left"):
    """Queue one table; flush_tables puts as many as fit on each page.

    A long heading is wrapped to the width of its column's values, and every column is made wide
    enough for its longest line, so headings and values never run into the next column.
    """
    rows = rows or [[""] * len(columns)]
    longest_cell = [max([len(str(r[c])) for r in rows] + [1]) for c in range(len(columns))]
    columns = [textwrap.fill(str(h), max(12, longest_cell[c])) for c, h in enumerate(columns)]
    needed = [CHAR_W * max(longest_cell[c], max(len(line) for line in columns[c].split("\n"))) + 0.014
              for c in range(len(columns))]
    widths = [max(w, n) for w, n in zip(widths, needed)]
    if sum(widths) > 1.0:
        widths = [w / sum(widths) for w in widths]
    PENDING.append(dict(heading=heading, caption=textwrap.fill(caption, CAPTION_CHARS), columns=columns,
                        rows=rows, widths=widths, align=align))


def _header_lines(block):
    return max(h.count("\n") + 1 for h in block["columns"])


def _height(block):
    return (0.30 + 0.16 * (block["caption"].count("\n") + 1) + 0.12
            + ROW_IN * (len(block["rows"]) + _header_lines(block)) + 0.35)


def flush_tables(pdf):
    """Place the queued tables top to bottom; a new page only when the next one does not fit."""
    top, bottom = 0.35, 0.30
    fig, y = None, None
    for block in PENDING:
        h = _height(block)
        if fig is None or y - h < bottom:
            if fig is not None:
                pdf.savefig(fig)
                plt.close(fig)
            fig, y = plt.figure(figsize=(PAGE_W, PAGE_H)), PAGE_H - top
        fig.text(0.04, (y - 0.20) / PAGE_H, block["heading"], fontsize=12, fontweight="bold")
        n_lines = block["caption"].count("\n") + 1
        fig.text(0.04, (y - 0.32) / PAGE_H, block["caption"], fontsize=8.5, color="#374151", va="top")
        table_top = y - 0.30 - 0.16 * n_lines - 0.12
        height = ROW_IN * (len(block["rows"]) + _header_lines(block))
        width = 0.92 * min(1.0, sum(block["widths"]))
        ax = fig.add_axes((0.04, (table_top - height) / PAGE_H, width, height / PAGE_H))
        ax.axis("off")
        tab = ax.table(cellText=block["rows"], colLabels=block["columns"], colWidths=block["widths"],
                       cellLoc=block["align"], colLoc=block["align"], bbox=[0, 0, 1, 1])
        _style(tab, block["widths"])
        for (r, _), cell in tab.get_celld().items():     # a wrapped heading gets a taller row
            cell.set_height(_header_lines(block) if r == 0 else 1.0)
        y -= h
    if fig is not None:
        pdf.savefig(fig)
        plt.close(fig)
    PENDING.clear()


def table_pages(pdf, heading, caption, columns, rows, widths, per_page=30):
    """Rows of strings as one or more A4 pages; long tables continue on the next page."""
    for start in range(0, max(len(rows), 1), per_page):
        _page(pdf, heading + ("  (continued)" if start else ""), caption, columns,
              rows[start:start + per_page], widths)


def _number(v, digits=2):
    return "0" if abs(v) < 0.5 * 10 ** -digits else f"{v:.{digits}f}"


def matrix_pages(pdf, heading, caption, lead_headers, lead_rows, col_headers, values, digits=2,
                 lead_widths=None, total=False):
    """A matrix as in the notes: rows x columns of numbers, split into blocks of pages when too big."""
    values = np.asarray(values)
    lead_widths = lead_widths or [0.08] * len(lead_headers)
    pages = [(r, c) for r in range(0, max(len(lead_rows), 1), MATRIX_ROWS)
             for c in range(0, max(len(col_headers), 1), MATRIX_COLUMNS)]
    for n, (r, c) in enumerate(pages):
        cols = col_headers[c:c + MATRIX_COLUMNS]
        width = min(0.065, (0.98 - sum(lead_widths)) / max(len(cols), 1))
        widths = list(lead_widths) + [width] * len(cols)
        rows = [list(lead) + [_number(v, digits) for v in values[i, c:c + MATRIX_COLUMNS]]
                for i, lead in enumerate(lead_rows[r:r + MATRIX_ROWS], start=r)]
        if total and r + MATRIX_ROWS >= len(lead_rows):
            rows.append(["total"] + [""] * (len(lead_headers) - 1)
                        + [_number(v, digits) for v in values[:, c:c + MATRIX_COLUMNS].sum(0)])
        part = f"  (part {n + 1} of {len(pages)})" if len(pages) > 1 else ""
        _page(pdf, heading + part, caption, list(lead_headers) + list(cols), rows, widths, align="center")


def listed(values, labels, minimum=BOND_SHARE_MIN, at_least=2, most=6):
    """ "label value, ..." for the values above ``minimum`` (at least the ``at_least`` largest)."""
    order = np.argsort(-values)
    keep = [i for i in order if values[i] > minimum][:most] or list(order[:at_least])
    text = ", ".join(f"{labels[i]} {values[i]:.2f}" for i in keep)
    rest = int((values[order[len(keep):]] > 0.02).sum())
    return text + (f"  (+{rest} more above 0.02)" if rest else "")


def tables(pdf, name, mol, table, loc, chosen):
    """The five tables of the notes, from the canonical orbitals to their bond shares."""
    who = f"{DISPLAY_NAME.get(name, name)} ({formula(mol)})"
    atoms = [f"{mol.atom_symbol(a)}{a}" for a in range(mol.natm)]
    nocc, ncore = loc["nocc"], loc["ncore"]
    bond, anti = loc["bond"], loc["antibond"]
    bid = [f"B{i + 1}" for i in range(len(bond["names"]))]
    aid = [f"A{j + 1}" for j in range(len(anti["names"]))]
    occ = chosen[chosen["occupied"]]
    vir = chosen[~chosen["occupied"]]
    occ_names = [nice_name(n) for n in occ["name"]]
    vir_names = [nice_name(n) for n in vir["name"]]
    occ_k = occ["index"].to_numpy() - ncore
    vir_k = vir["index"].to_numpy() - nocc

    matrix_pages(pdf, f"{who}: Table 1. Occupied valence orbitals on the atoms",
                 "How much of each canonical orbital is on each atom, measured with the intrinsic atomic orbitals "
                 "(IAOs); each row adds up to 1. A canonical orbital is spread over many atoms, so it cannot be "
                 "named as one bond.",
                 ["MO"], [[n] for n in occ_names], atoms, loc["mo_charges"][occ_k], lead_widths=[0.09])

    names = list(bond["names"]) + list(anti["names"])
    lead = [[i, n] for i, n in zip(bid + aid, names)]
    weights = np.vstack([bond["iao_weights"], anti["iao_weights"]])
    caption2 = ("B = intrinsic bond orbitals (IBOs), made by mixing the occupied valence orbitals. A = localized "
                "intrinsic valence virtual orbitals (LIVVOs), the same for the virtual orbitals: the antibonding "
                "partners. Each entry is the square of the overlap with one IAO; each row adds up to 1.")
    heavy_1s = [k for k, l in enumerate(loc["iao_labels"]) if l.endswith(" 1s") and not l.startswith("H")]
    cols = [k for k in range(len(loc["iao_labels"])) if k not in heavy_1s]
    if len(cols) <= IAO_MATRIX_LIMIT:
        matrix_pages(pdf, f"{who}: Table 2. Localized orbitals on the IAOs", caption2 + " Core 1s IAOs are left out.",
                     ["orbital", "name"], lead, [loc["iao_labels"][k] for k in cols], weights[:, cols],
                     digits=3, lead_widths=[0.06, 0.10])
    else:
        rows = [[i, n, listed(w, loc["iao_labels"])] for (i, n), w in zip(lead, weights)]
        table_pages(pdf, f"{who}: Table 2. Localized orbitals on the IAOs",
                    caption2 + f" With {len(cols)} valence IAOs, only the IAOs above {BOND_SHARE_MIN} are listed.",
                    ["orbital", "name", f"IAOs (share > {BOND_SHARE_MIN})"], rows, [0.07, 0.13, 0.80])

    matrix_pages(pdf, f"{who}: Table 3. Localized orbitals on the atoms, and their names",
                 "Table 2 added up by atom. The name comes from the atoms above 0.15: with H it is X–H, one heavy "
                 "atom is a lone pair, two heavy atoms are X–X; σ or π from the share out of the plane; * marks the "
                 "antibonding (virtual) orbitals.",
                 ["orbital", "name"], lead, atoms, np.vstack([bond["charges"], anti["charges"]]),
                 lead_widths=[0.06, 0.10])

    matrix_pages(pdf, f"{who}: Table 4a. Bond orbitals inside each occupied orbital",
                 "The square of the overlap of each IBO with each occupied valence orbital. Every column adds up to 1.",
                 ["IBO", "name"], [[i, n] for i, n in zip(bid, bond["names"])], occ_names,
                 bond["overlap"][:, occ_k] ** 2, lead_widths=[0.05, 0.10], total=True)
    w = anti["overlap"][:, vir_k] ** 2
    matrix_pages(pdf, f"{who}: Table 4b. Antibond orbitals inside each virtual orbital",
                 "The square of the overlap of each LIVVO with each virtual orbital. The LIVVOs add up to the valence "
                 "part of the orbital; the rest is no bond (outside valence): it comes from the extra functions of the basis set (for C: 3s, 3p, 3d; for H: 2s, 2p).",
                 ["LIVVO", "name"], [[j, n] for j, n in zip(aid, anti["names"])] + [["", NON_VALENCE]],
                 vir_names, np.vstack([w, 1 - w.sum(0)]), lead_widths=[0.05, 0.10], total=True)

    for part, rows_, label in (("a", occ, "occupied valence"), ("b", vir, "virtual")):
        kinds = legend_order(n for d in rows_["bond_shares"] for n, v in d.items() if v > 0.005)
        values = np.array([[d.get(n, 0.0) for n in kinds] for d in rows_["bond_shares"]])
        lead = [[nice_name(r["name"]), f"{r['energy'] * HARTREE_TO_EV:.2f}", f"{r['pi_score']:.2f}", r["main_bond"]]
                for _, r in rows_.iterrows()]
        matrix_pages(pdf, f"{who}: Table 5{part}. Bond shares of each {label} orbital",
                     "Table 4 added up over the orbitals with the same name; each row adds up to 1. P = share on the "
                     "p and d functions pointing out of the plane (above 0.5: π). Degenerate orbitals share one "
                     f"average. The isosurface titles list the shares above {BOND_SHARE_MIN}.",
                     ["MO", "energy (eV)", "P", "largest"], lead, kinds, values,
                     lead_widths=[0.08, 0.08, 0.05, 0.11])
    flush_tables(pdf)


def run_molecule(name, args):
    t0 = time.time()
    mf = run_scf(molecules[name], args.basis, max_memory=args.max_memory)
    table = classify_orbitals(mf)
    mol = mf.mol
    nocc = table.attrs["nocc"]
    folder = os.path.join(args.outdir, name)
    os.makedirs(os.path.join(folder, "orbitals"), exist_ok=True)
    keep = ["index", "name", "occupied", "core", "energy", "label", "degenerate_group", "pi_score",
            "share_pi", "share_xh", "share_xx", "share_lone", "h_pop"]
    out = table[keep + ["bonds", "main_bond", "no_bond_outside_valence"]].assign(energy_eV=table["energy"] * HARTREE_TO_EV)
    names = legend_order(n for d in table["bond_shares"] for n in d)
    for n in names:
        out[f"share: {n}"] = [d.get(n, 0.0) for d in table["bond_shares"]]
    out.round(5).to_csv(os.path.join(folder, f"{name}_orbitals.csv"), index=False, encoding="utf-8-sig")
    loc = localized_orbitals(mf)
    atoms = [f"{mol.atom_symbol(a)}{a}" for a in range(mol.natm)]
    with open(os.path.join(folder, f"{name}_localized.csv"), "w", encoding="utf-8-sig") as f:
        f.write("orbital,kind,name,atoms,iaos\n")
        for tag, part, what in (("B", loc["bond"], "IBO"), ("A", loc["antibond"], "LIVVO")):
            for i, n in enumerate(part["names"]):
                f.write(f"{tag}{i + 1},{what},{n},\"{listed(part['charges'][i], atoms, 0.15)}\","
                        f"\"{listed(part['iao_weights'][i], loc['iao_labels'])}\"\n")
    # every orbital, for Jmol, Avogadro, IboView or Multiwfn
    molden.from_mo(mol, os.path.join(folder, f"{name}.molden"), mf.mo_coeff, ene=mf.mo_energy, occ=mf.mo_occ)

    diagram = os.path.join(folder, f"{name}_levels{args.suffix}.png")
    draw = level_diagram_clear if args.level_style == "clear" else level_diagram
    marked = select_active_orbitals(table, 6, 7).active
    chosen = table[(~table["core"]) & (table["index"] < nocc + args.n_virtual)]
    shown_bonds = [b for d in chosen["bond_shares"] for b, _ in bond_summary(d)]
    colours = clear_palette(shown_bonds) if args.level_style == "clear" else palette(shown_bonds)
    draw(name, table, marked, diagram, mf, colours)

    grid = Grid(mol, args.spacing, args.padding)
    bonds = find_bonds(mol)
    images = []
    clipped = []
    for _, row in chosen.iterrows():
        k = int(row["index"])
        path = os.path.join(folder, "orbitals", f"{k:03d}_{row['name']}.png")
        title = f"{nice_name(row['name'])}   {row['energy'] * HARTREE_TO_EV:.4f} eV"
        parts = [f"{b} {w:.2f}" for b, w in bond_summary(row["bond_shares"])]
        subtitle = "\n".join(", ".join(parts[i:i + 3]) for i in range(0, len(parts), 3))
        main = bond_summary(row["bond_shares"])
        colour = colours[main[0][0]] if main else "#374151"
        psi = grid.orbital(mf.mo_coeff[:, k])
        if grid.touches_box(psi, args.iso):
            clipped.append(row["name"])
        if not (args.reuse and os.path.exists(path)):
            panel(mol, grid, psi, title, colour, args.iso, path, bonds, subtitle)
        images.append(path)
        if args.cube:
            os.makedirs(os.path.join(folder, "cube"), exist_ok=True)
            cubegen.orbital(mol, os.path.join(folder, "cube", f"{k:03d}_{row['name']}.cube"), mf.mo_coeff[:, k],
                            resolution=args.cube_resolution, margin=args.cube_margin)

    with PdfPages(os.path.join(folder, f"{name}_hf_orbitals{args.suffix}.pdf")) as pdf:
        draw(name, table, marked, diagram, mf, colours, pdf=pdf)
        tables(pdf, name, mol, table, loc, chosen)
        page_of_panels(pdf, name, mol, images)
    print(f"[DONE] {name}: E_HF {mf.e_tot:.8f}, {len(chosen)} orbitals, grid {grid.shape}, "
          f"{time.time() - t0:.0f} s, cut off by the box: {clipped or 'none'}", flush=True)


def main():
    global VIRTUALS, ISO, DPI
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--molecule", nargs="*", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--basis", default="cc-pVDZ")
    p.add_argument("--n-virtual", type=int, default=10, help="virtual orbitals per molecule")
    p.add_argument("--iso", type=float, default=0.04, help="isosurface value (a.u.)")
    p.add_argument("--spacing", type=float, default=0.2, help="grid spacing (Å)")
    p.add_argument("--padding", type=float, default=6.0, help="box beyond the atoms (Å)")
    p.add_argument("--max-memory", type=int, default=12000)
    p.add_argument("--reuse", action="store_true", help="keep orbital images that already exist")
    p.add_argument("--dpi", type=int, default=300, help="resolution of the images, kept as is in the PDF")
    p.add_argument("--cube", action="store_true", help="also write a cube file per plotted orbital")
    p.add_argument("--cube-resolution", type=float, default=0.3, help="cube grid spacing (Bohr)")
    p.add_argument("--cube-margin", type=float, default=5.0, help="cube box beyond the atoms (Bohr)")
    p.add_argument("--outdir", default=os.path.join("analysis", "figures", "hf_orbitals"))
    p.add_argument("--level-style", choices=["compact", "clear"], default="compact",
                   help="clear: long lines, shares written next to them; compact: the earlier two-column diagram")
    p.add_argument("--suffix", default="", help="added to the PDF and diagram file names, e.g. _v2")
    args = p.parse_args()
    VIRTUALS, ISO, DPI = args.n_virtual, args.iso, args.dpi
    names = list(molecules) if args.all else (args.molecule or [])
    if not names:
        raise SystemExit("Provide --molecule NAME ... or --all.")
    for name in names:
        run_molecule(name, args)
    if len(names) > 1:
        from pypdf import PdfWriter
        merged = PdfWriter()
        for name in names:
            merged.append(os.path.join(args.outdir, name, f"{name}_hf_orbitals{args.suffix}.pdf"), outline_item=name)
        merged.write(os.path.join(args.outdir, f"hf_orbitals_all{args.suffix}.pdf"))


if __name__ == "__main__":
    main()
