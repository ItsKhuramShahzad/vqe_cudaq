"""The analysis scripts read results/ and keep every molecule's marker and colour."""

import os

from conftest import RESULTS

# Position = marker/colour index in every figure. Methylene keeps slot 0 so that no
# other molecule moves; Tetracene sits in the old Benzaanthracene slot.
MOL_ORDER = ["Methylene", "Ethylene", "Benzene", "Naphthalene", "Tetracene", "Pentacene",
             "NH2-", "Methanamide", "Adenine", "Thymine", "Uracil", "Cytosine", "Guanine"]


def test_every_script_uses_the_same_molecule_order():
    from analysis import energy_csv, latex_report, scatter_interactive, scatter_plots
    for module in (latex_report, energy_csv, scatter_plots, scatter_interactive):
        assert module.MOL_ORDER == MOL_ORDER, module.__name__


def test_default_input_folders_are_the_shipped_results():
    from analysis import energy_csv, latex_report
    assert energy_csv.CPU_DIR == os.path.join(RESULTS, "cpu")
    assert energy_csv.GPU_DIR == os.path.join(RESULTS, "gpu")
    import sys
    argv, sys.argv = sys.argv, ["latex_report.py"]
    try:
        args = latex_report.cli()
    finally:
        sys.argv = argv
    assert args.cpu_dir == os.path.join(RESULTS, "cpu")
    assert args.gpu_dir == os.path.join(RESULTS, "gpu")
    assert args.out == os.path.join(os.path.dirname(RESULTS), "analysis", "tex_out")


def test_latex_report_pairs_all_12_molecules():
    from analysis import latex_report
    from vqe_cudaq.molecules import molecules
    cpu, gpu, matched = latex_report.load_data(os.path.join(RESULTS, "cpu"),
                                               os.path.join(RESULTS, "gpu"))
    assert sorted(matched) == sorted(molecules)
    assert matched == [m for m in MOL_ORDER if m in molecules]     # canonical order
    assert "Benzaanthracene" not in cpu and "Tetracene" in gpu


def test_latex_report_writes_every_figure(tmp_path, monkeypatch):
    """The whole report runs on the shipped results and writes standalone + embeddable .tex."""
    import sys
    from analysis import latex_report
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["latex_report.py",
                                      "--cpu_dir", os.path.join(RESULTS, "cpu"),
                                      "--gpu_dir", os.path.join(RESULTS, "gpu"),
                                      "--out", str(tmp_path / "tex_out"), "--no_table"])
    latex_report.main()
    written = os.listdir(tmp_path / "tex_out")
    assert "vqe_preamble.tex" in written
    assert any(f.startswith("fig1_") and f.endswith("_embed.tex") for f in written)
    assert len([f for f in written if f.endswith(".tex")]) > 40


def test_energy_table(tmp_path, monkeypatch):
    """energy_csv writes one row per molecule and benchmark active space."""
    import csv
    from analysis import energy_csv
    out = tmp_path / "table.csv"
    monkeypatch.setattr(energy_csv, "OUT_CSV", str(out))
    energy_csv.main()
    with open(out) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 12 * 9
    assert {r["Molecule"] for r in rows} >= {"Tetracene", "NH2-", "Cytosine"}
