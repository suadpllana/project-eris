"""Assemble solution notebooks from the cell-marked sources in this folder.

Each notebook = header markdown + common.py + vN.py. Cells are separated by
'# %%' lines; '# %% [markdown]' starts a markdown cell whose lines are '# '-prefixed.
Usage: python build_notebooks.py   (writes ../solution_v1.ipynb etc.)
"""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).parent

HEADERS = {
    "v1": "# Beijing Virtual Air-Quality Stations: solution v1 (baseline)\n\n"
          "First full pipeline. It scores a simple spatial baseline under leave-one-station-out "
          "validation, then trains one LightGBM model per pollutant on calendar features, local "
          "weather and leave-self-out network aggregates.",
    "v2": "# Beijing Virtual Air-Quality Stations: solution v2 (temporal context)\n\n"
          "Builds on v1. It adds temporal context from the network (centred rolling means, "
          "lags and leads), 24-hour wind and rain summaries, and the cross-pollutant Ox = NO2 + O3 "
          "signal.",
    "v3": "# Beijing Virtual Air-Quality Stations: solution v3 (final)\n\n"
          "Same features as v2. It trains two complementary LightGBM models per pollutant, one on "
          "the log concentration and one on the station's offset from the leave-self-out network "
          "mean. It blends them with weights chosen on leave-one-station-out predictions and "
          "reports per-station diagnostics. "
          "Things tried during development that did **not** help under LOSO (and so are left out): "
          "nearest-neighbour readings picked by weather similarity, proximity-weighted network "
          "means, station-level climate descriptors, and heavier regularisation.",
}


def cells_from(path):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        text = "\n".join(buf).strip("\n")
        if not text:
            return
        if kind == "md":
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.splitlines())
            cells.append(nbf.v4.new_markdown_cell(text))
        else:
            cells.append(nbf.v4.new_code_cell(text))

    for line in Path(path).read_text().splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("md" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    return cells


def main():
    for name, header in HEADERS.items():
        src = HERE / f"{name}.py"
        if not src.exists():
            continue
        nb = nbf.v4.new_notebook()
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nb.cells = [nbf.v4.new_markdown_cell(header)] + cells_from(HERE / "common.py") + cells_from(src)
        out = HERE.parent / f"solution_{name}.ipynb"
        nbf.write(nb, out)
        print("wrote", out, len(nb.cells), "cells")


if __name__ == "__main__":
    main()
