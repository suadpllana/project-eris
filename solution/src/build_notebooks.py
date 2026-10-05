"""Assemble solution notebooks from the cell-marked sources in this folder.

Each notebook = header markdown + common.py + vN.py. Cells are separated by
'# %%' lines; '# %% [markdown]' starts a markdown cell whose lines are '# '-prefixed.
Usage: python build_notebooks.py   (writes ../solution_v1.ipynb etc.)
"""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).parent

HEADERS = {
    "v1": "# Cold-Start Air-Quality Forecasting: solution v1 (baseline)\n\n"
          "Cuts the continuous training period into the same 72 h context + 24 h target "
          "episodes as the test. Every network station takes a turn as a pseudo-unmonitored "
          "target, with leave-self-out network features. Validation holds out stations and the "
          "last training year. Model: one LightGBM per pollutant on calendar features, "
          "target-hour weather and network context levels.",
    "v2": "# Cold-Start Air-Quality Forecasting: solution v2 (richer features)\n\n"
          "Builds on v1. It adds target-day weather summaries, weather change against the "
          "context (clean-up fronts), the site's weather relative to the network (site offsets), "
          "network trend and diurnal profile, and cross-pollutant signals (Ox, PM2.5/PM10).",
    "v3": "# Cold-Start Air-Quality Forecasting: solution v3 (final)\n\n"
          "Same features as v2. It adds a second model family that forecasts the change from the "
          "recent network level, blends it with the direct model using weights chosen on "
          "validation only, and reports diagnostics by station and forecast hour.",
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
