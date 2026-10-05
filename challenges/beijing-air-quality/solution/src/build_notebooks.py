"""Assemble solution notebooks from the cell-marked sources in this folder.

Each notebook = header markdown + common.py + vN.py. Cells are separated by
'# %%' lines; '# %% [markdown]' starts a markdown cell whose lines are '# '-prefixed.
Usage: python build_notebooks.py   (writes ../solution_v1.ipynb etc.)
"""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).parent

HEADERS = {
    "v1": "# From-Scratch Neural Forecasting: solution v1 (feed-forward baseline)\n\n"
          "Cuts the continuous history into the same 72 h context + 24 h target episodes as the "
          "test. Every network station takes a turn as the pseudo-unmonitored target, with "
          "leave-self-out network inputs. The model is a small feed-forward network trained from "
          "scratch in PyTorch on context summaries plus the target-day weather. Validation holds "
          "out stations and the last training year.",
    "v2": "# From-Scratch Neural Forecasting: solution v2 (GRU encoder-decoder)\n\n"
          "Replaces the feed-forward network with a sequence model trained from scratch. A GRU "
          "encoder reads the full 72-hour context (masked network aggregates, site weather, site "
          "offsets). A GRU decoder, driven by the target-day weather, emits 24 h × 6 pollutants "
          "as a correction to the recent network level.",
    "v3": "# From-Scratch Neural Forecasting: solution v3 (final)\n\n"
          "Same GRU encoder-decoder as v2. It adds network-dropout augmentation (random extra "
          "stations hidden during training, kept only if validation improves) and a 5-seed "
          "ensemble trained on all samples.",
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
