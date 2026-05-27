import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def to_jsonable(value):
    # convert numpy-heavy experiment payloads into json-safe python objects.
    # many of the gaussian studies save arrays, scalars, and nested dicts side by side.
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_csv(path, fieldnames: list[str], rows) -> None:
    # write a csv with a fixed column order for paper-facing experiment artifacts.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path, payload, *, indent: int = 2, sort_keys: bool = True) -> None:
    # write a json artifact after converting numpy values into native python objects.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=int(indent), sort_keys=bool(sort_keys))


def write_markdown(path, lines: list[str]) -> None:
    # write a markdown artifact from a list of already-formatted lines.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    if text and not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")


def save_dual_figure(fig, save_stem: str, dpi: int) -> None:
    # save the same matplotlib figure as both pdf and png for the paper figures.
    out_dir = os.path.dirname(save_stem)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{save_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{save_stem}.png", dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
