from pathlib import Path

import pandas as pd


def read_txt(path):
    data_set = []
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    lines = lines[3:]
    data = []
    for line in lines:
        data_line = line.strip("\n").split()
        if len(data_line) == 0:
            if len(data) != 0:
                data_set.append(data)
                data = []
        elif data_line[0].startswith("T") or data_line[0].startswith("C"):
            continue
        else:
            data.append(list(map(float, data_line)))
    return data_set


def _resolve_instance_path(file_name):
    path = Path(file_name)
    candidates = [path]
    if path.suffix == "":
        candidates.append(path.with_suffix(".xlsx"))

    for candidate in candidates:
        if candidate.is_absolute() and candidate.exists():
            return candidate
        if candidate.exists():
            return candidate

    for parent in Path(__file__).resolve().parents:
        for candidate in candidates:
            instance_path = parent / "Instance" / candidate
            if instance_path.exists():
                return instance_path

    return Path(__file__).resolve().parent.parent / "Instance" / file_name


def read_excel(file_name):
    df = pd.read_excel(_resolve_instance_path(file_name))
    return [list(row) for _, row in df.iterrows()]
