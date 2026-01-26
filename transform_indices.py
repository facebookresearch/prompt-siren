#!/usr/bin/env python3
"""Transform indices_v2.json to group lines by filename."""

import json
from collections import defaultdict


def transform_indices(input_path: str, output_path: str) -> None:
    with open(input_path) as f:
        data = json.load(f)

    transformed = {}

    for key, entries in data.items():
        # Group lines by filename
        lines_by_file: dict[str, list[int]] = defaultdict(list)
        for filename, line in entries:
            lines_by_file[filename].append(line)

        # Convert to list of objects with sorted lines
        transformed[key] = [
            {"filename": filename, "lines": sorted(lines)}
            for filename, lines in lines_by_file.items()
        ]

    with open(output_path, "w") as f:
        json.dump(transformed, f, indent=4)


if __name__ == "__main__":
    transform_indices("indices_v2.json", "indices_transformed.json")
