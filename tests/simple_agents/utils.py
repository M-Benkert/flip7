import json
from pathlib import Path
from typing import Any

import numpy as np

NUMBER_OF_ITERATIONS = 1_000_000
RESULT_PATH = Path(__file__).parent / "results"


def calculate_statistics(scores: list[int]) -> dict:
    return {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "max": float(np.max(scores)),
        "count": len(scores),
        "count_0": sum(1 for s in scores if s == 0),
        "count_0_10": sum(1 for s in scores if 0 < s <= 10),
        "count_10_20": sum(1 for s in scores if 10 < s <= 20),
        "count_20_30": sum(1 for s in scores if 20 < s <= 30),
        "count_30_40": sum(1 for s in scores if 30 < s <= 40),
        "count_40_50": sum(1 for s in scores if 40 < s <= 50),
        "count_above_50": sum(1 for s in scores if s > 50),
    }


def store_results(scores: list[int], agent_name: str, mark: str, agent_params: dict[str, Any]):
    stats = {
        "agent": agent_name,
        "parameters": agent_params,
    } | calculate_statistics(scores)

    # Store statistics in JSON file
    result_path = RESULT_PATH / agent_name
    result_path.mkdir(exist_ok=True)

    result_file = result_path / f"{mark}.json"
    with result_file.open("w") as f:
        json.dump(stats, f, indent=4)
