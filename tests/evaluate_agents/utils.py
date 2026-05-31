import json
from pathlib import Path
from typing import Any

import numpy as np

NUMBER_OF_ITERATIONS = int(1e6)
RESULT_PATH = Path(__file__).parent / "results"


def calculate_statistics(scores: list[int]) -> dict:
    scores_ = np.array(scores)

    return {
        "mean": float(np.mean(scores_)),
        "median": float(np.median(scores_)),
        "max": float(np.max(scores_)),
        "count": scores_.size,
        "count_0": int(np.sum(scores_ == 0)),
        "count_0_10": int(np.sum((0 < scores_) & (scores_ <= 10))),
        "count_10_20": int(np.sum((10 < scores_) & (scores_ <= 20))),
        "count_20_30": int(np.sum((20 < scores_) & (scores_ <= 30))),
        "count_30_40": int(np.sum((30 < scores_) & (scores_ <= 40))),
        "count_40_50": int(np.sum((40 < scores_) & (scores_ <= 50))),
        "count_above_50": int(np.sum(scores_ > 50)),
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
