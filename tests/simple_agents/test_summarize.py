import json

import pandas as pd

from tests.simple_agents.utils import RESULT_PATH


def test_summarize():
    # List alls directories in the results path
    result_dirs = [d for d in RESULT_PATH.iterdir() if d.is_dir()]

    # load all json files in the result directories and concatenate them into a single DataFrame
    all_results = []
    for result_dir in result_dirs:
        json_files = list(result_dir.glob("*.json"))
        for json_file in json_files:
            with open(json_file, "r") as f:
                data = json.load(f)
                data["parameters"] = json.dumps(data["parameters"])
                all_results.append(data)

    df = pd.DataFrame(all_results)

    # Save the DataFrame to a CSV fil
    output_path = RESULT_PATH / "summary.csv"
    df.to_csv(output_path, index=False)
