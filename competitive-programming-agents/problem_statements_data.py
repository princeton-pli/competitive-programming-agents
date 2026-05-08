import copy
import logging
from pathlib import Path
import time
from tqdm import tqdm
import json

from select_problems import get_clean_problem_dict, get_problem_statements_for_contest


def get_instance_id(data: dict) -> str:
    return str(data["contestId"]) + data["index"]


def to_swebench_format(data: dict) -> dict:
    data_extra = copy.deepcopy(data)
    data_extra.pop("statement")
    data_extra["contestId"] = str(data_extra["contestId"])
    return {
        "instance_id": get_instance_id(data),
        "problem_statement": json.dumps(data),  # The models need to have access to examples and to contest and problem ids
        "image_name": "swea-cf-tiny",
        "repo_name": "./SWE-agent",
        "extra_fields": data_extra,
    }


def add_to_dataset(name, contest_ids):
    dataset_path = Path(f"./data/{name}.json")
    dataset = json.load(open(dataset_path)) if dataset_path.exists() else []

    instance_ids = set(instance["instance_id"] for instance in dataset)

    for contest_id in tqdm(contest_ids, desc="Processing contests"):
        problem_statements = get_problem_statements_for_contest(contest_id)["napiProblemStatements"]
        for raw_problem in problem_statements:
            problem_dict = get_clean_problem_dict(raw_problem)
            swebench_style_problem_dict = to_swebench_format(problem_dict)
            if swebench_style_problem_dict["instance_id"] in instance_ids:
                continue
            dataset.append(swebench_style_problem_dict)
        time.sleep(0.2)

    with open(dataset_path, "w") as f:
        json.dump(dataset, f, indent=2)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    # contest_ids = list_recent_contests(num=100, div=div)
    contest_ids = {
        "div3": [2051, 2060, 2072, 2074, 2091, 2093, 2106, 2114, 2117, 2121],
        "div2": [2102, 2104, 2107, 2108, 2109, 2110, 2111, 2113, 2116, 2118],
        "div1": [2061, 2062, 2066, 2077, 2084, 2089, 2096, 2097, 2101, 2115],
    }

    for div in [3, 2, 1]:
        add_to_dataset(f"div{div}", contest_ids[f"div{div}"])


if __name__ == "__main__":
    main()