
from pathlib import Path
import json
import copy

OUTPUT_PATH = "cf_swebench_style_test.json"

def main():
    extract_dir = Path(".") / "cf_submissions"
    assert extract_dir.exists()

    statements = list(extract_dir.glob("**/statement.json"))
    print(f"Found {len(statements)} statements")
    data = [json.loads(f.read_text()) for f in statements]

    sweb_data = []
    for d in data:
        try:
            sweb_data.append(to_swebench_format(d))
        except Exception as e:
            print(f"Error processing {d['name']}: {e}")
            continue

    Path(OUTPUT_PATH).write_text(json.dumps(sweb_data, indent=2))


def get_instance_id(data: dict) -> str:
    name = data["name"].lower()
    suffix = name.split()[0]
    if suffix in ["a", "the", "yet", "another", "old", "new", "one", "two", "first", "last", "another", "many"] and len(name.split()) > 1:
        suffix = name.split()[1]
    # Ensure suffix only has alphanumeric characters
    suffix = ''.join(c for c in suffix if c.isalnum())
    id_ = data["contestId"] + data["index"].lower()
    return f"{id_}_{suffix}"

def to_swebench_format(data: dict) -> dict:
    data_extra = copy.deepcopy(data)
    data_extra.pop("statement")
    return {
        "instance_id": get_instance_id(data),
        "problem_statement": data["statement"],
        "image_name": "swea-cf-tiny",
        "repo_name": "./SWE-agent",
        "extra_fields": data_extra,
    }


if __name__ == "__main__":
    main()
