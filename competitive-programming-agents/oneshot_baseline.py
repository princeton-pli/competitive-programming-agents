#!/usr/bin/env python3

import argparse
import json
import logging
import os
import threading
import random
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import jinja2
import litellm
import pydantic
import requests
import yaml
from dotenv import load_dotenv
from tqdm import tqdm


litellm.suppress_debug_info = True
# Drop unsupported OpenAI parameters.
litellm.drop_params = True


class CFRateLimiter:
    def __init__(self, wait_seconds: float):
        self.wait = float(wait_seconds)
        self._next_earliest = 0.0
        self._lock = threading.Lock()

    def acquire(self):
        while True:
            with self._lock:
                now = time.monotonic()
                to_sleep = self._next_earliest - now
                if to_sleep <= 0:
                    base = max(self._next_earliest, now)
                    self._next_earliest = base + self.wait
                    return
            time.sleep(to_sleep)


cf_rate_limiter = CFRateLimiter(wait_seconds=40)


class NoCodeFoundError(Exception):
    pass


class Config(pydantic.BaseModel):
    system_template: str
    instance_template: str


class DatasetInstance(pydantic.BaseModel):
    instance_id: str
    problem_statement: str
    image_name: str
    repo_name: str
    extra_fields: dict[str, Any]


class OutputRecord(pydantic.BaseModel):
    config: Config
    query: list[dict[str, str]]
    response: str
    code: str
    exit_status: str  # "success", "error", or "failure"
    cost: float
    problem: DatasetInstance
    # Type checking does not allow str | None (?)
    error: Any # str | None
    model: str


def load_config(config_path: Path) -> Config:
    """Load and validate configuration from YAML file."""
    config_data = yaml.safe_load(config_path.read_text())
    return Config(**config_data)


def load_dataset(dataset_path: Path) -> list[DatasetInstance]:
    """Load dataset from JSON file."""
    # Note only json for python>=3.7 preserves order in loads.
    dataset_data = json.loads(dataset_path.read_text())
    return [DatasetInstance(**item) for item in dataset_data]


def load_and_process_dataset(
    dataset_path: Path,
    *,
    shuffle: bool = False,
    shuffle_seed: int = 42,
    slice_str: str = None,
    instance_filter: list[str] = None,
    level_filter: list[str] = None,
    contest_id_filter: list[str] = None,
) -> list[DatasetInstance]:
    """Load dataset and apply all processing steps: shuffle, instance filter, level filter, contest_id filter, then slice."""
    # Load dataset
    dataset = load_dataset(dataset_path)

    # Apply shuffle (first)
    if shuffle:
        random.seed(shuffle_seed)
        dataset = dataset.copy()  # Don't modify original list
        random.shuffle(dataset)

    # Apply instance filter (second)
    if instance_filter is not None:
        # Convert to set for faster lookup
        filter_set = set(instance_filter)
        dataset = [instance for instance in dataset if instance.instance_id in filter_set]

    # Apply level filter (third)
    if level_filter is not None:
        # Convert to set for faster lookup
        filter_set = set(level_filter)
        dataset = [instance for instance in dataset if instance.extra_fields["index"] in filter_set]

    # Apply contest_id filter (fourth)
    if contest_id_filter is not None:
        # Convert to set for faster lookup
        filter_set = set(contest_id_filter)
        dataset = [instance for instance in dataset if instance.extra_fields["contestId"] in filter_set]

    # Apply slice (last)
    if slice_str is not None:
        # Parse slice string (e.g., "10:20", ":10", "10:", ":")
        parts = slice_str.split(":")
        start = int(parts[0]) if parts[0] else None
        end = int(parts[1]) if len(parts) > 1 and parts[1] else None
        dataset = dataset[start:end]

    return dataset


def format_messages(config: Config, instance: DatasetInstance) -> list[dict[str, str]]:
    """Format system and user messages using Jinja templates."""
    jinja_env = jinja2.Environment()

    system_template = jinja_env.from_string(config.system_template)
    instance_template = jinja_env.from_string(config.instance_template)

    system_message = system_template.render()
    user_message = instance_template.render(problem_statement=instance.problem_statement)

    return [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]


def query_model(
        messages: list[dict[str, str]], model: str, args: argparse.ArgumentParser
) -> tuple[str, float]:
    """Query the language model and return response and cost."""
    response = litellm.completion(model=model, messages=messages, temperature=args.temperature)

    content = response.choices[0].message.content  # type: ignore
    cost = litellm.completion_cost(completion_response=response)  # type: ignore

    return content, cost  # type: ignore


def extract_code(response: str) -> str:
    """Extract code from <code> tags or fenced code blocks."""
    code_pattern = r"<code>(.*?)</code>"
    match = re.search(code_pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    fenced_pattern = r"```(?:[a-zA-Z0-9_+-]+)?\s*(.*?)```"
    match = re.search(fenced_pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    raise NoCodeFoundError("No code found in response")


def submit_code(contest_id: str, problem_index: str, code: str) -> str:
    """Submit code to Codeforces API and return submission ID."""
    url = "https://codeforces.com/api/v2/submissions"

    api_key = os.getenv("CF_API_KEY")
    if not api_key:
        raise ValueError("CF_API_KEY environment variable not set")

    data = {
        "contestId": contest_id,
        "problemIndex": problem_index,
        "sourceType": "cpp.gcc14-64-msys2-g++23",
        "sourceText": code,
    }

    headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.post(url, headers=headers, json=data)
    response_data = response.json()

    if "napiSubmissionId" not in response_data:
        raise ValueError(f"Submission failed: {response_data}")

    return response_data["napiSubmissionId"]


def check_submission_status(submission_id: str) -> str:
    """Check submission status and return status: 'success' or 'failed_submission'."""
    logger = logging.getLogger(__name__)

    url = "https://codeforces.com/api/v2/submissionReports"

    api_key = os.getenv("CF_API_KEY")
    if not api_key:
        raise ValueError("CF_API_KEY environment variable not set")

    data = {
        "napiSubmissionIds": submission_id,
        "details": True,
    }

    headers = {"Authorization": f"Bearer {api_key}"}

    retry_ladder = [30, 30, 60, 60, 120, 120]
    retry_index = 0
    final_retry_interval = 240

    while True:
        response = requests.post(url, headers=headers, json=data)
        submission_reports = response.json()["napiSubmissionReports"]

        if len(submission_reports) != 1:
            raise ValueError("Expected exactly one submission report")

        submission_report = submission_reports[0]
        logger.info(f"Submission report: {submission_report}")

        if submission_report["status"] != "COMPLETED":
            if retry_index < len(retry_ladder):
                wait_time = retry_ladder[retry_index]
                retry_index += 1
            else:
                wait_time = final_retry_interval

            logger.info(f"Submission not completed, waiting {wait_time} seconds before retry")
            time.sleep(wait_time)
            continue

        return "success" if submission_report["verdict"] == "OK" else "failure"


def save_result(
        output_record: OutputRecord,
        output_dir: Path,
        contest_id: str,
        problem_index: str,
        status: str,
    ) -> None:
    """Save result to JSON file with unix timestamp."""
    timestamp = str(int(time.time() * 1000000))  # microsecond precision
    output_path = output_dir / contest_id / f"{problem_index}_{timestamp}_{status}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(output_record.model_dump_json(indent=2))


def process_single_attempt(
        instance: DatasetInstance, config: Config, model: str, output_dir: Path, contest_id: str, problem_index: str,
        args: argparse.ArgumentParser,
        how_many_retries: int = 6,
) -> tuple[dict[str, Any], float]:
    """Process a single attempt for an instance. Returns (exit status count, cost)."""
    logger = logging.getLogger(__name__)

    # Initialize data dictionary to build up all information
    data = {
        "config": config,
        "query": [],
        "response": "",
        "code": "",
        "exit_status": "error",  # Default to error, will be updated on success
        "cost": 0.0,
        "problem": instance,
        "error": None,
        "model": model,
        "submission_id": None,
    }

    cost = 0.0

    for attempt in range(how_many_retries):
        try:
            # Format messages
            data["query"] = format_messages(config, instance)

            # Query model
            response, cost = query_model(data["query"], model, args=args)
            data["response"] = response
            data["cost"] = cost

            logger.info(f"Model response for {instance.instance_id}: cost=${cost:.4f}, response_length={len(response)}")
            logger.info(f"Full model response for {instance.instance_id}: {response}")

            # Extract code
            data["code"] = extract_code(response)

            # Submit and check
            cf_rate_limiter.acquire()
            submission_id = submit_code(contest_id, problem_index, data["code"])
            time.sleep(25)
            status = check_submission_status(submission_id)
            data["exit_status"] = status

            data["submission_id"] = submission_id

            logger.info(f"Submission for {instance.instance_id}: status={status}, submission_id={submission_id}")
            break

        except Exception as e:
            if attempt == how_many_retries - 1:
                # On exception, just set error status and capture what we can
                data["exit_status"] = "error"
                data["error"] = traceback.format_exc()
                if not data["response"]:
                    data["response"] = str(e)

                logger.error(f"Exception in {instance.instance_id}: {str(e)}")
                logger.debug(f"Full traceback for {instance.instance_id}: {data['error']}")
            else:
                logger.warning(f"Exception in {instance.instance_id}: {str(e)}")
                logger.debug(f"Full traceback for {instance.instance_id}: {data['error']}")
                time.sleep(30)

    # Always save the result for this attempt
    output_record = OutputRecord(**data)
    save_result(output_record, output_dir, contest_id, instance.extra_fields["index"], data["exit_status"])

    return data, cost


def process_instance(
        instance: DatasetInstance, config: Config, model: str, output_dir: Path, args: argparse.ArgumentParser, k: int = 1
) -> tuple[dict[str, int], float]:
    """Process a single dataset instance with k attempts. Returns (status_counts, total_cost)."""
    contest_id = instance.extra_fields["contestId"]
    problem_index = instance.extra_fields["index"]
    total_cost = 0.0
    status_counts = {"success": 0, "failure": 0, "error": 0}

    for _attempt in range(k):
        data, cost = process_single_attempt(
            instance, config, model, output_dir, contest_id, problem_index,
            args=args
        )
        total_cost += cost
        status_counts[data["exit_status"]] += 1

    return status_counts, total_cost


def get_cli() -> argparse.ArgumentParser:
    """Set up and return the command line argument parser."""
    parser = argparse.ArgumentParser(description="Generate and evaluate LM on competitive programming dataset")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML file")
    parser.add_argument(
        "--dataset", type=Path, required=False, help="Path to dataset JSON file", default=Path("cf_swebench_style.json")
    )
    parser.add_argument("--k", type=int, default=1, help="Number of attempts per problem")
    parser.add_argument("--temperature", type=float, default=0., help="Temperature for LLM decoding.")
    parser.add_argument(
        "--shuffle", action="store_true", help="Shuffle dataset before applying slice (deterministic with fixed seed)"
    )
    parser.add_argument("--shuffle_seed", type=int, default=42, help="Seed for deterministic shuffling")
    parser.add_argument("--slice", type=str, help="Dataset slice (e.g., '10:20', ':10', '10:')")
    parser.add_argument(
        "--instance_filter", type=str, nargs="+", help="List of instance IDs to evaluate (applied after slice)"
    )
    parser.add_argument(
        "--level_filter",
        type=str,
        nargs="+",
        help="List of problem levels/indices to evaluate (matches extra_fields['index'])",
    )
    parser.add_argument(
        "--contest_id_filter",
        type=str,
        nargs="+",
        help="List of contest IDs to evaluate (matches extra_fields['contestId'])",
    )
    parser.add_argument("--model", type=str, required=True, help="Model name for litellm")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")

    return parser


def main() -> None:
    """Main function."""
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv()
    else:
        logger.warning("No .env file found. Please set all language model API keys as environment variables.")
        print("Warning: No .env file found. Please set all language model API keys as environment variables.")
        print("For example: export OPENAI_API_KEY=your_key_here")

    parser = get_cli()
    args = parser.parse_args()

    # Set up logging
    log_dir = Path(args.output) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"oneshot_baseline_{datetime.now().isoformat().replace(':', '-')}.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    # Load and process dataset
    config = load_config(args.config)
    dataset = load_and_process_dataset(
        args.dataset,
        shuffle=args.shuffle,
        shuffle_seed=args.shuffle_seed,
        slice_str=args.slice,
        instance_filter=args.instance_filter,
        level_filter=args.level_filter,
        contest_id_filter=args.contest_id_filter,
    )

    logger.info(f"Args: {args}")
    logger.info(f"Config: {config}")

    # Process instances with progress tracking
    total_cost = 0.0
    all_attempts_cnt = 0
    overall_status_counts = {"success": 0, "failure": 0, "error": 0}
    overall_dict_of_statuses = {}

    with tqdm(dataset, desc="Processing instances") as pbar:
        for instance in pbar:
            # TODO: re-order args.
            instance_status_counts, cost = process_instance(instance, config, args.model, args.output, args, args.k)
            overall_dict_of_statuses[instance.instance_id] = instance_status_counts

            total_cost += cost

            # Accumulate status counts from this instance
            for status, count in instance_status_counts.items():
                overall_status_counts[status] += count
                all_attempts_cnt += count

            pbar.set_postfix(
                {
                    "success": f"{overall_status_counts['success']}/{all_attempts_cnt}",
                    "failed": f"{overall_status_counts['failure']}/{all_attempts_cnt}",
                    "error": f"{overall_status_counts['error']}/{all_attempts_cnt}",
                    "cost": f"${total_cost:.4f}",
                }
            )

    logger.info(f"Overall status dict:\n{overall_dict_of_statuses}")
    logger.info(f"Final results:")
    logger.info({
        "success": f"{overall_status_counts['success']}/{all_attempts_cnt}",
        "failed": f"{overall_status_counts['failure']}/{all_attempts_cnt}",
        "error": f"{overall_status_counts['error']}/{all_attempts_cnt}",
        "cost": f"${total_cost:.4f}",
    })


if __name__ == "__main__":
    main()
