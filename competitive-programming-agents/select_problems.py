from collections import defaultdict
import base64
import json
import logging
import os
import requests
import shutil
import subprocess

from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CF_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_problem_statements_for_contest(contest_id):
    url = f"https://codeforces.com/api/v2/problemStatements?contestId={contest_id}"
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    attempts_num = 10

    for attempt in range(attempts_num):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt == attempts_num - 1:  # Last attempt
                logging.error(f"Failed to fetch problem statements for contest {contest_id} after 5 attempts: {e}")
                raise
            logging.warning(f"Attempt {attempt + 1} failed for contest {contest_id}: {e}")
            continue


def get_clean_problem_dict(raw_problem):
    problem_dict = dict()
    contest_id = raw_problem["contestId"]
    problem_dict["contestId"] = contest_id
    problem_dict["index"] = raw_problem["index"]
    problem_dict["name"] = raw_problem["englishName"]
    for text_dict in raw_problem["texts"]:
        if text_dict["language"] == "en":
            statement = text_dict["content"]
            problem_dict["statement"] = statement
            problem_dict["examples"] = dict()
            for resource_dict in text_dict["resources"]:
                resource_name = resource_dict["name"]
                if (
                        resource_name.startswith("example")
                        and not resource_name.endswith(".mu")
                        and not resource_name.endswith(".png")
                ):
                    try:
                        resource_content = base64.b64decode(resource_dict["content"]).decode("utf-8")
                        problem_dict["examples"][resource_name] = resource_content
                    except Exception as e:
                        print(f"Error decoding resource {resource_name} for contest {contest_id} problem {problem_dict['index']}: {e}")
                        continue
    return problem_dict


def get_problem_texts_with_examples(contest_id, problem_indices):
    print(f"Contest {contest_id}")
    problem_statements_list = get_problem_statements_for_contest(contest_id)["napiProblemStatements"]
    for raw_problem in problem_statements_list:
        if raw_problem["index"] not in problem_indices:
            continue
        problem_dir = os.path.join("cf_submissions", f"{contest_id}{raw_problem['index']}")
        os.makedirs(problem_dir, exist_ok=True)
        with open(os.path.join(problem_dir, "raw_problem.json"), "w") as f:
            json.dump(raw_problem, f)
        problem_dict = get_clean_problem_dict(raw_problem)
        with open(os.path.join(problem_dir, "statement.json"), "w") as f:
            json.dump(problem_dict, f)


def download_and_save_all_problem_statements():
    with open("submissions.json", "r") as f:
        submissions = json.load(f)

    submission_cnt = 0
    submission_list_per_problem = defaultdict(list)
    for contest_id, contest_submissions in submissions.items():
        submission_cnt += len(contest_submissions)
        for submission in contest_submissions:
            problem = submission["problem"]
            submission_list_per_problem[(problem["contestId"], problem["index"])].append(submission)

    print(f"Total submission count: {submission_cnt}")
    print(f"Total problem count: {len(submission_list_per_problem)}")

    for contest_id in submissions.keys():
        problem_indices = set([
            submission["problem"]["index"] for submission in submissions[contest_id]
        ])
        get_problem_texts_with_examples(contest_id, problem_indices)


def decide_whether_sol_is_unique(problem_dict):
    # client = OpenAI(api_key=OPENAI_API_KEY)

    # completion = client.chat.completions.create(
    #   model="o1",
    #   messages=[
    #     {
    #         "role": "user",
    #         "content": "I will give you a competitive programming problem statement with examples, in json. Your task is to figure out whether the correct output is unique or not. "
    #         "Examples are in fields like 'example.01' and the output is 'example.01.out'. Please pay attention to them, and make sure you understand the output format (it can be tricky: for example, if you are asked to find the length of the maximum sequence with some property, the output is the length, not the sequence itself; the sequences may not be unique, but the maximum length is). "
    #         "If there are notes in the text saying something like 'if there are many possible answers, output any of them', then the output is likely not unique. "
    #         "You can write your thoughts but at the very end output 'Answer: unique' or 'Answer: not unique' and nothing else (in particular, do not include any punctuation marks or line breaks at the end). "
    #         "The json is as follows:\n" + json.dumps(problem_dict)
    #     }
    #   ]
    # )

    # response_text = completion.choices[0].message.content
    # logging.info("Got the following response from o1: %s", response_text)
    # if response_text.endswith("Answer: unique"):
    #     return True
    # if response_text.endswith("Answer: not unique"):
    #     return False
    # raise ValueError(f"Invalid response from o1: {response_text}")
    print(problem_dict['statement'])
    ans = input("Is the solution unique? (y/n)")
    return ans == "y"


def filter_problems_with_unique_solutions():
    good_problems = []

    for num, problem_name in enumerate(os.listdir("cf_submissions")):
        problem_dir = os.path.join("cf_submissions", problem_name)
        statement_filepath = os.path.join(problem_dir, "statement.json")
        if not os.path.exists(statement_filepath):
            continue
        with open(statement_filepath, "r") as f:
            problem_dict = json.load(f)
        logging.info(f"Checking {problem_name}")
        if decide_whether_sol_is_unique(problem_dict):
            good_problems.append(problem_name)
            shutil.move(problem_dir, os.path.join("cf_submissions", "unique", problem_name))
        else:
            # move the dir to a new folder called "not_unique"
            shutil.move(problem_dir, os.path.join("cf_submissions", "not_unique", problem_name))
        if num > 10:
            break


def collect_pairs_of_submissions():
    submission_cnt = 0
    problem_user_to_submissions_dct = defaultdict(list)
    for submissions_filename in [
        "submissions_2025.json",
        "submissions_2024.json",
        "submissions_2023.json",
        "submissions_2022.json",
        "submissions_2021.json",
        "submissions_2020.json",
    ]:
        with open(submissions_filename, "r") as f:
            submissions = json.load(f)
        for contest_id, contest_submissions in submissions.items():
            for submission in contest_submissions:
                if "C++" not in submission["programmingLanguage"]:
                    continue
                submission_cnt += 1
                problem = submission["problem"]
                problem_index = problem["index"]
                problem_user_to_submissions_dct[(contest_id, problem_index, submission["author"]["members"][0]["handle"])].append(submission)

    pairs_cnt = 0
    problem_user_pair = dict()
    for problem_user, submissions in problem_user_to_submissions_dct.items():
        if len(submissions) < 2:
            continue
        submissions.sort(key=lambda x: x["creationTimeSeconds"])
        # select the first WA and the last AC
        first_wa = None
        last_ac = None
        for submission in submissions:
            if submission['verdict'] == "WRONG_ANSWER" and first_wa is None:
                first_wa = submission
            if submission['verdict'] == "OK":
                last_ac = submission
        if first_wa is None or last_ac is None:
            continue
        if first_wa['creationTimeSeconds'] > last_ac['creationTimeSeconds']:
            continue
        contest_id, problem_index, user_name = problem_user
        problem_name = f"{contest_id}{problem_index}"
        if problem_name not in problem_user_pair:
            problem_user_pair[problem_name] = {user_name: (first_wa, last_ac)}
        else:
            problem_user_pair[problem_name][user_name] = (first_wa, last_ac)
        pairs_cnt += 1

    # for problem_name in problem_user_pair.keys():
    #     for user_name in problem_user_pair[problem_name].keys():
    #         value = problem_user_pair[problem_name][user_name]
    #         assert len(value) == 2
    #         assert value[0]['verdict'] == "WRONG_ANSWER"
    #         assert value[1]['verdict'] == "OK"
    #         assert value[0]['creationTimeSeconds'] < value[1]['creationTimeSeconds']

    with open("problem_user_pair.json", "w") as f:
        json.dump(problem_user_pair, f)
    print(f"Total pairs count: {pairs_cnt}")


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    # collect_pairs_of_submissions()

    # print(f"Total submission count: {submission_cnt}")
    # print(f"Total cpp submission count: {cpp_submission_cnt}")
    # print(f"Total problem count: {len(problem_set)}")

    # download_and_save_all_problem_statements()
    # filter_problems_with_unique_solutions()
    # create_contest_folders("2089", "..")
    for contest_id in [
        2107, 2108, 2104, 2097, 2098, 2103, 2096, 2084,
        2086, 2092, 2089, 2090, 2085, 2075, 2077, 2078,
        2071, 2070, 2069, 2064, 2066, 2067, 2059, 2062,
        2063, 2061, 2056, 2055,
    ]:
        create_contest_folders(contest_id, "..")


if __name__ == "__main__":
    main()
