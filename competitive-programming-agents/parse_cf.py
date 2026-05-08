from bs4 import BeautifulSoup
import datetime
from collections import defaultdict
import logging
import os
from pathlib import Path
import quopri
import re
import requests
import json
import time
import shutil

import pyperclip
from tqdm import tqdm


CF_SUBMISSIONS_DIR = "cf_submissions"


def get_contest_list():
    url = "https://codeforces.com/api/contest.list?gym=false"
    response = requests.get(url)
    data = response.json()
    return data["result"]


def get_contest_standings(contest_id, from_position=1, count=10):
    """
    Fetch the contest standings for a given Codeforces contest,
    starting at position 'from_position' and retrieving 'count' rows.
    """
    # Construct the API URL.
    # For example: https://codeforces.com/api/contest.standings?contestId=566&from=1&count=10
    url = (
        "https://codeforces.com/api/contest.standings"
        f"?contestId={contest_id}"
        f"&from={from_position}"
        f"&count={count}"
    )

    # Send request to Codeforces API
    response = requests.get(url)
    data = response.json()

    if data["status"] != "OK":
        logging.error(f"Error: {data.get('comment', 'Unable to fetch standings')}")
        return None

    return data["result"]


def _get_interesting_submissions_for_contest(contest_id, min_rejected_attempts_count=3):
    """
    Returns a dictionary handle -> list of problem indices this user solved and that are interesting to consider.
    """
    result = get_contest_standings(contest_id, count=1000)

    interesting_submissions = defaultdict(list)

    # You can explore fields inside 'result'
    contest_info = result["contest"]
    problems = result["problems"]
    rows = result["rows"]  # This contains participants and their ranks

    logging.debug("Contest Information:")
    logging.debug("  ID: %s", contest_info["id"])
    logging.debug("  Name: %s", contest_info["name"])
    logging.debug("  Type: %s", contest_info["type"])
    logging.debug("  Phase: %s", contest_info["phase"])

    problem_indices = []

    logging.debug("Problems:")
    for problem in problems:
        logging.debug(
            "  %s: %s (Rating: %s)",
            problem["index"],
            problem["name"],
            problem.get("rating", "N/A"),
        )
        # if 2000 < problem.get("rating", 0) < 2800:
        problem_indices.append(problem["index"])

    logging.debug("Top Standings:")
    for row in rows:
        party = row["party"]
        handles = [member["handle"] for member in party["members"]]
        rank = row["rank"]
        points = row["points"]

        assert len(handles) == 1

        logging.debug("  Rank %s: %s with %s points", rank, handles[0], points)
        assert len(row["problemResults"]) == len(problems)

        for problem_num, problem in enumerate(row["problemResults"]):
            problem_ind = problems[problem_num]["index"]
            if problem_ind not in problem_indices:
                continue
            points_received = problem["points"]
            rejected_attempts_count = problem["rejectedAttemptCount"]
            if (
                points_received > 0
                and rejected_attempts_count >= min_rejected_attempts_count
            ):
                interesting_submissions[handles[0]].append(problem_ind)

    return interesting_submissions


def get_user_submissions(
    contest_id,
    handle,
    problem_indices,
    from_position=1,
    count=1000,
    min_wa_count=3,
    min_passed_tests_count=3,
):
    """
    Returns a list of submissions for a given user in a given contest, such that the submissions got WA and passed at least a few tests or AC.
    """

    url = (
        "https://codeforces.com/api/contest.status"
        f"?contestId={contest_id}"
        f"&handle={handle}"
        f"&from={from_position}"
        f"&count={count}"
    )
    response = requests.get(url)
    data = response.json()
    if data["status"] != "OK":
        logging.error("Error: %s", data.get("comment", "Unable to fetch submissions"))
        return None

    submissions_per_problem = defaultdict(list)
    submissions = []
    for submission in data["result"]:
        index = submission["problem"]["index"]
        if index not in problem_indices:
            continue
        if submission["verdict"] == "OK":
            submissions_per_problem[index].append(submission)
        if (
            submission["verdict"] == "WRONG_ANSWER"
            and submission["passedTestCount"] >= min_passed_tests_count
        ):
            submissions_per_problem[index].append(submission)

    # make sure we have at least two WA submissions for each problem:
    submissions = []
    for problem_index_, submissions_for_this_problem in submissions_per_problem.items():
        num_wa = len(
            [
                submission
                for submission in submissions_for_this_problem
                if submission["verdict"] == "WRONG_ANSWER"
            ]
        )
        num_ac = len(
            [
                submission
                for submission in submissions_for_this_problem
                if submission["verdict"] == "OK"
            ]
        )
        if num_wa >= min_wa_count or num_ac >= 1:
            submissions.extend(submissions_for_this_problem)

    return submissions


def get_all_eligible_submissions(contest_id):
    logging.info("Contest %s, analyzing standings...", contest_id)
    interesting_submissions = _get_interesting_submissions_for_contest(contest_id)
    logging.info(
        "... finished. For contest %s, will consider %s users",
        contest_id,
        len(interesting_submissions),
    )

    submissions = []
    for handle, problem_indices in tqdm(
        interesting_submissions.items(), desc="Fetching submissions", unit="user"
    ):
        user_submissions = get_user_submissions(contest_id, handle, problem_indices)
        submissions.extend(user_submissions)
        time.sleep(2)

    return submissions


def write_down_submissions(contest_ids):
    submissions_filename = "submissions.json"
    if os.path.exists(submissions_filename):
        with open(submissions_filename, "r") as infile:
            submissions = json.load(infile)
    else:
        submissions = {}

    for contest_id in contest_ids:
        if str(contest_id) in submissions:
            logging.info("Skipping %s because it already exists", contest_id)
            continue

        logging.info("Getting all eligible submissions for %s", contest_id)

        for attempt in range(5):
            try:
                submissions[contest_id] = get_all_eligible_submissions(contest_id)
                break
            except Exception as e:
                logging.error(f"Attempt {attempt+1} failed: {e}")
                if attempt < 4:
                    logging.info(f"Sleeping for 5 minutes before retry...")
                    time.sleep(300)
                else:
                    logging.error(f"All 5 attempts failed for contest {contest_id}")
                    raise

        logging.info(
            "Got %s submissions for contest %s",
            len(submissions[contest_id]),
            contest_id,
        )

        with open(submissions_filename, "w") as outfile:
            json.dump(submissions, outfile, indent=4)


def get_submission_urls(submissions):
    submission_urls = []
    for contest_id, submissions in submissions.items():
        for submission in submissions:
            submission_urls.append(
                f"https://codeforces.com/contest/{contest_id}/submission/{submission['id']}"
            )
    return submission_urls


def put_submission_urls_in_file():
    submissions = json.load(open("submissions.json"))
    submission_urls = get_submission_urls(submissions)
    with open("submission_urls.txt", "w") as outfile:
        for submission_url in submission_urls:
            print(submission_url, file=outfile)


def parse_submission_file(content):
    """
    Parses Codeforces submission HTML (possibly quoted-printable from MHTML)
    and returns:
        (problem_identifier, verdict_string, language_string)
    For example:
        ("2046C - 24", "Wrong answer on pretest 2", "C++20 (GCC 13-64)")
    """

    # 1) Decode Quoted-Printable text => real HTML
    decoded = quopri.decodestring(content)
    html_content = decoded.decode("utf-8", errors="replace")

    soup = BeautifulSoup(html_content, "html.parser")

    # 2) Find the <td> that contains the problem link, e.g. /contest/NNNN/problem/...
    td_with_problem = None
    for td in soup.find_all("td", class_="bottom dark"):
        link = td.find("a", href=re.compile(r"/contest/\d+/problem/"))
        if link:
            td_with_problem = td
            break
    if not td_with_problem:
        raise ValueError(
            "Could not find the <td> containing the problem link in the HTML."
        )

    # Extract problem code, e.g. "2066C"
    link = td_with_problem.find("a", href=re.compile(r"/contest/\d+/problem/"))
    problem_code = link.get_text(strip=True)

    # Extract revision if present, e.g. <span title="problem revision">23</span>
    revision_span = td_with_problem.find("span", title="problem revision")
    revision_str = revision_span.get_text(strip=True) if revision_span else ""

    # Combine them into something like "2066C - 23"
    if revision_str:
        problem_identifier = f"{problem_code} - {revision_str}"
    else:
        problem_identifier = problem_code

    # 3) Extract the language from the same row
    # Get the <tr> that holds this <td>, then get all the <td class="bottom dark"> in that row
    tr = td_with_problem.find_parent("tr")
    if not tr:
        raise ValueError("Could not find the parent <tr> for the problem cell.")
    tds_in_row = tr.find_all("td", class_="bottom dark")

    if len(tds_in_row) < 5:
        raise ValueError(
            "This row doesn't have enough <td> cells to contain a language/verdict."
        )

    language_td = tds_in_row[2]
    language_text = language_td.get_text(strip=True)

    # 4) Extract the verdict
    # The 5th <td> in that row typically shows the verdict, possibly containing something like:
    # <span class="verdict-rejected">Time limit exceeded on pretest <span>5</span></span>
    # We can either find it specifically in tds_in_row[4], or just do our original approach:
    verdict_span = tds_in_row[3].find("span", class_=re.compile(r"^verdict-"))
    if verdict_span:
        verdict_text = verdict_span.get_text()
    else:
        # fallback if the span isn't directly in that cell for some reason
        # or if we want to search the entire document
        fallback_span = soup.find("span", class_=re.compile(r"^verdict-"))
        verdict_text = (
            fallback_span.get_text(strip=True) if fallback_span else "Unknown verdict"
        )

    return problem_identifier, verdict_text, language_text


def organize_submissions():
    for filename in os.listdir(CF_SUBMISSIONS_DIR):
        if not filename.endswith(".mhtml"):
            continue
        with open(os.path.join(CF_SUBMISSIONS_DIR, filename), "r") as infile:
            html_content = infile.read()

        submission_id = filename.split("#")[1].split(" -")[0]
        problem_identifier_with_revision, verdict_text, language_text = (
            parse_submission_file(html_content)
        )

        print(submission_id)
        print(problem_identifier_with_revision, verdict_text, language_text)
        problem_id = problem_identifier_with_revision.split()[0]
        print(problem_id)

        if "C++" not in language_text:
            continue

        short_verdict = None
        if verdict_text.startswith("Wrong answer on"):
            short_verdict = "WA"
        elif verdict_text.startswith("Accepted"):
            short_verdict = "AC"
        elif verdict_text.startswith("Runtime error on"):
            short_verdict = "RE"
        elif verdict_text.startswith("Time limit exceeded on"):
            short_verdict = "TLE"
        elif verdict_text.startswith("Memory limit exceeded on"):
            short_verdict = "MLE"
        elif verdict_text.startswith("Compilation error"):
            short_verdict = "CE"
        else:
            raise ValueError(f"Unknown verdict: {verdict_text}")

        os.makedirs(os.path.join(CF_SUBMISSIONS_DIR, problem_id), exist_ok=True)
        # copy the file to the problem_id directory
        shutil.copy(
            os.path.join(CF_SUBMISSIONS_DIR, filename),
            os.path.join(CF_SUBMISSIONS_DIR, problem_id, filename),
        )
        shutil.copy(
            os.path.join(CF_SUBMISSIONS_DIR, submission_id),
            os.path.join(
                CF_SUBMISSIONS_DIR, problem_id, f"{submission_id}_{short_verdict}.cpp"
            ),
        )


def parse_problem_html(html_content: str) -> str:
    """
    Takes the raw HTML of a Codeforces problem page and returns
    a plain-text description containing:
      - TITLE
      - TIME LIMIT
      - MEMORY LIMIT
      - STATEMENT
      - INPUT
      - OUTPUT
      - EXAMPLES (list of {input, output})
    preserving LaTeX math in `$...$`.
    """

    soup = BeautifulSoup(html_content, "html.parser")

    # --- 1) TITLE ---
    # Usually in <div class="problem-statement"> <div class="header"> <div class="title">Some Title</div>
    title_elem = soup.select_one(".problem-statement .header .title")
    title = title_elem.get_text(strip=True) if title_elem else "Unknown Title"

    # --- 2) TIME LIMIT ---
    # Typically in <div class="problem-statement"> <div class="header"> <div class="time-limit">3 seconds</div>
    # The text is something like:  <div class="property-title">time limit per test</div>3 seconds
    time_elem = soup.select_one(".problem-statement .header .time-limit")
    time_text = (
        time_elem.get_text(" ", strip=True) if time_elem else "Unknown time limit"
    )
    # e.g. "time limit per test 3 seconds" => we usually only want "3 seconds"
    # We can do a quick extraction:
    match = re.search(r"(\d+\s*\w+)", time_text)
    time_limit = match.group(1) if match else "Unknown"

    # --- 3) MEMORY LIMIT ---
    # Similar to above: <div class="memory-limit">256 megabytes</div>
    mem_elem = soup.select_one(".problem-statement .header .memory-limit")
    mem_text = (
        mem_elem.get_text(" ", strip=True) if mem_elem else "Unknown memory limit"
    )
    match = re.search(r"(\d+\s*\w+)", mem_text)
    memory_limit = match.group(1) if match else "Unknown"

    # --- 4) PROBLEM STATEMENT TEXT ---
    # The entire .problem-statement content is typically broken into:
    #   .header, (the "Input/Output specification" blocks), the main text in <div> or <p> elements
    # We'll gather anything inside .problem-statement except the .header, .input-specification,
    # .output-specification, .sample-tests
    statement_div = soup.select_one(".problem-statement")
    if not statement_div:
        return "Problem statement not found in HTML."

    header_text = []
    input_text = []
    output_text = []
    note_text = []
    examples_list = []
    for selector in [
        ".header",
        ".input-specification",
        ".output-specification",
        ".sample-tests",
        ".note",
    ]:
        for sec in statement_div.select(selector):
            if selector == ".header":
                header_text.append(sec.get_text("\n"))
            elif selector == ".input-specification":
                # ignore section-title here
                for child in sec.children:
                    # check if child is a div with class section-title
                    if child.name == "div" and "section-title" in child.get(
                        "class", []
                    ):
                        continue
                    elif child.name == "p":
                        input_text.append(child.get_text("\n"))
                    else:
                        raise ValueError(f"Unknown child: {child.name}")
            elif selector == ".output-specification":
                for child in sec.children:
                    if child.name == "div" and "section-title" in child.get(
                        "class", []
                    ):
                        continue
                    elif child.name == "p":
                        output_text.append(child.get_text("\n"))
                    else:
                        raise ValueError(f"Unknown child: {child.name}")
            elif selector == ".sample-tests":
                examples_list = []
                for s_test in sec.select(".sample-test"):
                    in_div = s_test.select_one(".input > pre")
                    out_div = s_test.select_one(".output > pre")
                    in_text = []
                    out_text = []
                    for child in in_div.children:
                        if child.name == None and child.get_text(strip=True) == "":
                            continue
                        if child.name == None or (
                            child.name == "div"
                            and "test-example-line" in child.get("class", [])
                        ):
                            in_text.append(child.get_text("", strip=True))
                        else:
                            raise ValueError(f"Unknown child: {child.name}")
                    in_text = "\n".join(in_text)
                    for child in out_div.children:
                        if child.name == None and child.get_text(strip=True) == "":
                            continue
                        if child.name == None or (
                            child.name == "div"
                            and "test-example-line" in child.get("class", [])
                        ):
                            out_text.append(child.get_text("", strip=True))
                        else:
                            raise ValueError(f"Unknown child: {child.name}")
                    out_text = "\n".join(out_text)

                    examples_list.append({"input": in_text, "output": out_text})
            elif selector == ".note":
                for child in sec.children:
                    if child.name == "div" and "section-title" in child.get(
                        "class", []
                    ):
                        continue
                    elif child.name == "p" or child.name == None:
                        note_text.append(child.get_text("\n"))
                    elif child.name == "ol" or child.name == "ul":
                        note_text.append(child.get_text("\n"))
                    elif child.name == "center":
                        note_text.append(child.get_text("\n"))
                    else:
                        raise ValueError(f"Unknown child {child.name}: {child}")
            sec.decompose()

    statement_text = []
    assert len(list(statement_div.children)) == 1
    statement_div = next(statement_div.children)
    for child in statement_div.children:
        if child.name == "p" or child.name == None:
            statement_text.append(child.get_text("\n"))
        elif child.name == "ol" or child.name == "ul":
            statement_text.append(child.get_text("\n"))
        elif child.name == "div" and "epigraph" in child.get("class", []):
            continue
        elif child.name == "div" and "statement-footnote" in child.get("class", []):
            footnote_text = child.get_text("\n")
            # if footnote_text.startswith(r"$$$^{\text{"):
            #     footnote_symbol_prefix_length = len(r"$$$^{\text{") + 1 + len("}}$$$")
            #     footnote_text = footnote_text[footnote_symbol_prefix_length:]
            statement_text.append(r"\footnote{" + footnote_text + "}")
        elif child.name == "center":
            only_img_found = False
            for grandchild in child.children:
                if grandchild.name == "img":
                    only_img_found = True
                elif grandchild.name:
                    only_img_found = False
                    break
            if only_img_found:
                continue
            raise ValueError(f"Unknown child {child.name}: {child}")
        else:
            raise ValueError(f"Unknown child {child.name}: {child}")
    statement_text = "\n\n".join(statement_text)

    examples_json = json.dumps(examples_list, indent=2)
    input_text = "\n\n".join(input_text)
    output_text = "\n\n".join(output_text)
    note_text = "\n\n".join(note_text)

    result_lines = []
    result_lines.append(f"TITLE: {title}")
    result_lines.append(f"TIME LIMIT: {time_limit}")
    result_lines.append(f"MEMORY LIMIT: {memory_limit}")
    result_lines.append("")
    result_lines.append("--- STATEMENT ---")
    result_lines.append(statement_text)
    result_lines.append("")
    if input_text.strip():
        result_lines.append("--- INPUT ---")
        result_lines.append(input_text)
    if output_text.strip():
        result_lines.append("")
        result_lines.append("--- OUTPUT ---")
        result_lines.append(output_text)
    if examples_list:
        result_lines.append("")
        result_lines.append("--- EXAMPLES ---")
        result_lines.append(examples_json)
    if note_text.strip():
        result_lines.append("")
        result_lines.append("--- NOTE ---")
        result_lines.append(note_text)

    # Join everything with a newline.
    return "\n".join(result_lines), examples_list


def download_problem_htmls(problem_ids):
    for problem_id in problem_ids:
        contest_id = re.search(r"(\d+)", problem_id).group(1)
        problem_index = problem_id[len(contest_id) :]
        problem_html_filepath = f"{CF_SUBMISSIONS_DIR}/{problem_id}/problem.html"
        if os.path.exists(problem_html_filepath):
            logging.info("Skipping %s because it already exists", problem_id)
            continue

        # download the problem page from https://codeforces.com/contest/2006/problem/C
        url = f"https://codeforces.com/contest/{contest_id}/problem/{problem_index}"
        print(url)

        pyperclip.waitForNewPaste()
        text = pyperclip.paste()

        with open(problem_html_filepath, "w") as outfile:
            outfile.write(text)


def create_problem_statement_text_files(problem_ids):
    for problem_id in problem_ids:
        print(problem_id)
        problem_dir_path = f"{CF_SUBMISSIONS_DIR}/{problem_id}"
        problem_html_filepath = f"{problem_dir_path}/problem.html"
        problem_statement_filepath = f"{problem_dir_path}/statement.txt"
        if not os.path.exists(problem_html_filepath):
            logging.info("Skipping %s because it doesn't exist", problem_id)
            continue
        html_content = open(problem_html_filepath).read()
        problem_statement_str, examples_list = parse_problem_html(html_content)
        print(examples_list)

        for example_num, example in enumerate(examples_list):
            example_input_filepath = f"{problem_dir_path}/{example_num + 1}.in"
            example_output_filepath = f"{problem_dir_path}/{example_num + 1}.out"
            with open(example_input_filepath, "w") as outfile:
                outfile.write(example["input"])
            with open(example_output_filepath, "w") as outfile:
                outfile.write(example["output"])
        with open(problem_statement_filepath, "w") as outfile:
            outfile.write(problem_statement_str)


def list_recent_contests(num, div=1, from_date=None, up_to_date=None):
    contest_list = get_contest_list()
    filtered_contest_list = [
        contest
        for contest in contest_list
        if contest["phase"] == "FINISHED"
        and (f"Div. {div}" in contest["name"])
        and "Unrated" not in contest["name"]
    ]

    filtered_contest_list.sort(key=lambda x: x["startTimeSeconds"], reverse=True)
    if up_to_date:
        filtered_contest_list = [
            contest
            for contest in filtered_contest_list
            if datetime.datetime.fromtimestamp(contest["startTimeSeconds"]) <= up_to_date
        ]
    if from_date:
        filtered_contest_list = [
            contest
            for contest in filtered_contest_list
            if datetime.datetime.fromtimestamp(contest["startTimeSeconds"]) >= from_date
        ]
    filtered_contest_list = filtered_contest_list[:num]

    return [contest["id"] for contest in filtered_contest_list]


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    # organize_submissions()
    # download_problem_htmls(problem_ids)
    # create_problem_statement_text_files(problem_ids)
    # write_down_submissions(contest_ids)
    list_recent_contests()


if __name__ == "__main__":
    main()
