import ast
import getpass
import json
import math
from bisect import bisect_right
import functools
from collections import defaultdict
from decimal import Decimal
import os
import yaml
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from analyze_agent_trajectories import get_agent_costs_per_div
from report_utils import generate_latex_code_for_figures
from analyze_agent_trajectories import (
    collect_cost_dicts_for_kshot_agents,
    calc_queries_until_first_succ_for_kshot,
    calc_cost_until_first_succ_for_kshot,
    get_exit_status_dict,
)


def _extract_brace_block(text: str, anchor: str) -> str | None:
    """
    Find *anchor* in *text* and return the first {...} block that follows.
    Returns None if the anchor or a balanced brace block is not found.
    """
    anchor_pos = text.find(anchor)
    if anchor_pos == -1:
        return None

    # search from first '{' after the anchor
    start = text.find('{', anchor_pos)
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:            # balanced root-level brace closed
                return text[start : i + 1]
    return None                       # unbalanced / not found


def make_latex_table(successes_dict : dict[str, dict[str, bool]], model : str, div : int, only_rows : bool = False) -> str:
    """
    Build a LaTeX table summarising `successes_dict`.

    Parameters
    ----------
    successes_dict : dict
        Mapping like {'2061A': True, '2061B': False, …}.
    only_row : bool, optional
        If True, return the compact table **without** the “Sum” column
        or caption/label (mimics DataFrame.to_latex(index=True)).
        If False (default) include the totals column plus caption & label.

    Returns
    -------
    str
        LaTeX code ready to drop into your document.
    """

    contests = None
    num_of_problems = None
    res_lines = []
    for row_name in successes_dict:
        agg = dict()
        if num_of_problems is None:
            num_of_problems = len(successes_dict[row_name])
        else:
            assert num_of_problems == len(successes_dict[row_name])

        for key, is_solved in successes_dict[row_name].items():
            m = re.match(r"(\d{4})", key)
            if not m:
                raise ValueError(f"Could not parse key {key}")
            contest = m.group(1)
            if contest not in agg:
                agg[contest] = 0
            if is_solved:
                agg[contest] += 1

        if contests is None:
            contests = sorted(agg)
        else:
            assert contests == sorted(agg)

        vals = [agg[c] for c in contests]
        vals.append(f"{sum(vals)} / {num_of_problems}")
        res_lines.append(row_name + " & " + " & ".join(map(str, vals)) + r" \\")

    col_fmt  = "l" + "r"*len(contests) + ("|r")
    header   = ["{}"] + contests + (["Sum"])

    lines = []
    if not only_rows:
        caption = rf"\textbf{{{model}}}, Division {div}"
        label   = f"tab:{model}-div{div}"
        lines.append(r"\begin{table}[h!]")
        lines.append(fr"\caption{{{caption}}}")
        lines.append(fr"\label{{{label}}}")

        lines.append(rf"\begin{{tabular}}{{{col_fmt}}}")
        lines.append(r"\toprule")
        lines.append(" & ".join(header) + r" \\")
        lines.append(r"\midrule")

    lines.extend(res_lines)

    if not only_rows:
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

    return "\n".join(lines)


def extract_attempt_costs(text: str) -> dict[str, float]:
    cost_re = re.compile(
        r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*?\bModel response for (?P<prob>[A-Za-z0-9._-]+): cost=\$(?P<cost>\d+(?:\.\d+)?)\b",
        re.IGNORECASE,
    )

    cost_per_problem = {}

    for line in text.splitlines():
        m_cost = cost_re.search(line)
        if m_cost:
            prob = m_cost.group("prob")
            cost = Decimal(m_cost.group("cost"))
            cost_per_problem[prob] = cost
            continue
    return cost_per_problem


def parse_oneshot_runs(model : str, div : int, kshot : int, must_have_kshot : bool = True):
    logs_dir_candidates = [
        Path("oneshot_baselines") / model / f"div{div}" / "logs",
        Path("oneshot_baselines") / (model + "_t-0.50") / f"div{div}" / "logs",
        Path("oneshot_baselines") / (model + "_t-1.00") / f"div{div}" / "logs",
    ]

    for logs_dir in logs_dir_candidates:
        if logs_dir.is_dir():
            break
    else:
        print(f"% No logs directory found for {model} in div {div}")
        return {}, {}

    merged_results = {}
    shots_processed = 0
    cost_until_success = defaultdict(Decimal)
    queries_until_success = defaultdict(int)
    successes = set()
    # print(f"% Logs dir: {logs_dir}")
    for filename in sorted(os.listdir(logs_dir)):
        path = Path(logs_dir) / filename
        if not path.is_file() or path.suffix != ".log":
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")

        block = _extract_brace_block(text, "Overall status dict:")
        if block is None:
            continue  # this file doesn’t contain the section we need

        try:
            status_dict = ast.literal_eval(block)
        except Exception as exc:       # invalid literal – skip but warn
            raise ValueError(f"Could not parse dict in {path}: {exc}")

        cost_per_problem = extract_attempt_costs(text)

        # print(f"%\n logs_dir = {logs_dir}")
        # print(f"\n=== {path.relative_to(logs_dir)} ===")
        # print(cost_per_problem)

        for key in status_dict:
            if key in successes:
                continue
            if status_dict[key]["success"]:
                assert status_dict[key]["success"] == 1
                successes.add(key)
                cost_until_success[key] += cost_per_problem[key]
                queries_until_success[key] += 1
            elif status_dict[key]["failure"] or status_dict[key]["error"]:
                assert status_dict[key]["failure"] == 1 or status_dict[key]["error"] == 1
                cost_until_success[key] += cost_per_problem[key]
                queries_until_success[key] += 1

        for key in status_dict:
            if key not in merged_results:
                merged_results[key] = {}
            for status in status_dict[key]:
                merged_results[key][status] = merged_results.get(key, {}).get(status, 0) + status_dict[key][status]

        shots_processed += 1
        if shots_processed == kshot:
            break

    if shots_processed < kshot and must_have_kshot:
        raise ValueError(f"Only {shots_processed} shots processed, expected {kshot}")

    successes = {}
    for key, counts in merged_results.items():
        if not model.startswith("o3"):
            assert counts["error"] == 0, f"Error count is not 0 for model={model}, key={key}, div={div}, kshot={kshot}"
        successes[key] = True if counts.get("success", 0) > 0 else False

    cost_until_success = {key: cost_until_success[key] for key in successes if successes[key]}
    queries_until_success = {key: queries_until_success[key] for key in successes if successes[key]}

    return merged_results, cost_until_success, queries_until_success


def parse_agent_runs(agent_traj_dirname : str):
    agent_traj_dir = Path(".") / "agent_trajectories" / agent_traj_dirname
    exit_status_dict = get_exit_status_dict(agent_traj_dir)

    results_dict = {}
    for key in exit_status_dict.get('ok', []):
        results_dict[key] = True
    for exit_status in exit_status_dict:
        if exit_status == 'ok':
            continue
        for key in exit_status_dict[exit_status]:
            results_dict[key] = False

    return results_dict


def plot_cumulative_solved(
        list_of_costs_per_div_dicts: list[dict[int, list[float | int]]],
        list_of_labels: list[str],
        out_pdf: Path,
        xlim: tuple[float, float] = (0.0, 2.0),
        xlabel: str = "Cost (USD)",
        ylabel: str = "Solved problems ≤ cost",
        xmajor_step: float = 0.5,
        colors: list[str] | None = None,
    ):

    all_divs = set()
    for ind_dict in list_of_costs_per_div_dicts:
        all_divs.update(ind_dict.keys())
    ncols = len(all_divs)

    fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(8, 3))
    for ind, div in enumerate(sorted(all_divs)):
        ax = axs[ind]

        for series_idx, (ind_dict, label) in enumerate(zip(list_of_costs_per_div_dicts, list_of_labels)):
            if div not in ind_dict:
                continue
            costs = ind_dict[div]
            costs_are_ints = (
                len(costs) > 0
                and all(isinstance(cost, int) and not isinstance(cost, bool) for cost in costs)
            )
            xs_start = 0 if costs_are_ints else 0.0
            xs = [xs_start] + costs + [xlim[1]]
            ys = [0] + list(range(1, len(costs) + 1)) + [len(costs)]
            kwargs = {}
            if colors is not None:
                kwargs["color"] = colors[series_idx]
            ax.step(xs, ys, where="post", label=label, **kwargs)

        ax.set_xlabel(xlabel)
        ax.set_title(f"Div {div}", fontsize=9)
        ax.set_xlim(xlim[0], xlim[1])

        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xmajor_step))
        ax.grid(True, which="both", axis="x", alpha=0.35, linewidth=0.6)
        ax.grid(True, which="both", axis="y", alpha=0.35, linewidth=0.6)

    fig.supylabel(ylabel)

    fig.subplots_adjust(left=0.1, top=0.92, wspace=0.3)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_pdf


def get_oneshot_costs_per_div(k: int, model_name: str, divs: list[int], must_have_kshot : bool = True):
    sorted_costs_per_div = {}
    sorted_queries_per_div = {}
    per_problem_costs = {}
    per_problem_queries = {}
    for div in divs:
        merged_results, cost_until_success, queries_until_success = parse_oneshot_runs(model_name, div, k, must_have_kshot)
        costs = sorted(cost_until_success.values())
        queries = sorted(queries_until_success.values())
        sorted_costs_per_div[div] = costs
        sorted_queries_per_div[div] = queries
        per_problem_costs[div] = cost_until_success
        per_problem_queries[div] = queries_until_success
        for key in merged_results:
            if key not in per_problem_costs[div]:
                per_problem_costs[div][key] = None
            if key not in per_problem_queries[div]:
                per_problem_queries[div][key] = None
    return sorted_costs_per_div, sorted_queries_per_div, per_problem_costs, per_problem_queries


_MODEL_NAMES = ["o3", "claude-sonnet-4-20250514", "gemini/gemini-2.5-pro"]
_PRETTY_MODEL_NAMES = {
    "o3": "OpenAI O3",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "gemini/gemini-2.5-pro": "Gemini 2.5 Pro",
}
_SERIES_COLORS = ["C0", "C1", "C2"]
_COLOR_LEGEND = (
    r"{\color{mplblue}agent}, "
    r"{\color{mplorange}$k$-shot}, "
    r"{\color{mplgreen}budget-partitioned agent ($\nicefrac{1}{3}\times 3$)}"
)


@functools.cache
def _collect_method_data():
    """Collect (agent, oneshot, kshot) × (costs, queries) per div for each model."""
    result = {}
    for model_name in _MODEL_NAMES:
        oneshot_costs, oneshot_queries, _, _ = get_oneshot_costs_per_div(
            118, model_name, [1, 2, 3], must_have_kshot=False)
        agent_costs, agent_queries, _, _ = get_agent_costs_per_div(model_name, [1, 2, 3])
        kshot_costs, kshot_queries = {}, {}
        for div in [1, 2, 3]:
            sc, fc, sq, fq = collect_cost_dicts_for_kshot_agents(model_name, div, Decimal("0.67"))
            kshot_costs[div] = sorted(calc_cost_until_first_succ_for_kshot(sc, fc).values())
            kshot_queries[div] = sorted(calc_queries_until_first_succ_for_kshot(sq, fq).values())
        result[model_name] = {
            "costs": (agent_costs, oneshot_costs, kshot_costs),
            "queries": (agent_queries, oneshot_queries, kshot_queries),
        }
    return result


def _fig_cum_solved(
    metric: str,
    xlim, xlabel, ylabel, xmajor_step,
    pic_prefix: str,
    caption_metric: str,
    fig_label: str,
):
    data = _collect_method_data()
    pictures = []
    subcaptions = []
    for model_name in _MODEL_NAMES:
        agent_pd, oneshot_pd, kshot_pd = data[model_name][metric]
        basename = f"{pic_prefix}_{model_name.split('/')[-1]}"
        pic = plot_cumulative_solved(
            [agent_pd, oneshot_pd, kshot_pd],
            ["agent", "k-shot", "agent-(1/3)×3"],
            out_pdf=Path("..") / "overleaf" / "pics" / f"{basename}.pdf",
            xlim=xlim, xlabel=xlabel, ylabel=ylabel,
            xmajor_step=xmajor_step, colors=_SERIES_COLORS,
        ).name
        pictures.append(pic)
        subcaptions.append(_PRETTY_MODEL_NAMES[model_name])

    print(generate_latex_code_for_figures(
        pictures, [1, 1, 1],
        f"Cumulative solved problems versus {caption_metric}: {_COLOR_LEGEND}.",
        fig_label, subcaptions=subcaptions, subfigure_width=0.9, use_tabular=True,
    ))


def fig_cum_solved_vs_cost():
    _fig_cum_solved(
        "costs", xlim=(0.0, 2.0), xlabel="Cost (USD)",
        ylabel="Solved problems ≤ cost", xmajor_step=0.5,
        pic_prefix="solved_vs_cost", caption_metric="inference cost",
        fig_label="cum-solved-vs-cost",
    )


def fig_cum_solved_vs_queries():
    _fig_cum_solved(
        "queries", xlim=(0, 50), xlabel="Number of queries",
        ylabel="Solved problems ≤ queries", xmajor_step=10,
        pic_prefix="solved_vs_queries", caption_metric="number of queries",
        fig_label="cum-solved-vs-queries",
    )


def table_success_results(div):
    model_names = [
        "o3",
        "claude-sonnet-4-20250514",
        "gemini/gemini-2.5-pro",
    ]

    short_model_names = {
        "o3": "o3",
        "claude-sonnet-4-20250514": "sonnet",
        "gemini/gemini-2.5-pro": "gemini",
    }

    per_contest_per_model_per_problem_costs = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    per_contest_per_model_per_problem_queries = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    agent_per_contest_per_model_per_problem_costs = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    agent_per_contest_per_model_per_problem_queries = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for model_name in model_names:
        oneshot_costs_per_div, oneshot_queries_per_div, per_problem_costs, per_problem_queries = get_oneshot_costs_per_div(118, model_name, [div], must_have_kshot=False)
        agent_costs_per_div, agent_queries_per_div, agent_per_problem_costs, agent_per_problem_queries = get_agent_costs_per_div(model_name, [div])

        for key in per_problem_costs[div]:
            contest_id = key[:4]
            problem_id = key[4:]
            per_contest_per_model_per_problem_costs[contest_id][model_name][problem_id] = per_problem_costs[div][key]
        for key in per_problem_queries[div]:
            contest_id = key[:4]
            problem_id = key[4:]
            per_contest_per_model_per_problem_queries[contest_id][model_name][problem_id] = per_problem_queries[div][key]
        for key in agent_per_problem_costs[div]:
            contest_id = key[:4]
            problem_id = key[4:]
            agent_per_contest_per_model_per_problem_costs[contest_id][model_name][problem_id] = agent_per_problem_costs[div][key]
        for key in agent_per_problem_queries[div]:
            contest_id = key[:4]
            problem_id = key[4:]
            agent_per_contest_per_model_per_problem_queries[contest_id][model_name][problem_id] = agent_per_problem_queries[div][key]

    # For each contest in this division, create a table
    labels_list = []
    for contest_id in sorted(per_contest_per_model_per_problem_costs.keys()):
        # Get all problem IDs for this contest (across all models)
        all_problem_ids = set()
        for model_name in model_names:
            if contest_id in per_contest_per_model_per_problem_costs:
                all_problem_ids.update(per_contest_per_model_per_problem_costs[contest_id][model_name].keys())
        problem_ids = sorted(all_problem_ids)

        if not problem_ids:
            continue

        # Subtable for this contest
        col_spec = "l" + "c"*len(problem_ids)
        print(r"\begin{table}[h]")
        print(r"\centering")
        print(rf"\begin{{tabular}}{{{col_spec}}}")
        print(r"\toprule")
        print(" & " + " & ".join(problem_ids) + r" \\")
        print(r"\midrule")

        # For each model, add cost and query rows
        for model_idx, model_name in enumerate(model_names):
            if contest_id not in per_contest_per_model_per_problem_costs:
                continue

            short_model_name = short_model_names[model_name]
            # Cost row
            cost_row = []
            for pid in problem_ids:
                cost_val = per_contest_per_model_per_problem_costs[contest_id][model_name].get(pid)
                cost_row.append("" if cost_val is None else f"{cost_val:.2f}")
            print(rf"{short_model_name} costs & " + " & ".join(cost_row) + r" \\")

            # Query row
            query_row = []
            for pid in problem_ids:
                query_val = per_contest_per_model_per_problem_queries[contest_id][model_name].get(pid)
                query_row.append("" if query_val is None else str(query_val))
            print(rf"{short_model_name} queries & " + " & ".join(query_row) + r" \\")

            # Agent cost row
            agent_cost_row = []
            for pid in problem_ids:
                agent_cost_val = agent_per_contest_per_model_per_problem_costs[contest_id][model_name].get(pid)
                agent_cost_row.append("" if agent_cost_val is None else f"{agent_cost_val:.2f}")
            print(rf"{short_model_name} (ag) costs & " + " & ".join(agent_cost_row) + r" \\")

            # Agent query row
            agent_query_row = []
            for pid in problem_ids:
                agent_query_val = agent_per_contest_per_model_per_problem_queries[contest_id][model_name].get(pid)
                agent_query_row.append("" if agent_query_val is None else str(agent_query_val))
            print(rf"{short_model_name} (ag) queries & " + " & ".join(agent_query_row) + r" \\")

            # Add midrule between models (but not after the last one)
            if model_idx < len(model_names) - 1:
                print(r"\midrule")
            else:
                print(r"\bottomrule")

        print(r"\end{tabular}")
        print(rf"\caption{{Contest {contest_id} (Div. {div}); \textit{{(ag)}} = agent}}")
        label = f"tab:contest_{contest_id}_div{div}"
        print(rf"\label{{{label}}}")
        print(r"\end{table}")
        print()
        labels_list.append(label)

    labels_str = ", ".join(labels_list)
    print(f"% {labels_str}")


def _plot_averaged_cumulative(
        method_curves: list[list[tuple[list, int]]],
        colors: list[str],
        labels: list[str],
        xlim: tuple[float, float],
        xlabel: str,
        ylabel: str,
        xmajor_step: float,
        out_pdf: Path,
    ):
    """Plot a single figure with one averaged curve per method.

    Parameters
    ----------
    method_curves : list (one per method) of lists of (sorted_values, n_total)
        Each inner list has one entry per (model, div) pair (9 entries).
        *sorted_values* are the ascending costs/queries of solved problems;
        *n_total* is the total number of problems in that pair (for
        normalisation).
    """
    x_grid = np.linspace(float(xlim[0]), float(xlim[1]), 2000)

    # Match one subplot of `plot_cumulative_solved` (figsize=(8, 3) with 3
    # subplots, rendered in LaTeX at subfigure_width=0.9).
    fig, ax = plt.subplots(figsize=(8 / 3, 3))

    for curves, color, label in zip(method_curves, colors, labels):
        ys_all = []
        for sorted_vals, n_total in curves:
            sorted_floats = [float(v) for v in sorted_vals]
            ys = np.array([bisect_right(sorted_floats, x) / n_total
                           for x in x_grid])
            ys_all.append(ys)
        avg_ys = np.mean(ys_all, axis=0)
        ax.plot(x_grid, avg_ys, color=color, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.yaxis.set_ticks([])
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xmajor_step))
    ax.grid(True, which="both", axis="x", alpha=0.35, linewidth=0.6)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_pdf


def fig_avg_cum_solved_vs_cost():
    """Produce a single averaged figure across all 9 (model × div) pairs."""
    model_names = ["o3", "claude-sonnet-4-20250514", "gemini/gemini-2.5-pro"]
    series_colors = ["C0", "C1", "C2"]
    series_labels = [
        "SWE-agent",
        "$k$-shot",
        "budget-partitioned SWE-agent",
    ]

    # method_cost_curves[method_idx] = [(sorted_costs, n_total), ...] (9 entries)
    method_cost_curves: list[list[tuple[list, int]]] = [[], [], []]
    method_query_curves: list[list[tuple[list, int]]] = [[], [], []]

    for model_name in model_names:
        _, _, agent_pp_costs, agent_pp_queries = get_agent_costs_per_div(
            model_name, [1, 2, 3])
        _, _, oneshot_pp_costs, oneshot_pp_queries = get_oneshot_costs_per_div(
            118, model_name, [1, 2, 3], must_have_kshot=False)

        for div in [1, 2, 3]:
            n_total = len(agent_pp_costs[div])

            agent_costs = sorted(
                v for v in agent_pp_costs[div].values() if v is not None)
            oneshot_costs = sorted(
                v for v in oneshot_pp_costs[div].values() if v is not None)

            agent_queries = sorted(
                v for v in agent_pp_queries[div].values() if v is not None)
            oneshot_queries = sorted(
                v for v in oneshot_pp_queries[div].values() if v is not None)

            sc, fc, sq, fq = collect_cost_dicts_for_kshot_agents(
                model_name, div, Decimal("0.67"))
            kshot_agent_costs = sorted(
                calc_cost_until_first_succ_for_kshot(sc, fc).values())
            kshot_agent_queries = sorted(
                calc_queries_until_first_succ_for_kshot(sq, fq).values())

            method_cost_curves[0].append((agent_costs, n_total))
            method_cost_curves[1].append((oneshot_costs, n_total))
            method_cost_curves[2].append((kshot_agent_costs, n_total))

            method_query_curves[0].append((agent_queries, n_total))
            method_query_curves[1].append((oneshot_queries, n_total))
            method_query_curves[2].append((kshot_agent_queries, n_total))

    color_legend = (
        r"{\color{mplblue}SWE-agent}, "
        r"{\color{mplorange}$k$-shot}, "
        r"{\color{mplgreen}budget-partitioned SWE-agent ($\nicefrac{1}{3}\times 3$)}"
    )

    cost_pdf = _plot_averaged_cumulative(
        method_cost_curves,
        series_colors,
        series_labels,
        xlim=(0.0, 2.0),
        xlabel="Cost (USD)",
        ylabel="Solved problems \u2264 cost",
        xmajor_step=0.5,
        out_pdf=Path("..") / "overleaf" / "pics" / "avg_solved_vs_cost.pdf",
    )

    query_pdf = _plot_averaged_cumulative(
        method_query_curves,
        series_colors,
        series_labels,
        xlim=(0, 50),
        xlabel="Number of queries",
        ylabel="Solved problems \u2264 queries",
        xmajor_step=10,
        out_pdf=Path("..") / "overleaf" / "pics" / "avg_solved_vs_queries.pdf",
    )

    print(generate_latex_code_for_figures(
        [cost_pdf.name, query_pdf.name],
        [2],
        f"Averaged cumulative solved problems (across all models and divisions) versus inference cost (left) and number of queries (right): {color_legend}.",
        "avg-cum-solved",
        use_subfigure=False,
        # Match the per-division figures: each panel occupies 0.9/3 = 0.30
        # of \textwidth, so a row can host three such panels even though
        # the averaged figure only has two.
        subfigure_width=0.30,
    ))


# ---------------------------------------------------------------------------
# Clopper-Pearson exact CI (pure-Python, no scipy)
# ---------------------------------------------------------------------------

def _beta_cf(x, a, b, max_iter=300, tol=1e-14):
    tiny = 1e-30
    f = 1.0
    C = 1.0
    D = 1.0 - (a + b) * x / (a + 1)
    if abs(D) < tiny:
        D = tiny
    D = 1.0 / D
    f = D
    for m in range(1, max_iter + 1):
        num = m * (b - m) * x / ((a + 2*m - 1) * (a + 2*m))
        D = 1.0 + num * D
        if abs(D) < tiny: D = tiny
        C = 1.0 + num / C
        if abs(C) < tiny: C = tiny
        D = 1.0 / D
        f *= D * C
        num = -(a + m) * (a + b + m) * x / ((a + 2*m) * (a + 2*m + 1))
        D = 1.0 + num * D
        if abs(D) < tiny: D = tiny
        C = 1.0 + num / C
        if abs(C) < tiny: C = tiny
        D = 1.0 / D
        delta = D * C
        f *= delta
        if abs(delta - 1.0) < tol:
            break
    return f


def _beta_cdf(x, a, b):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _beta_cdf(1 - x, b, a)
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
    return front * _beta_cf(x, a, b)


def _beta_ppf(p, a, b, tol=1e-10):
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _beta_cdf(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def _clopper_pearson(k, n, alpha=0.05):
    lo = _beta_ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = _beta_ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


# ---------------------------------------------------------------------------
# Probability-estimation data collection & plots
# ---------------------------------------------------------------------------

SWE_AGENT_PATH = Path("..") / "SWE-agent"
SWE_AGENT_USER = os.environ.get("SWE_AGENT_USER", getpass.getuser())


def _collect_agent_prob_data(
    model: str, config_name: str, temperature: float,
    div: int, instance_id: str, cost_limits: list[float],
    min_runs: int = 10,
):
    """For each cost limit, count successes/total and mean actual cost."""
    results = []
    for cost_limit in cost_limits:
        cost_str = f"{cost_limit:.2f}"
        base = (
            SWE_AGENT_PATH / "trajectories" / SWE_AGENT_USER
            / f"{config_name}__{model}__t-{temperature:.2f}__p-1.00__c-{cost_str}___div{div}"
        )
        successes, total = 0, 0
        actual_costs = []
        api_calls_list = []
        i = 0
        while True:
            run_dir = base.with_name(base.name + f"_{i}")
            if not run_dir.is_dir():
                break
            traj_file = run_dir / instance_id / f"{instance_id}.traj"
            if traj_file.exists():
                traj = json.load(open(traj_file))
                total += 1
                if traj["info"]["exit_status"] == "ok":
                    successes += 1
                ms = traj.get("info", {}).get("model_stats", {})
                if "instance_cost" in ms:
                    actual_costs.append(ms["instance_cost"])
                if "api_calls" in ms:
                    api_calls_list.append(ms["api_calls"])
            i += 1
        if total >= min_runs:
            mean_cost = sum(actual_costs) / len(actual_costs) if actual_costs else cost_limit
            mean_queries = sum(api_calls_list) / len(api_calls_list) if api_calls_list else 0
            results.append({
                "cost_limit": cost_limit,
                "mean_cost": mean_cost,
                "mean_queries": mean_queries,
                "successes": successes,
                "total": total,
            })
    return results


def _collect_oneshot_prob_data(
    model: str, temperature: float, div: int,
    contest_id: str, problem_index: str,
):
    """Count successes/total and mean cost from oneshot result files."""
    results_dir = (
        Path(f"oneshot_baselines/{model}_t-{temperature:.2f}/div{div}") / contest_id
    )
    successes, total, max_cost = 0, 0, 0.0
    for f in results_dir.iterdir():
        if not f.name.startswith(f"{problem_index}_") or f.suffix != ".json":
            continue
        data = json.load(open(f))
        total += 1
        max_cost = max(max_cost, data.get("cost", 0.0))
        if f.name.endswith("_success.json"):
            successes += 1
    return {"cost": max_cost, "successes": successes, "total": total}


def fig_prob_estimation():
    model = "claude-sonnet-4-20250514"
    config_name = "cp_claude"
    temperature = 0.50
    div = 1
    instance_id = "2115A"
    contest_id = "2115"
    problem_index = "A"
    cost_limits = [0.0625, 0.125, 0.1875, 0.25, 0.375, 0.50, 0.75, 1.00, 1.25, 2.25]
    if div == 1:
        cost_limits = cost_limits[1:]

    agent_data = _collect_agent_prob_data(
        model, config_name, temperature, div, instance_id, cost_limits)
    print("agent data collected")
    oneshot_data = _collect_oneshot_prob_data(
        model, temperature, div, contest_id, problem_index)
    print("oneshot data collected")

    # Build arrays for agent
    agent_costs = [d["cost_limit"] for d in agent_data]
    agent_ps = [d["successes"] / d["total"] for d in agent_data]
    agent_cis = [_clopper_pearson(d["successes"], d["total"]) for d in agent_data]
    agent_lo = [ci[0] for ci in agent_cis]
    agent_hi = [ci[1] for ci in agent_cis]
    agent_yerr_lo = [p - lo for p, lo in zip(agent_ps, agent_lo)]
    agent_yerr_hi = [hi - p for p, hi in zip(agent_ps, agent_hi)]

    # Oneshot
    os_c = oneshot_data["cost"]
    os_p = oneshot_data["successes"] / oneshot_data["total"]
    os_lo, os_hi = _clopper_pearson(oneshot_data["successes"], oneshot_data["total"])

    os_k_vals = [c / os_c for c in agent_costs]
    oneshot_kshot     = [1 - (1 - os_p)  ** k for k in os_k_vals]
    oneshot_kshot_lo  = [1 - (1 - os_lo) ** k for k in os_k_vals]
    oneshot_kshot_hi  = [1 - (1 - os_hi) ** k for k in os_k_vals]

    # ---- Plot 1: success probability with error bars ----
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.errorbar(agent_costs, agent_ps,
                yerr=[agent_yerr_lo, agent_yerr_hi],
                fmt="o-", color="C0", capsize=4, label="SWE-agent")
    os_yerr_lo = [p - lo for p, lo in zip(oneshot_kshot, oneshot_kshot_lo)]
    os_yerr_hi = [hi - p for p, hi in zip(oneshot_kshot, oneshot_kshot_hi)]
    ax.errorbar(agent_costs, oneshot_kshot,
                yerr=[os_yerr_lo, os_yerr_hi],
                fmt="s-", color="C1", capsize=4, label=r"$k$-shot $1-(1-\hat{p})^k$")
    ax.set_xlabel("Cost limit per attempt (USD)")
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.35, linewidth=0.6)
    out1 = Path("..") / "overleaf" / "pics" / f"prob_est_{instance_id}_success.pdf"
    out1.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out1, bbox_inches="tight")
    plt.close(fig)

    # ---- Plot 2: -ln(1-p) / c ----
    def _neg_logpc(p, c):
        if p >= 1.0 or c <= 0:
            return float("nan")
        return -math.log(1 - p) / c

    agent_nlogpc = [_neg_logpc(p, c) for p, c in zip(agent_ps, agent_costs)]
    agent_nlogpc_lo = [_neg_logpc(lo, c) for lo, c in zip(agent_lo, agent_costs)]
    agent_nlogpc_hi = [_neg_logpc(hi, c) for hi, c in zip(agent_hi, agent_costs)]
    agent_nlogpc_yerr_lo = [v - lo for v, lo in zip(agent_nlogpc, agent_nlogpc_lo)]
    agent_nlogpc_yerr_hi = [hi - v for v, hi in zip(agent_nlogpc, agent_nlogpc_hi)]

    os_nlogpc = _neg_logpc(os_p, os_c)
    os_nlogpc_lo = _neg_logpc(os_lo, os_c)
    os_nlogpc_hi = _neg_logpc(os_hi, os_c)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.errorbar(agent_costs, agent_nlogpc,
                yerr=[agent_nlogpc_yerr_lo, agent_nlogpc_yerr_hi],
                fmt="o-", color="C0", capsize=4, label="SWE-agent")
    ax.axhline(os_nlogpc, color="C1", linewidth=1.5, label="1-shot")
    ax.axhspan(os_nlogpc_lo, os_nlogpc_hi, color="C1", alpha=0.2)
    ax.set_xlabel("Cost limit per attempt (USD)")
    ax.set_ylabel(r"$-\ln(1 - p)\,/\,c$")
    ax.legend()
    ax.grid(True, alpha=0.35, linewidth=0.6)
    out2 = Path("..") / "overleaf" / "pics" / f"prob_est_{instance_id}_logpc.pdf"
    plt.savefig(out2, bbox_inches="tight")
    plt.close(fig)

    # ---- Plot 3: ln(1-p) with power-law fits ----
    def _log1mp(p):
        if p >= 1.0:
            return float("nan")
        return math.log(1 - p)

    agent_log1mp    = [_log1mp(p) for p in agent_ps]
    agent_log1mp_lo = [_log1mp(hi) for hi in agent_hi]  # higher p → more negative
    agent_log1mp_hi = [_log1mp(lo) for lo in agent_lo]
    agent_log1mp_yerr_lo = [v - lo for v, lo in zip(agent_log1mp, agent_log1mp_lo)]
    agent_log1mp_yerr_hi = [hi - v for v, hi in zip(agent_log1mp, agent_log1mp_hi)]

    os_log1mp    = [_log1mp(p) for p in oneshot_kshot]
    os_log1mp_lo = [_log1mp(hi) for hi in oneshot_kshot_hi]
    os_log1mp_hi = [_log1mp(lo) for lo in oneshot_kshot_lo]
    os_log1mp_yerr_lo = [v - lo for v, lo in zip(os_log1mp, os_log1mp_lo)]
    os_log1mp_yerr_hi = [hi - v for v, hi in zip(os_log1mp, os_log1mp_hi)]

    # Linear fit through origin: ln(1-p) ≈ α·c  (zero intercept because p=0 at c=0)
    def _linear_fit_origin(cs, ys):
        valid = [(c, y) for c, y in zip(cs, ys) if math.isfinite(y) and c > 0]
        if len(valid) < 1:
            return None
        c_arr = np.array([c for c, _ in valid])
        y_arr = np.array([y for _, y in valid])
        alpha = float(np.dot(c_arr, y_arr) / np.dot(c_arr, c_arr))
        return alpha

    agent_alpha = _linear_fit_origin(agent_costs, agent_log1mp)
    os_alpha = _linear_fit_origin(agent_costs, os_log1mp)

    c_fit = np.linspace(0, max(agent_costs), 200)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.errorbar(agent_costs, agent_log1mp,
                yerr=[agent_log1mp_yerr_lo, agent_log1mp_yerr_hi],
                fmt="o", color="C0", capsize=4, label="SWE-agent")
    ax.errorbar(agent_costs, os_log1mp,
                yerr=[os_log1mp_yerr_lo, os_log1mp_yerr_hi],
                fmt="s", color="C1", capsize=4, label=r"$k$-shot")

    if agent_alpha is not None:
        ax.plot(c_fit, agent_alpha * c_fit, "--", color="C0", alpha=0.7,
                label=rf"fit ${agent_alpha:.2f}\,c$")
    if os_alpha is not None:
        ax.plot(c_fit, os_alpha * c_fit, "--", color="C1", alpha=0.7,
                label=rf"fit ${os_alpha:.2f}\,c$")

    ax.set_xlabel("Cost limit per attempt (USD)")
    ax.set_ylabel(r"$\ln(1 - p)$")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.35, linewidth=0.6)
    out3 = Path("..") / "overleaf" / "pics" / f"prob_est_{instance_id}_log1mp.pdf"
    plt.savefig(out3, bbox_inches="tight")
    plt.close(fig)

    # ---- Plot 4: cost vs number of queries ----
    agent_mean_queries = [d["mean_queries"] for d in agent_data]
    agent_mean_costs = [d["mean_cost"] for d in agent_data]

    # For k-shot: each attempt costs os_c and uses 1 query, so k attempts = k queries, cost = k * os_c
    os_queries_range = np.linspace(1, max(agent_mean_queries), 200)
    os_cost_curve = os_queries_range * os_c

    # Power-law fit with offset: cost ≈ c0 + c1 * q^γ
    from scipy.optimize import curve_fit

    def _power_with_offset(q, c0, c1, gamma):
        return c0 + c1 * np.power(q, gamma)

    q_arr = np.array(agent_mean_queries)
    c_arr = np.array(agent_mean_costs)
    popt, _ = curve_fit(_power_with_offset, q_arr, c_arr, p0=[0.0, 0.01, 1.5], maxfev=10000)
    c0_fit, c1_fit, gamma_fit = popt

    q_fit = np.linspace(0, max(agent_mean_queries) * 1.05, 200)
    cost_fit = _power_with_offset(q_fit, c0_fit, c1_fit, gamma_fit)
    positive_mask = cost_fit >= 0
    q_fit = q_fit[positive_mask]
    cost_fit = cost_fit[positive_mask]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(agent_mean_queries, agent_mean_costs,
            "o-", color="C0", label="SWE-agent")
    ax.plot(q_fit, cost_fit, "--", color="C0", alpha=0.7,
            label=rf"fit ${c0_fit:.3f} + {c1_fit:.4f}\,q^{{{gamma_fit:.2f}}}$")
    ax.plot(os_queries_range, os_cost_curve,
            "-", color="C1", label=r"$k$-shot")
    ax.set_xlabel("Mean number of queries")
    ax.set_ylabel("Mean cost (USD)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.35, linewidth=0.6)
    out4 = Path("..") / "overleaf" / "pics" / f"prob_est_{instance_id}_cost_vs_queries.pdf"
    plt.savefig(out4, bbox_inches="tight")
    plt.close(fig)

    print(generate_latex_code_for_figures(
        [out1.name, out3.name, out2.name, out4.name],
        [2, 2],
        rf"Probability estimation for problem {instance_id} (Sonnet 4, div.\@ {div}). "
        r"Top left: success probability with 95\% Clopper--Pearson CI. "
        r"Top right: $\ln(1-p)$ with linear fits. "
        r"Bottom left: failure-rate efficiency $-\ln(1-p)/c$. "
        r"Bottom right: mean cost versus mean number of queries.",
        f"prob-estimation-{instance_id}",
        use_subfigure=False,
    ))


def main():
    fig_prob_estimation()
    # fig_cum_solved_vs_cost()
    # fig_cum_solved_vs_queries()
    # fig_avg_cum_solved_vs_cost()
    # fig_avg_cum_solved_vs_cost()


if __name__ == "__main__":
    main()
