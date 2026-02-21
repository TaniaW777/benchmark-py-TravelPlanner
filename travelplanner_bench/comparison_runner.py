"""Unified comparison runner: run same tasks across multiple frameworks."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from travelplanner_bench.backend import BackendResult
from travelplanner_bench.models import TravelPlannerResult, TravelPlannerTask
from travelplanner_bench.token_tracking import estimate_cost

LOGS_DIR = Path(__file__).parent.parent / "logs"

log = logging.getLogger(__name__)


def _create_compare_dir() -> Path:
    """Create a timestamped comparison run directory."""
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = LOGS_DIR / f"compare_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _run_single_task(
    task: TravelPlannerTask,
    backend: Any,
    framework: str,
) -> dict[str, Any]:
    """Run a single task with a given backend and evaluate."""
    from travelplanner_bench.evaluation import evaluate_plan
    from travelplanner_bench.tools import ReferenceDatabase

    start = time.perf_counter()
    try:
        result: BackendResult = backend.solve(task)
        elapsed = time.perf_counter() - start

        db = ReferenceDatabase(task.reference_information)
        eval_result = evaluate_plan(result.plan, task, db)
        eval_result.iterations = result.iterations
        eval_result.wall_time_seconds = elapsed
        eval_result.framework = framework
        eval_result.model = getattr(backend, "_model", "")

        output = eval_result.model_dump()
        output["token_usage"] = result.token_usage.model_dump()
        output["raw_logs"] = result.raw_logs
        return output
    except Exception as e:
        elapsed = time.perf_counter() - start
        log.exception("Task %s failed with %s: %s", task.task_id, framework, e)
        output = TravelPlannerResult(
            task_id=task.task_id,
            query=task.query,
            level=task.level,
            days=task.days,
            framework=framework,
            wall_time_seconds=elapsed,
            error=str(e),
        ).model_dump()
        output["token_usage"] = {}
        output["raw_logs"] = []
        return output


def _write_task_log(
    fw_dir: Path, index: int, result: dict[str, Any]
) -> None:
    """Write a per-task log file for a framework."""
    log_path = fw_dir / f"task_{index:04d}_{result['task_id']}.md"
    lines = [
        f"# Task {index}: {result['task_id']}",
        "",
        f"**Framework**: {result.get('framework', '')}",
        f"**Query**: {result.get('query', '')}",
        f"**Level**: {result.get('level', '')}",
        f"**Days**: {result.get('days', '')}",
        "",
        "## Result",
        f"**Plan delivered**: {result.get('plan_delivered', False)}",
        f"**Final pass**: {result.get('final_pass', False)}",
        f"**Iterations**: {result.get('iterations', 0)}",
        f"**Time**: {result.get('wall_time_seconds', 0):.2f}s",
        "",
        "## Commonsense Constraints",
        f"- Within sandbox: {result.get('within_sandbox', False)}",
        f"- Complete info: {result.get('complete_info', False)}",
        f"- Within current city: {result.get('within_current_city', False)}",
        f"- Reasonable city route: {result.get('reasonable_city_route', False)}",
        f"- Diverse restaurants: {result.get('diverse_restaurants', False)}",
        f"- Diverse attractions: {result.get('diverse_attractions', False)}",
        f"- Non-conflicting transport: {result.get('non_conflicting_transport', False)}",
        f"- Valid accommodation: {result.get('valid_accommodation', False)}",
        f"- **Micro**: {result.get('commonsense_micro', 0):.2f}",
        f"- **Macro**: {result.get('commonsense_macro', False)}",
        "",
        "## Hard Constraints",
        f"- Budget: {result.get('budget_ok')}",
        f"- Room rule: {result.get('room_rule_ok')}",
        f"- Room type: {result.get('room_type_ok')}",
        f"- Cuisine: {result.get('cuisine_ok')}",
        f"- Transportation: {result.get('transportation_ok')}",
        f"- **Micro**: {result.get('hard_micro', 0):.2f}",
        f"- **Macro**: {result.get('hard_macro', False)}",
        "",
    ]

    if result.get("error"):
        lines.extend(["## Error", f"```\n{result['error']}\n```", ""])

    # Token usage
    tu = result.get("token_usage", {})
    if tu:
        lines.extend([
            "## Token Usage",
            f"- Input tokens: {tu.get('input_tokens', 0)}",
            f"- Output tokens: {tu.get('output_tokens', 0)}",
            f"- Total tokens: {tu.get('total_tokens', 0)}",
            f"- LLM calls: {tu.get('llm_calls', 0)}",
            f"- Retrieval: {tu.get('retrieval_input_tokens', 0)} in / {tu.get('retrieval_output_tokens', 0)} out ({tu.get('retrieval_llm_calls', 0)} calls)",
            f"- Assembly: {tu.get('assembly_input_tokens', 0)} in / {tu.get('assembly_output_tokens', 0)} out ({tu.get('assembly_llm_calls', 0)} calls)",
            "",
        ])

    plan = result.get("plan")
    if plan:
        lines.extend([
            "## Plan",
            "```json",
            json.dumps(plan, indent=2, default=str),
            "```",
            "",
        ])

    # Write raw logs (agent I/O, tool calls, sandbox enforcement)
    raw_logs = result.get("raw_logs", [])
    if raw_logs:
        lines.append("## Agent Logs")
        for entry in raw_logs:
            phase = entry.get("phase", "unknown")
            lines.append(f"\n### {phase}")
            if "description" in entry:
                lines.extend(["```", entry["description"], "```"])
            if "output" in entry:
                output = entry["output"]
                # Truncate very long outputs
                if len(output) > 5000:
                    output = output[:5000] + "\n... (truncated)"
                lines.extend(["```", output, "```"])
            if "gathered_data" in entry:
                lines.extend([
                    "```json",
                    json.dumps(entry["gathered_data"], indent=2, default=str),
                    "```",
                ])
            if "cleared_fields" in entry:
                for cf in entry["cleared_fields"]:
                    lines.append(
                        f"- Day {cf['day']}: cleared `{cf['field']}` = \"{cf['value']}\""
                    )
        lines.append("")

    log_path.write_text("\n".join(lines))


def run_comparison(
    frameworks: list[str],
    model: str,
    provider: str,
    split: str = "train",
    level: str | None = None,
    num: int | None = None,
    max_iterations: int = 10,
    parallel: int = 1,
    task_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Run the same tasks across multiple frameworks and generate comparison report."""
    from dotenv import load_dotenv

    load_dotenv()

    from travelplanner_bench.backends import get_backend
    from travelplanner_bench.comparison_report import generate_comparison_report
    from travelplanner_bench.data import get_level_counts, load_travelplanner

    # Load tasks
    if task_ids:
        max_index = max(int(tid.split("_")[1]) for tid in task_ids) + 1
        all_tasks = load_travelplanner(split=split, level=None, num=max_index, shuffle=False)
        requested = set(task_ids)
        tasks = [t for t in all_tasks if t.task_id in requested]
    else:
        tasks = load_travelplanner(split=split, level=level, num=num)
    level_counts = get_level_counts(tasks)

    print(f"\n{'=' * 60}")
    print("TravelPlanner Framework Comparison")
    print(f"{'=' * 60}")
    print(f"Model: {provider}/{model}")
    print(f"Frameworks: {', '.join(frameworks)}")
    print(f"Tasks: {len(tasks)} ({', '.join(f'{k}={v}' for k, v in level_counts.items())})")

    run_dir = _create_compare_dir()
    print(f"Logs: {run_dir}")

    # Run each framework
    all_results: dict[str, list[dict]] = {}

    for framework in frameworks:
        print(f"\n{'─' * 40}")
        print(f"Running: {framework}")
        print(f"{'─' * 40}")

        backend = get_backend(framework, model, provider, max_iterations=max_iterations)
        fw_dir = run_dir / framework
        fw_dir.mkdir(exist_ok=True)

        results: list[dict[str, Any]] = []
        total = len(tasks)
        completed = 0

        workers = min(parallel, total)
        if workers <= 1:
            for i, task in enumerate(tasks, 1):
                print(f"  [{i}/{total}] {task.query[:60]}...")
                result = _run_single_task(task, backend, framework)
                results.append(result)
                _write_task_log(fw_dir, i, result)
                status = "PASS" if result.get("final_pass") else "FAIL"
                tu = result.get("token_usage", {})
                tokens = tu.get("total_tokens", 0)
                print(
                    f"    -> {status} | "
                    f"cs={result.get('commonsense_micro', 0):.0%} | "
                    f"hard={result.get('hard_micro', 0):.0%} | "
                    f"tokens={tokens} | "
                    f"time={result.get('wall_time_seconds', 0):.1f}s"
                )
        else:
            futures_map: dict[Any, tuple[int, TravelPlannerTask]] = {}
            indexed_results: dict[int, dict] = {}

            with ThreadPoolExecutor(max_workers=workers) as executor:
                for i, task in enumerate(tasks, 1):
                    future = executor.submit(_run_single_task, task, backend, framework)
                    futures_map[future] = (i, task)

                for future in as_completed(futures_map):
                    idx, task = futures_map[future]
                    completed += 1
                    try:
                        result = future.result()
                    except Exception as e:
                        result = TravelPlannerResult(
                            task_id=task.task_id,
                            query=task.query,
                            level=task.level,
                            days=task.days,
                            framework=framework,
                            error=str(e),
                        ).model_dump()
                        result["token_usage"] = {}
                        result["raw_logs"] = []
                    indexed_results[idx] = result
                    _write_task_log(fw_dir, idx, result)
                    status = "PASS" if result.get("final_pass") else "FAIL"
                    print(f"    [{completed}/{total}] {status} | {task.query[:50]}...")

            for i in range(1, total + 1):
                results.append(indexed_results[i])

        # Save per-framework results
        (fw_dir / "results.json").write_text(
            json.dumps(results, indent=2, default=str)
        )

        all_results[framework] = results

        # Print framework summary
        passed = sum(1 for r in results if r.get("final_pass"))
        delivered = sum(1 for r in results if r.get("plan_delivered"))
        total_tokens = sum(
            r.get("token_usage", {}).get("total_tokens", 0) for r in results
        )
        print(f"  Summary: {passed}/{total} passed, {delivered}/{total} delivered, {total_tokens} total tokens")

    # Generate comparison report
    report = generate_comparison_report(all_results, frameworks, model)
    (run_dir / "comparison_summary.json").write_text(
        json.dumps(report, indent=2, default=str)
    )

    # Generate markdown report
    from travelplanner_bench.comparison_report import generate_markdown_report

    md_report = generate_markdown_report(report)
    (run_dir / "comparison_report.md").write_text(md_report)

    # Print comparison
    _print_comparison(report)
    print(f"\nFull report: {run_dir / 'comparison_report.md'}")

    return report


def _print_comparison(report: dict[str, Any]) -> None:
    """Print side-by-side comparison to console."""
    frameworks = report["frameworks"]

    print(f"\n{'=' * 60}")
    print("COMPARISON RESULTS")
    print(f"{'=' * 60}")

    # Reliability
    print("\nReliability:")
    header = f"  {'Metric':<30}" + "".join(f"{fw:>16}" for fw in frameworks)
    print(header)
    print("  " + "─" * (30 + 16 * len(frameworks)))

    reliability = report.get("reliability", {})
    for metric in ["delivery_rate", "final_pass_rate", "commonsense_macro_rate", "hard_macro_rate", "error_rate"]:
        label = metric.replace("_", " ").title()
        row = f"  {label:<30}"
        for fw in frameworks:
            val = reliability.get(fw, {}).get(metric, 0)
            row += f"{val:>15.1%}"
        print(row)

    # Token Efficiency
    print("\nToken Efficiency:")
    header = f"  {'Metric':<30}" + "".join(f"{fw:>16}" for fw in frameworks)
    print(header)
    print("  " + "─" * (30 + 16 * len(frameworks)))

    tokens = report.get("token_efficiency", {})
    for metric, fmt in [
        ("avg_tokens_per_task", "{:>15,.0f}"),
        ("avg_llm_calls_per_task", "{:>15.1f}"),
        ("avg_retrieval_tokens", "{:>15,.0f}"),
        ("avg_assembly_tokens", "{:>15,.0f}"),
        ("estimated_cost_usd", "${:>14.3f}"),
        ("cost_per_passing_task", "${:>14.4f}"),
    ]:
        label = metric.replace("_", " ").title()
        row = f"  {label:<30}"
        for fw in frameworks:
            val = tokens.get(fw, {}).get(metric, 0)
            row += fmt.format(val)
        print(row)

    # Timing
    print("\nTiming:")
    timing = report.get("timing", {})
    for metric in ["avg_per_task", "p50_per_task", "p95_per_task"]:
        label = metric.replace("_", " ").title()
        row = f"  {label:<30}"
        for fw in frameworks:
            val = timing.get(fw, {}).get(metric, 0)
            row += f"{val:>15.1f}s"
        print(row)


def main() -> int:
    """CLI entry point for framework comparison."""
    parser = argparse.ArgumentParser(
        description="TravelPlanner Framework Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all 3 frameworks on train split
  uv run python -m travelplanner_bench.comparison_runner \\
      --frameworks opensymbolicai,langchain,crewai \\
      --model gpt-4o --provider openai

  # Quick test with just LangChain on 5 tasks
  uv run python -m travelplanner_bench.comparison_runner \\
      --frameworks langchain --model gpt-4o --provider openai --num 5

  # Compare OpenSymbolicAI vs LangChain on easy tasks
  uv run python -m travelplanner_bench.comparison_runner \\
      --frameworks opensymbolicai,langchain --level easy \\
      --model gpt-4o --provider openai
        """,
    )
    parser.add_argument(
        "--frameworks", "-f",
        type=str,
        default="opensymbolicai,langchain,crewai",
        help="Comma-separated list of frameworks (default: all three)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="LLM model name/ID (same for all frameworks)",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "fireworks", "groq", "ollama"],
        required=True,
        help="LLM provider",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="train",
        help="Dataset split (default: train)",
    )
    parser.add_argument(
        "--level", "-l",
        choices=["easy", "medium", "hard"],
        default=None,
        help="Filter by difficulty level",
    )
    parser.add_argument(
        "--num", "-n",
        type=int,
        default=None,
        help="Number of tasks to run (default: all)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Max agent iterations per task (default: 10)",
    )
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=1,
        help="Number of parallel workers per framework (default: 1)",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        default=None,
        help="Comma-separated task IDs to run (e.g., tp_0003,tp_0010)",
    )
    args = parser.parse_args()

    frameworks = [f.strip() for f in args.frameworks.split(",")]
    task_ids = args.task_ids.split(",") if args.task_ids else None

    run_comparison(
        frameworks=frameworks,
        model=args.model,
        provider=args.provider,
        split=args.split,
        level=args.level,
        num=args.num,
        max_iterations=args.max_iterations,
        parallel=args.parallel,
        task_ids=task_ids,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
