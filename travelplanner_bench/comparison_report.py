"""Comparison report generator: side-by-side metrics and Markdown output."""

from __future__ import annotations

import statistics
from typing import Any

from travelplanner_bench.backend import TokenUsage
from travelplanner_bench.token_tracking import estimate_cost


def generate_comparison_report(
    all_results: dict[str, list[dict]],
    frameworks: list[str],
    model: str,
) -> dict[str, Any]:
    """Generate side-by-side comparison metrics from framework results.

    Args:
        all_results: Framework name -> list of per-task result dicts.
        frameworks: Ordered list of framework names.
        model: Model name used across all frameworks.

    Returns:
        Structured comparison report dict.
    """
    report: dict[str, Any] = {
        "frameworks": frameworks,
        "model": model,
        "task_count": len(next(iter(all_results.values()), [])),
        "reliability": {},
        "token_efficiency": {},
        "timing": {},
        "per_level": {},
        "per_task": [],
    }

    for fw in frameworks:
        results = all_results.get(fw, [])
        total = len(results)
        if total == 0:
            continue

        # --- Reliability ---
        delivered = sum(1 for r in results if r.get("plan_delivered"))
        final_pass = sum(1 for r in results if r.get("final_pass"))
        cs_macro = sum(1 for r in results if r.get("commonsense_macro"))
        hard_macro = sum(1 for r in results if r.get("hard_macro"))
        errors = sum(1 for r in results if r.get("error"))

        cs_micros = [r.get("commonsense_micro", 0) for r in results]
        hard_micros = [r.get("hard_micro", 0) for r in results if r.get("plan_delivered")]

        report["reliability"][fw] = {
            "total": total,
            "delivered": delivered,
            "delivery_rate": delivered / total if total else 0,
            "final_pass": final_pass,
            "final_pass_rate": final_pass / total if total else 0,
            "commonsense_macro_rate": cs_macro / total if total else 0,
            "commonsense_micro_avg": statistics.mean(cs_micros) if cs_micros else 0,
            "hard_macro_rate": hard_macro / total if total else 0,
            "hard_micro_avg": statistics.mean(hard_micros) if hard_micros else 0,
            "errors": errors,
            "error_rate": errors / total if total else 0,
        }

        # --- Token Efficiency ---
        token_data = []
        for r in results:
            tu = r.get("token_usage", {})
            if tu:
                token_data.append(tu)

        total_input = sum(t.get("input_tokens", 0) for t in token_data)
        total_output = sum(t.get("output_tokens", 0) for t in token_data)
        total_tokens = sum(t.get("total_tokens", 0) for t in token_data)
        total_calls = sum(t.get("llm_calls", 0) for t in token_data)
        total_retrieval = sum(
            t.get("retrieval_input_tokens", 0) + t.get("retrieval_output_tokens", 0)
            for t in token_data
        )
        total_assembly = sum(
            t.get("assembly_input_tokens", 0) + t.get("assembly_output_tokens", 0)
            for t in token_data
        )

        usage = TokenUsage(
            input_tokens=total_input,
            output_tokens=total_output,
            total_tokens=total_tokens,
        )
        cost = estimate_cost(usage, model)
        cost_per_pass = cost / final_pass if final_pass > 0 else 0

        report["token_efficiency"][fw] = {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_tokens,
            "total_llm_calls": total_calls,
            "avg_tokens_per_task": total_tokens / total if total else 0,
            "avg_input_per_task": total_input / total if total else 0,
            "avg_output_per_task": total_output / total if total else 0,
            "avg_llm_calls_per_task": total_calls / total if total else 0,
            "avg_retrieval_tokens": total_retrieval / total if total else 0,
            "avg_assembly_tokens": total_assembly / total if total else 0,
            "estimated_cost_usd": cost,
            "cost_per_passing_task": cost_per_pass,
        }

        # --- Timing ---
        times = [r.get("wall_time_seconds", 0) for r in results]
        sorted_times = sorted(times)

        report["timing"][fw] = {
            "total_seconds": sum(times),
            "avg_per_task": statistics.mean(times) if times else 0,
            "p50_per_task": sorted_times[len(sorted_times) // 2] if sorted_times else 0,
            "p95_per_task": sorted_times[int(len(sorted_times) * 0.95)] if sorted_times else 0,
            "min_per_task": min(times) if times else 0,
            "max_per_task": max(times) if times else 0,
        }

        # --- Per-Level ---
        for level in ("easy", "medium", "hard"):
            level_results = [r for r in results if r.get("level") == level]
            if not level_results:
                continue
            lt = len(level_results)
            if level not in report["per_level"]:
                report["per_level"][level] = {}
            report["per_level"][level][fw] = {
                "total": lt,
                "delivery_rate": sum(1 for r in level_results if r.get("plan_delivered")) / lt,
                "final_pass_rate": sum(1 for r in level_results if r.get("final_pass")) / lt,
                "avg_tokens": sum(
                    r.get("token_usage", {}).get("total_tokens", 0) for r in level_results
                ) / lt,
            }

    # --- Per-Task comparison ---
    task_count = len(next(iter(all_results.values()), []))
    for i in range(task_count):
        task_comparison: dict[str, Any] = {}
        for fw in frameworks:
            results = all_results.get(fw, [])
            if i < len(results):
                r = results[i]
                task_comparison[fw] = {
                    "task_id": r.get("task_id"),
                    "final_pass": r.get("final_pass"),
                    "total_tokens": r.get("token_usage", {}).get("total_tokens", 0),
                    "wall_time": r.get("wall_time_seconds", 0),
                }
        if task_comparison:
            # Use task_id from first framework
            first_fw = frameworks[0]
            task_comparison["task_id"] = task_comparison.get(first_fw, {}).get("task_id", f"task_{i}")
            report["per_task"].append(task_comparison)

    return report


def generate_markdown_report(report: dict[str, Any]) -> str:
    """Generate a Markdown comparison report from the structured report."""
    frameworks = report["frameworks"]
    lines: list[str] = []

    lines.append("# TravelPlanner Framework Comparison Report")
    lines.append("")
    lines.append("## Configuration")
    lines.append(f"- **Model**: {report.get('model', 'N/A')}")
    lines.append(f"- **Tasks**: {report.get('task_count', 0)}")
    lines.append(f"- **Frameworks**: {', '.join(frameworks)}")
    lines.append("")

    # Reliability table
    lines.append("## Reliability")
    lines.append("")
    header = "| Metric |" + " | ".join(frameworks) + " |"
    sep = "|" + "---|" * (len(frameworks) + 1)
    lines.append(header)
    lines.append(sep)

    reliability = report.get("reliability", {})
    metrics = [
        ("Delivery Rate", "delivery_rate", True),
        ("Final Pass Rate", "final_pass_rate", True),
        ("Commonsense Macro", "commonsense_macro_rate", True),
        ("Commonsense Micro Avg", "commonsense_micro_avg", True),
        ("Hard Constraint Macro", "hard_macro_rate", True),
        ("Hard Constraint Micro Avg", "hard_micro_avg", True),
        ("Error Rate", "error_rate", True),
    ]
    for label, key, is_pct in metrics:
        row = f"| {label} |"
        for fw in frameworks:
            val = reliability.get(fw, {}).get(key, 0)
            row += f" {val:.1%} |" if is_pct else f" {val:.2f} |"
        lines.append(row)

    lines.append("")

    # Token efficiency table
    lines.append("## Token Efficiency")
    lines.append("")
    lines.append(header)
    lines.append(sep)

    tokens = report.get("token_efficiency", {})
    token_metrics = [
        ("Total Tokens", "total_tokens", "{:,.0f}"),
        ("Avg Tokens/Task", "avg_tokens_per_task", "{:,.0f}"),
        ("Avg Input/Task", "avg_input_per_task", "{:,.0f}"),
        ("Avg Output/Task", "avg_output_per_task", "{:,.0f}"),
        ("Avg LLM Calls/Task", "avg_llm_calls_per_task", "{:.1f}"),
        ("Avg Retrieval Tokens", "avg_retrieval_tokens", "{:,.0f}"),
        ("Avg Assembly Tokens", "avg_assembly_tokens", "{:,.0f}"),
        ("Estimated Cost (USD)", "estimated_cost_usd", "${:.3f}"),
        ("Cost/Passing Task", "cost_per_passing_task", "${:.4f}"),
    ]
    for label, key, fmt in token_metrics:
        row = f"| {label} |"
        for fw in frameworks:
            val = tokens.get(fw, {}).get(key, 0)
            row += f" {fmt.format(val)} |"
        lines.append(row)

    lines.append("")

    # Timing table
    lines.append("## Timing")
    lines.append("")
    lines.append(header)
    lines.append(sep)

    timing = report.get("timing", {})
    for label, key in [
        ("Total Time (s)", "total_seconds"),
        ("Avg/Task (s)", "avg_per_task"),
        ("P50/Task (s)", "p50_per_task"),
        ("P95/Task (s)", "p95_per_task"),
    ]:
        row = f"| {label} |"
        for fw in frameworks:
            val = timing.get(fw, {}).get(key, 0)
            row += f" {val:.1f} |"
        lines.append(row)

    lines.append("")

    # Per-level breakdown
    per_level = report.get("per_level", {})
    if per_level:
        lines.append("## Per-Level Breakdown")
        lines.append("")
        for level in ("easy", "medium", "hard"):
            if level not in per_level:
                continue
            lines.append(f"### {level.title()}")
            lines.append("")
            lines.append(header)
            lines.append(sep)
            level_data = per_level[level]
            for label, key, fmt in [
                ("Tasks", "total", "{:.0f}"),
                ("Delivery Rate", "delivery_rate", "{:.1%}"),
                ("Final Pass Rate", "final_pass_rate", "{:.1%}"),
                ("Avg Tokens", "avg_tokens", "{:,.0f}"),
            ]:
                row = f"| {label} |"
                for fw in frameworks:
                    val = level_data.get(fw, {}).get(key, 0)
                    row += f" {fmt.format(val)} |"
                lines.append(row)
            lines.append("")

    # Per-task comparison (first 20)
    per_task = report.get("per_task", [])
    if per_task:
        lines.append("## Per-Task Results (first 20)")
        lines.append("")
        task_header = "| Task |" + " | ".join(
            f"{fw} Pass | {fw} Tokens" for fw in frameworks
        ) + " |"
        task_sep = "|" + "---|" * (len(frameworks) * 2 + 1)
        lines.append(task_header)
        lines.append(task_sep)

        for tc in per_task[:20]:
            task_id = tc.get("task_id", "?")
            row = f"| {task_id} |"
            for fw in frameworks:
                fw_data = tc.get(fw, {})
                passed = fw_data.get("final_pass", False)
                toks = fw_data.get("total_tokens", 0)
                row += f" {'PASS' if passed else 'FAIL'} | {toks:,} |"
            lines.append(row)
        lines.append("")

    return "\n".join(lines)
