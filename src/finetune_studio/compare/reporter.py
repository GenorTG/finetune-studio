"""Format comparison results for display.

WHAT THIS FILE DOES
==================
Takes the raw comparison results and formats them for:
  - Console output (pretty tables, colored text)
  - HTML report (for sharing)
  - JSON file (for further analysis)
  - CSV (for spreadsheet import)

KEY CONCEPTS
============
- Presentation logic: separates "how to compute" from "how to display".
- Multiple output formats: the same data can be displayed differently.
- Summary statistics: in addition to per-test results, we compute
  averages and totals for quick comparison.
"""

"""Reporter — generate comparison reports."""
from datetime import datetime, timezone


def generate_report(comparison_results: list, scored: dict, output_path: str | None = None) -> str:
    """Generate a comparison report."""
    lines = []
    lines.append("=" * 60)
    lines.append("COMPARISON REPORT")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)

    # Summary
    lines.append("\n## SUMMARY\n")
    for source_name, data in scored["by_source"].items():
        lines.append(f"  {source_name}:")
        lines.append(f"    Pass rate: {data['pass_rate']}% ({data['passed']}/{data['total']})")
        lines.append(f"    Avg score: {data['avg_score']}")
        lines.append(f"    Avg time:  {data['avg_time_ms']}ms")
        lines.append("")

    # Detailed results
    lines.append("\n## DETAILED RESULTS\n")
    for test in comparison_results:
        lines.append(f"--- {test['name']} ---")
        for source_name, responses in test["responses"].items():
            for i, result in enumerate(responses):
                status = "ERROR" if result["error"] else "OK"
                lines.append(f"  [{source_name}] ({result['time_ms']}ms) {status}")
                if result["error"]:
                    lines.append(f"    Error: {result['error']}")
                else:
                    resp = result["response"]
                    lines.append(f"    {resp[:150]}{'...' if len(resp) > 150 else ''}")
        lines.append("")

    # Score breakdown
    lines.append("\n## SCORE BREAKDOWN\n")
    for score in scored["all_scores"]:
        if score.response:
            lines.append(f"  {score.test_name} [{score.source_name}]:")
            lines.append(f"    Total: {score.total_score} | Pass: {score.passed}")
            lines.append(f"    Keywords: {score.keyword_score} | Length: {score.length_score} | Time: {score.time_ms}ms")
            if score.details.get("keyword_misses"):
                lines.append(f"    Missing: {', '.join(score.details['keyword_misses'])}")
            if score.details.get("forbidden_hits"):
                lines.append(f"    Forbidden: {', '.join(score.details['forbidden_hits'])}")
            lines.append("")

    report_text = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report_text)

    return report_text


def generate_json_report(comparison_results: list, scored: dict) -> dict:
    """Generate a JSON-serializable report."""
    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "summary": {
            source: {
                "pass_rate": data["pass_rate"],
                "passed": data["passed"],
                "total": data["total"],
                "avg_score": data["avg_score"],
                "avg_time_ms": data["avg_time_ms"],
            }
            for source, data in scored["by_source"].items()
        },
        "results": [
            {
                "name": test["name"],
                "responses": {
                    source: [
                        {
                            "response": r["response"][:500],
                            "time_ms": r["time_ms"],
                            "error": r["error"],
                        }
                        for r in responses
                    ]
                    for source, responses in test["responses"].items()
                },
            }
            for test in comparison_results
        ],
    }
