"""`fts analyze` — quality analysis of training data."""
from __future__ import annotations

import json as json_mod


def cmd_analyze(args) -> None:
    from finetune_studio.training.data_quality import (
        DataQualityAnalyzer,
        generate_fixes,
    )

    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(args.data)

    print(f"\n{'='*60}")
    print("DATA QUALITY REPORT")
    print(f"{'='*60}")
    print(f"File: {result['file']}")
    print(f"Total examples: {result['total_examples']}")
    print(f"Severity: {result['severity'].upper()}")
    print()

    if result['stats']:
        print("Statistics:")
        for k, v in result['stats'].items():
            print(f"  {k}: {v}")
        print()

    if result['issues']:
        print("Issues:")
        for issue in result['issues']:
            icon = "🔴" if issue['severity'] == "high" else "🟡" if issue['severity'] == "medium" else "🟢"
            print(f"  {icon} [{issue['severity'].upper()}] {issue['message']}")
        print()

        fixes = generate_fixes(result)
        if fixes:
            print("Suggested fixes:")
            for fix in fixes:
                print(f"  {fix['action']}: {fix['command']}")
    else:
        print("No issues found!")

    if args.json:
        print(f"\n{json_mod.dumps(result, indent=2)}")
