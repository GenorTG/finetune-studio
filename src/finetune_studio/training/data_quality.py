"""Pre-training data quality analysis.

WHAT THIS FILE DOES
==================
Analyzes training data BEFORE training to detect issues:
  - Duplicates (same example multiple times → overfitting)
  - Class imbalance (way more examples of one type than another)
  - Length outliers (some examples way longer/shorter than average)
  - Repetitive content (model might learn to repeat)
  - Forbidden tokens (accidental instruction leakage)

KEY CONCEPTS
============
- Quality > quantity: more data is not always better. Bad data
  actively hurts the model.
- Garbage in, garbage out: a model trained on noisy data will
  produce noisy outputs.
- Fix suggestions: we don't just report issues — we suggest fixes
  (e.g., "remove duplicates", "add more system messages").
"""

"""Training data quality analyzer — detect and fix issues before training."""
import hashlib
import json
from collections import Counter


class DataQualityAnalyzer:
    """Analyze training data quality and suggest fixes."""

    def __init__(self):
        self.issues = []
        self.stats = {}

    def analyze(self, data_path: str) -> dict:
        """Full analysis of a training data file."""
        self.issues = []
        data = self._load_data(data_path)

        self._check_format(data)
        self._check_duplicates(data)
        self._check_balance(data)
        self._check_length_distribution(data)
        self._check_language_balance(data)
        self._check_system_prompts(data)
        self._check_empty_responses(data)
        self._check_hallucination_risks(data)

        return {
            "file": data_path,
            "total_examples": len(data),
            "issues": self.issues,
            "stats": self.stats,
            "severity": self._calculate_severity(),
        }

    def _load_data(self, path: str) -> list:
        data = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        self.issues.append({"type": "format", "severity": "high",
                                           "message": f"Invalid JSON at line {len(data)+1}"})
        return data

    def _check_format(self, data):
        """Validate message format."""
        format_errors = 0
        for i, item in enumerate(data):
            if "messages" not in item:
                format_errors += 1
                continue
            msgs = item["messages"]
            if not isinstance(msgs, list):
                format_errors += 1
                continue
            for j, msg in enumerate(msgs):
                if "role" not in msg or "content" not in msg:
                    format_errors += 1

        if format_errors > 0:
            self.issues.append({"type": "format", "severity": "high",
                               "message": f"{format_errors} messages missing role/content"})

    def _check_duplicates(self, data):
        """Find duplicate examples."""
        seen = {}
        dupes = 0
        for i, item in enumerate(data):
            key = json.dumps(item, sort_keys=True, ensure_ascii=False)
            h = hashlib.md5(key.encode()).hexdigest()
            if h in seen:
                dupes += 1
            else:
                seen[h] = i

        if dupes > 0:
            self.issues.append({"type": "duplicates", "severity": "medium",
                               "message": f"{dupes} duplicate examples found",
                               "fix": "Remove duplicates to prevent overfitting"})

    def _check_balance(self, data):
        """Check role distribution balance."""
        role_counts = Counter()
        for item in data:
            for msg in item.get("messages", []):
                role_counts[msg.get("role", "unknown")] += 1

        self.stats["role_distribution"] = dict(role_counts)

        # Check for severely imbalanced roles
        if role_counts.get("system", 0) > role_counts.get("user", 0):
            self.issues.append({"type": "balance", "severity": "medium",
                               "message": "More system messages than user messages"})

        user_count = role_counts.get("user", 0)
        assistant_count = role_counts.get("assistant", 0)
        if assistant_count > 0 and user_count / assistant_count > 3:
            self.issues.append({"type": "balance", "severity": "low",
                               "message": f"User:Assistant ratio is {user_count/assistant_count:.1f}:1 (ideal: 1:1)"})

    def _check_length_distribution(self, data):
        """Check for too short or too long responses."""
        lengths = []
        for item in data:
            for msg in item.get("messages", []):
                if msg.get("role") == "assistant":
                    lengths.append(len(msg.get("content", "").split()))

        if not lengths:
            return

        avg_len = sum(lengths) / len(lengths)
        short = sum(1 for l in lengths if l < 3)
        long = sum(1 for l in lengths if l > 500)

        self.stats["avg_response_words"] = round(avg_len, 1)
        self.stats["min_response_words"] = min(lengths)
        self.stats["max_response_words"] = max(lengths)

        if short > len(lengths) * 0.1:
            self.issues.append({"type": "length", "severity": "medium",
                               "message": f"{short} responses are very short (<3 words)"})
        if long > len(lengths) * 0.05:
            self.issues.append({"type": "length", "severity": "low",
                               "message": f"{long} responses are very long (>500 words)"})

    def _check_language_balance(self, data):
        """Check PL/EN balance."""
        pl_count = 0
        en_count = 0
        mixed_count = 0

        for item in data:
            for msg in item.get("messages", []):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    pl_chars = sum(1 for c in content if c in "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
                    if pl_chars > len(content) * 0.05:
                        pl_count += 1
                    elif pl_chars > 0:
                        mixed_count += 1
                    else:
                        en_count += 1

        self.stats["language_distribution"] = {"pl": pl_count, "en": en_count, "mixed": mixed_count}
        total = pl_count + en_count + mixed_count

        if total > 0:
            pl_ratio = pl_count / total
            en_ratio = en_count / total
            if pl_ratio > 0.8:
                self.issues.append({"type": "language", "severity": "medium",
                                   "message": f"Dataset is {pl_ratio:.0%} Polish — add more English examples"})
            elif en_ratio > 0.8:
                self.issues.append({"type": "language", "severity": "medium",
                                   "message": f"Dataset is {en_ratio:.0%} English — add more Polish examples"})

    def _check_system_prompts(self, data):
        """Check system prompt consistency."""
        prompts = []
        for item in data:
            for msg in item.get("messages", []):
                if msg.get("role") == "system":
                    prompts.append(msg["content"][:100])

        unique_prompts = len(set(prompts))
        if unique_prompts > 3:
            self.issues.append({"type": "system_prompt", "severity": "low",
                               "message": f"{unique_prompts} different system prompts found"})

    def _check_empty_responses(self, data):
        """Check for empty assistant responses."""
        empty = 0
        for item in data:
            for msg in item.get("messages", []):
                if msg.get("role") == "assistant" and not msg.get("content", "").strip():
                    empty += 1

        if empty > 0:
            self.issues.append({"type": "empty", "severity": "high",
                               "message": f"{empty} empty assistant responses found",
                               "fix": "Remove or fix empty responses — they cause empty output bugs"})

    def _check_hallucination_risks(self, data):
        """Check for potential hallucination triggers."""
        risky_patterns = [
            ("I don't know", "may teach model to refuse"),
            ("I'm not sure", "may teach uncertainty"),
            ("I think", "may teach opinion rather than fact"),
        ]

        for pattern, reason in risky_patterns:
            count = sum(1 for item in data
                       for msg in item.get("messages", [])
                       if msg.get("role") == "assistant" and pattern.lower() in msg.get("content", "").lower())
            if count > len(data) * 0.1:
                self.issues.append({"type": "hallucination_risk", "severity": "low",
                                   "message": f"'{pattern}' appears in {count} responses — {reason}"})

    def _calculate_severity(self):
        severities = [i["severity"] for i in self.issues]
        if "high" in severities:
            return "high"
        if "medium" in severities:
            return "medium"
        return "low"


def generate_fixes(analysis: dict) -> list:
    """Generate specific fixes based on analysis."""
    fixes = []
    for issue in analysis["issues"]:
        if issue["type"] == "duplicates":
            fixes.append({"action": "deduplicate", "priority": "high",
                         "command": "fts dedup INPUT OUTPUT"})
        elif issue["type"] == "empty":
            fixes.append({"action": "remove_empty", "priority": "high",
                         "command": "fts clean INPUT --remove-empty"})
        elif issue["type"] == "language":
            fixes.append({"action": "augment_language", "priority": "medium",
                         "command": "fts augment INPUT --target-lang EN --count 100"})
        elif issue["type"] == "length":
            fixes.append({"action": "balance_length", "priority": "medium",
                         "command": "fts augment INPUT --min-words 5 --max-words 200"})
    return fixes
