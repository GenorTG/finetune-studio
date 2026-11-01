"""Structured-record boundaries in parsed text, shared by both chunkers (stdlib only)."""
from __future__ import annotations

import re


def record_ranges(lines: list[str]) -> dict[int, int]:
    """Return start->end ranges for repeated XML elements, JSON objects and YAML list records.

    Files & Knowledge passages (``data.prep.chunker``) and Smart Brain passages (``smart_brain.chunking``) both keep
    these records whole, so the two chunkers agree on what a record is."""
    ranges: dict[int, int] = {}
    xml_names = [re.match(r"\s*<([A-Za-z_:][\w:.-]*)\b", line) for line in lines]
    xml_counts: dict[str, int] = {}
    for match in xml_names:
        if match:
            xml_counts[match.group(1)] = xml_counts.get(match.group(1), 0) + 1
    i = 0
    while i < len(lines):
        line = lines[i]
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        end = i
        xml_open = re.match(r"<([A-Za-z_:][\w:.-]*)\b[^>]*>\s*$", stripped)
        if xml_open and not stripped.startswith("</") and xml_counts.get(xml_open.group(1), 0) > 1:
            tag = xml_open.group(1)
            depth = 1
            for j in range(i + 1, len(lines)):
                cur = lines[j]
                cur_indent = len(cur) - len(cur.lstrip())
                if cur_indent < indent:
                    break
                if re.search(rf"</{re.escape(tag)}\s*>", cur):
                    depth -= 1
                    if depth == 0:
                        end = j
                        break
                if re.search(rf"<{re.escape(tag)}(?:\s|>)", cur) and not re.search(rf"</{re.escape(tag)}\s*>", cur):
                    depth += 1
        elif stripped == "{" or stripped.startswith("{\""):
            depth = stripped.count("{") - stripped.count("}")
            if depth > 0:
                for j in range(i + 1, len(lines)):
                    cur = lines[j]
                    cur_indent = len(cur) - len(cur.lstrip())
                    if cur_indent < indent:
                        break
                    depth += cur.count("{") - cur.count("}")
                    if depth <= 0:
                        end = j
                        break
        elif re.match(r"-\s+\S", stripped):
            for j in range(i + 1, len(lines)):
                cur = lines[j]
                cur_indent = len(cur) - len(cur.lstrip())
                if cur.strip() and cur_indent <= indent and re.match(r"-\s+\S", cur.lstrip()):
                    break
                if cur.strip() and cur_indent < indent:
                    break
                end = j
        if end > i:
            ranges[i] = end
        i += 1
    return ranges
