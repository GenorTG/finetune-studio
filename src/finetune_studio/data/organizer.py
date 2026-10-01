"""Organize and deduplicate training files.

WHAT THIS FILE DOES
==================
Cleans up a directory of training files:
  - scan_data_files: lists data files (.jsonl/.json/.csv/.txt), newest first
  - dedup_data: drops rows whose canonical JSON is identical
"""

import json
import os


def scan_data_files(directory):
    files = []
    for root, dirs, filenames in os.walk(directory):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in filenames:
            if f.endswith((".jsonl", ".json", ".csv", ".txt")):
                fp = os.path.join(root, f)
                stat = os.stat(fp)
                files.append({
                    "path": fp, "name": f,
                    "relative": os.path.relpath(fp, directory),
                    "size_kb": round(stat.st_size / 1024, 1),
                    "modified": stat.st_mtime,
                })
    return sorted(files, key=lambda x: x["modified"], reverse=True)

def dedup_data(data):
    seen = set()
    unique = []
    dupes = 0
    for item in data:
        key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            unique.append(item)
        else:
            dupes += 1
    return unique, dupes
