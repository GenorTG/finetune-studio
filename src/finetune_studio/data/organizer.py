"""Deduplicate training rows.

  - dedup_data: drops rows whose canonical JSON is identical
"""

import json


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
