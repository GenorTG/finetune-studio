"""Run one Guide tool for a project the way the Guide loop does, minus the UI event (test helper)."""

from finetune_studio.guide import tools as guide_tools


def run_tool(pid: str, name: str, args: dict) -> dict:
    result = guide_tools.run_tool(guide_tools.ToolContext(pid=pid), name, args)
    result.pop("ui_event", None)
    return result
