"""Compare: run the same questions through several models, then judge the saved answers.

Same two-step design as the Testing page (run, then judge). A compare *group* is one ordinary test run per model
(``benchmark_runs`` rows of kind ``compare`` sharing ``config["compare"]["group_id"]``), so every existing piece
applies unchanged: transcripts are saved with no verdict, the AI judge (``testing/judge.py``) or a person decides
afterwards, and each run's score follows its verdicts. Nothing here compares strings.

``session`` reads a group back as one side-by-side view; the jobs live in ``webui/compare_jobs.py``.
"""
