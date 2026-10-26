# Interview training

Daily practice for research scientist and applied scientist interview loops, living next to the offer board so one repo holds both the targets and the preparation.

- `curriculum.json`: the tracks and modules (coding interviews, ML fundamentals, build a transformer, ML system and research design).
- `assignments/<id>/`: one directory per assignment with `meta.json`, `README.md` (the problem), `starter.py`, `tests.py` and `solution.py` (written design prompts have only `meta.json` and `README.md`, with a rubric).
- `quiz.json`: concept flashcards graded with a Leitner schedule.
- `progress.json`: your settings and an append-only log of events (plans, starts, test runs, completions, quiz grades). Everything else is derived from it.
- `work/`: your solutions, one file per assignment.

The page `docs/train.html` is rendered from all of this by `python -m scripts.train render`. It runs Python tests in the browser (Pyodide) for pure-Python and NumPy assignments, keeps drafts and progress in the browser, and gives you a sync command to merge page progress back here.

Daily loop in Claude Code: type `/train`. For a timed mock interview: `/mock leetcode 45`. See `python -m scripts.train --help` for every command.
