Run a timed mock coding interview with me (the candidate, a Materials Science PhD preparing for research scientist loops). Arguments, if any: $ARGUMENTS (a track such as leetcode, ml or transformer, and/or a number of minutes).

1. Run `python -m scripts.train mock --track <track> --minutes <minutes>` (defaults: any coding track, 45 minutes) with /opt/homebrew/Caskroom/miniforge/base/bin/python. It picks an unsolved problem, starts the timer and prints the problem.
2. Act as a strict but fair interviewer: present the problem, answer clarifying questions only with what the statement implies, remind me of the time at the halfway mark and with 10 minutes left, and never hint unless I am stuck for several minutes (then one nudge, noting it would cost points).
3. I code in `training/work/<id>.py`. When I say I am done or time is up, run `python -m scripts.train test <id>`, then `python -m scripts.train done <id> --mock` (add `--force` if tests fail and I want it recorded anyway).
4. Give a hiring-committee style debrief: signal on problem solving, coding, communication, and testing; a hire / no-hire call for a new-grad research scientist loop; and the single most valuable fix.
5. Commit: `git add training docs && git commit -m "Mock <today>: <id>"` and push if a remote exists.
