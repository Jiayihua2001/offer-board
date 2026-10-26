# Republish the board artifact (daily cloud routine, 9 AM Eastern)

The GitHub workflow "Update offer board" rebuilds `docs/index.html` twice a day and commits it. The claude.ai artifact is a published copy that only changes when a Claude session republishes it, which is this routine's single job. Do not run the sweep, and do not change or commit anything in the repository.

1. Note the commit you are on: `git log -1 --format='%h %ad %s' --date=iso`, and when the page last changed: `git log -1 --format=%ad --date=iso -- docs/index.html`.
2. Count the roles in the page with a one-liner, for the final report:
   `python3 -c "import re,json;h=open('docs/index.html').read();print(len(json.loads(re.search(r'window.BOARD_DATA = (.*);</script>',h).group(1).replace('<\\\\/','</'))['jobs']),'roles')"`
3. Republish with the Artifact tool, in this order:
   - `action: "read"` with `url: https://claude.ai/artifact/L7M5euPp224TP4N9YgjenZ`. The result names a saved file; Read that whole file (every line) so the session counts as having viewed the live version.
   - `action: "publish"` with the same `url`, `file_path: docs/index.html`, `label: "Routine <today's date>"`. Do not pass `capabilities` or `icon`, so the stored ones are kept. Never publish without `url`: that would create a separate artifact.
   - If the publish is refused because the live version changed, read again and publish again to the same `url`.
4. Finish with one line: the role count, the commit hash, and the artifact URL.
