---
description: Execute a dev plan from docs/plans/
argument-hint: <plan filename, e.g. 000-example-dev-plan.md>
allowed-tools: Read, Write, Bash(find*), Bash(cat *), Bash(grep*)
model: claude-haiku-4-5-20251001
---
Do NOT ask clarifying questions. Follow the plan exactly as written.
Silent mode: no explanations. Reply "Done" or "Blocked: {reason}". Follow CLAUDE.md.

Read docs/plans/$ARGUMENTS and execute every step in order.
If you need a format reference, start with `docs/plans/000-example-dev-plan.md`.

Rules:
- Do exactly what the plan says - nothing more, nothing less
- Fix bugs listed in the plan before rename tasks
- Skip any step marked with "do not change"
- If a file from the plan does not exist - note it in final reply
- If worker.js was changed - stop and ask before deploying

After all steps:
1. Run grep to verify no missed references: `grep -rn "<old_term>" extension/ worker/`
2. Reply with: "Done. Changed: [list of files]" or "Blocked: {reason}"
