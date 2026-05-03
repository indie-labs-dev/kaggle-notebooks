---
description: Make a targeted code change in extension or worker
argument-hint: <what to change and where>
allowed-tools: Read, Write, Bash(find*), Bash(cat *)
model: claude-haiku-4-5-20251001
---
Do NOT ask clarifying questions. Make reasonable assumptions and proceed.
Silent mode: no explanations. Reply "Done" or "Blocked: {reason}". Follow CLAUDE.md.

$ARGUMENTS

Rules:
- Read only files mentioned in the task + directly related files
- Change minimum code needed - no refactoring beyond task scope
- User strings -> i18n.js, never hardcode
- No Gemini calls from extension - only via WORKER_URL
- No new libs in extension/

After writing:
1. Check for syntax errors
2. If worker.js changed - stop and ask: "worker.js changed. Deploy to Cloudflare? (wrangler deploy)"
3. Never deploy without explicit confirmation
4. Done