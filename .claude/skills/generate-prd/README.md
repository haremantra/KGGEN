# generate-prd skill

A shareable Claude Code skill that interviews a stakeholder through a structured PRD-discovery questionnaire and produces a Product Requirements Document.

## What it does

Walks the user through 10 topic areas (~80 questions) covering product boundary, CRM, autodraft, PM, digests, Google integration, auth/security, vision-check CLI, architecture shortcuts, and delivery/budget — then writes a populated PRD to `docs/PRD.md`.

## Files

- `SKILL.md` — skill metadata + interview rules Claude follows.
- `questions.md` — the question bank, organized by topic.
- `prd-template.md` — the output structure Claude fills in.

## How to install

Drop the `generate-prd/` directory into either:

- **Project:** `.claude/skills/generate-prd/` (shared with the repo)
- **User:** `~/.claude/skills/generate-prd/` (available across all projects)

## How to invoke

```
/generate-prd
```

Or just ask: "interview me to write a PRD" / "help me scope this build".

## Customizing the question bank

Edit `questions.md`. Each topic is an `## h2` heading; questions are numbered. Add or remove topics to match your domain — the skill walks whatever is in the file.
