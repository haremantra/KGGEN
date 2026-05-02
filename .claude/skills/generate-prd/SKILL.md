---
name: generate-prd
description: Interview the user with a structured PRD-discovery questionnaire and produce a Product Requirements Document. Use when the user wants help scoping a new build, writing a PRD, or making decisions across product/CRM/PM/integration/auth/architecture/budget tradeoffs. Walks 10 topic areas (product boundary, CRM, autodraft, PM, digests, Google integration, auth/security, vision-check CLI, architecture shortcuts, delivery/budget) and writes the result to docs/PRD.md.
---

# Generate PRD via structured interview

Use this skill to elicit decisions from a stakeholder and produce a PRD. The questions live in `questions.md`. The output template lives in `prd-template.md`.

## How to run

1. Read `questions.md` to load the full question bank.
2. Confirm scope with the user before starting:
   - Ask which topic areas apply (default: all 10).
   - Ask the delivery format (single PRD doc, decision log, or both).
   - Ask the output path (default: `docs/PRD.md`).
3. Walk the user through one topic area at a time. For each area:
   - Present the questions as a numbered list in a single message.
   - Tell the user they can answer inline, say "skip" per question, or say "default" to accept the most common pragmatic choice (and you'll surface the assumption).
   - Wait for the user's reply before moving to the next area.
4. After every 2-3 areas, summarize decisions captured so far in 5-10 bullets and ask "any corrections before we continue?".
5. When all selected areas are answered, draft the PRD using `prd-template.md`. Fill every section; mark unresolved items as `TBD — <what's needed to resolve>`.
6. Write the PRD to the chosen path. Show the user the final file path and a 5-bullet executive summary.

## Interview rules

- **One area per turn.** Do not dump all 100+ questions at once.
- **No leading answers.** Ask the question; don't pre-fill the user's choice unless they say "default".
- **Track assumptions explicitly.** Any "default" or skipped answer becomes an `Assumptions` row in the PRD with a note that it needs validation.
- **Flag tradeoffs.** When an answer creates a downstream constraint (e.g. "Gmail send scope" forces OAuth verification), call it out before moving on.
- **Quote the user.** When an answer is non-obvious (timeline, budget, must-have), quote it verbatim in the PRD.
- **Don't invent requirements.** If the user didn't specify it, write `TBD`, not a guess.

## Output structure

The PRD must include, in order:
1. Executive summary (3-5 sentences)
2. Goals and non-goals
3. Target users and audience size
4. Scope per PR/milestone (PR 1, PR 2, PR 3 if applicable)
5. Functional requirements by area (only the areas the user selected)
6. Non-functional requirements (auth, security, RLS, compliance)
7. Architecture decisions and shortcuts
8. Integrations (Google scopes, third-party services)
9. Delivery plan (timeline, budget ceiling, must-have vs deferrable)
10. Open questions / TBDs
11. Assumptions log

## When to use this skill

Trigger when the user says any of:
- "write a PRD"
- "help me scope this build"
- "I need a spec for [project]"
- "interview me about [product]"
- references the question bank shipped with this skill
