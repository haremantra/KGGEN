# {{Project name}} — Product Requirements Document

_Generated via the `generate-prd` skill on {{date}}._

## 1. Executive summary

{{3-5 sentences: what we're building, for whom, why now, and the key tradeoff being made.}}

## 2. Goals and non-goals

**Goals**
- {{goal}}

**Non-goals**
- {{non-goal}}

## 3. Target users

- **Primary audience:** {{e.g. exec team, 3-10 users}}
- **Secondary audience (later):** {{e.g. function leads, managers}}
- **Out of scope:** {{e.g. employees, external customers}}

## 4. Scope per milestone

### PR 1 — {{title}}
- {{requirement}}

### PR 2 — {{title}}
- {{requirement}}

### PR 3 — {{title}}
- {{requirement}}

## 5. Functional requirements

_Include only the areas the user selected._

### 5.1 Product boundary
- {{captured answer}}

### 5.2 CRM
- **Contact record fields:** {{...}}
- **Account grouping:** {{...}}
- **Notes model:** {{editable vs append-only, markdown, attachments}}
- **Sharing:** {{exec-only, ownership rules}}
- **Search/filter:** {{PR1 vs later}}

### 5.3 Autodraft
- **Trigger:** {{auto vs manual}}
- **Storage:** {{Gmail draft vs in-app review}}
- **Send policy:** {{drafts only / never auto-send}}
- **Tone:** {{...}}
- **Context sources:** {{notes, calendar, Gmail history depth}}
- **Citations:** {{required / not}}
- **Model:** {{Opus / Sonnet / configurable}}

### 5.4 PM
- **Task model:** {{fields, milestones, dependencies}}
- **Ownership:** {{individual vs team}}
- **Activity history:** {{yes/no}}
- **Recurring/priority/reminders:** {{...}}

### 5.5 Digests
- **Delivery channel:** {{Gmail-as-exec / shared address / in-app preview}}
- **Cadence:** {{daily, weekly, opt-in}}
- **Ranking logic:** {{rules vs LLM}}
- **Formatting:** {{deterministic markdown vs LLM}}
- **Timezone:** {{...}}

### 5.6 Google integration
- **OAuth strategy:** {{...}}
- **Scopes:** {{calendar.readonly, gmail.readonly, gmail.compose, gmail.send}}
- **Sync model:** {{persisted vs on-demand}}
- **Storage of Gmail content:** {{snippets / encrypted / avoid}}
- **Contact matching:** {{exact / fuzzy}}

### 5.7 Vision-check CLI
- **Required in PR 1:** {{yes/no}}
- **Live LLM vs local questionnaire:** {{...}}
- **Outputs:** {{docs/vision.md, plus PR2/PR3 specs?}}

## 6. Non-functional requirements

- **Auth provider:** {{stub / Clerk / WorkOS / Google Workspace / Auth.js / other}}
- **RLS policy:** {{exec_all only / function_lead+manager read / future employee access}}
- **Audit logging:** {{triggers required in PR1 / deferred; UI-visible / DB-only}}
- **Token encryption at rest:** {{required / deferred}}
- **Compliance:** {{none / SOC2 / other}}

## 7. Architecture decisions and shortcuts

- **Workers:** {{BullMQ+Redis / Next.js route handlers / manual scripts}}
- **Cron:** {{Vercel Cron / long-running worker}}
- **Local dev:** {{Docker+Postgres stable / cleanup needed}}
- **dbt/warehouse:** {{out of scope, confirmed}}
- **Tests:** {{automated / typecheck + manual}}
- **Seed/demo data:** {{yes/no}}
- **Write path:** {{server actions only}}
- **Styling:** {{minimal / polished dashboard}}

## 8. Integrations

| Service | Purpose | Scope/Plan | Required in |
|---------|---------|------------|-------------|
| Google Calendar | Context for drafts | readonly | PR 2 |
| Gmail | Drafts/digests | {{readonly / compose / send}} | PR 2 |
| {{other}} | {{...}} | {{...}} | {{PR}} |

## 9. Delivery plan

- **Budget ceiling:** {{$X}}
- **Priority axis:** {{speed / cost / architecture}}
- **Must-have outcome:** {{...}}
- **Easiest defer:** {{...}}
- **Spec-before-code per PR:** {{yes/no}}
- **Milestone model:** {{fixed-fee / T&M weekly}}
- **Definition of done (first milestone):** {{...}}

## 10. Open questions / TBDs

- [ ] {{question that needs an answer before PR1 starts}}

## 11. Assumptions log

| # | Assumption | Source | Needs validation? |
|---|------------|--------|-------------------|
| 1 | {{e.g. exec_all is 5 users}} | default chosen during interview | yes |
