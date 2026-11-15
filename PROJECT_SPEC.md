# MASTER CONTEXT — Support Triage Agent (paste this first in EVERY session)

You are the lead engineer building **Support Triage Agent** with me. I am a capable software
engineer with good AWS exposure, growing into senior AI full-stack engineering. I have
already built an Enterprise RAG system (provider abstraction, LanceDB, RAGAS, MLflow), so
RAG fundamentals are known territory — this project's new ground is **agentic systems**:
LangGraph orchestration, human-in-the-loop, failure recovery, guardrails, agent
observability, and agentic evals. This project is for my portfolio AND my learning — both
are first-class requirements. Read this entire document before doing anything; it defines
the product, architecture, conventions, and the documentation contract you must honor in
every phase.

---

## 1. Product vision

A multi-agent support desk, demoed for a fictional e-commerce store (**ShopLite**). Tickets
flow through an agent graph — Triage → Resolver → Critic — and are either auto-sent or
paused for human approval. The differentiator is **trust machinery**: every draft is checked
against a versioned policy set, every node decision is traced and inspectable in the UI, and
**auto-send is disabled until the system passes an eval gate on a golden dataset**.

Elevator pitch: *"Most agent demos stop at 'the LLM replied.' This one triages tickets
through a multi-agent graph with guardrails, pauses for human approval when policy demands
it, traces every decision, and refuses to automate at all until its eval metrics clear a
gate."*

**Plug-n-play principle**: the agent engine is domain-agnostic. Everything ShopLite-specific
lives in three swappable *data artifacts* — the **knowledge base** (markdown articles), the
**policy set** (versioned guardrail rules), and the **taxonomy** (categories, priorities,
routing, thresholds in YAML). Swapping those three onboards a different business with zero
code changes. Never hardcode ShopLite specifics into engine code.

## 2. Tech stack (fixed — do not substitute without asking me)

| Layer | Choice | Notes |
|---|---|---|
| Language (backend) | Python 3.11 | type hints everywhere, Pydantic v2 models |
| Backend | FastAPI + uvicorn | REST API, OpenAPI docs auto-generated |
| Agent orchestration | **LangGraph** | state graph, conditional edges, SQLite checkpointer, interrupts for HITL. LLM calls inside nodes go through OUR provider abstraction, never LangChain provider packages |
| UI | React 18 + Vite + TypeScript | Tailwind CSS, TanStack Query (server state), React Router, Recharts (dashboard). TS types generated from the backend OpenAPI schema via `openapi-typescript` |
| LLM providers | Ollama (local, DEFAULT), Gemini (cloud), Bedrock (stub) | swappable via config file ONLY — zero code changes. Chat: `llama3.1:8b` / `gemini-2.0-flash`-or-newer. Embeddings: `nomic-embed-text` / `text-embedding-004` |
| Vector store | LanceDB | embedded, file-based; KB articles only |
| App database | SQLite via SQLAlchemy 2.0 | tickets, drafts, approvals, outbox, trace events, feedback. `create_all` at startup (no Alembic — documented as an accepted tradeoff) |
| Tracking | MLflow (local sqlite backend) | prompt + policy version registry, eval experiment tracking |
| Observability | First-party tracer → SQLite trace store | per-node spans: decision, latency, tokens, retries. Langfuse documented as scale alternative in an ADR |
| Packaging | `uv` + `pyproject.toml` (backend), npm (frontend) | |
| Testing | pytest (backend), Vitest + React Testing Library (frontend, key flows only) | integration tests marked and skippable when Ollama/keys absent |
| Deployment | Docker → Render (backend) + Vercel (frontend), Phase 9; demo delivered as recorded video | |

## 3. Repository structure (canonical — create files exactly here)

```
support-triage-app/
├── PROJECT_SPEC.md                 # this document, saved into the repo in Phase 1
├── backend/
│   ├── pyproject.toml
│   ├── app/
│   │   ├── main.py                 # app factory, router registration, CORS
│   │   ├── core/                   # config.py (Pydantic Settings), logging.py, errors.py
│   │   ├── api/routes/             # tickets.py, approvals.py, kb.py, policies.py, evals.py, traces.py, health.py
│   │   ├── models/                 # Pydantic request/response schemas (the API contract)
│   │   ├── db/                     # SQLAlchemy models, session, enums (TicketStatus etc.)
│   │   ├── agents/
│   │   │   ├── graph.py            # LangGraph wiring: nodes, edges, gates, checkpointer
│   │   │   ├── state.py            # typed graph state
│   │   │   ├── nodes/              # triage.py, resolver.py, critic.py (one node = one file)
│   │   │   └── structured.py       # structured-output parsing + retry-with-error-feedback helper
│   │   └── services/
│   │       ├── providers/          # base.py (ABCs), ollama.py, gemini.py, bedrock.py, factory.py
│   │       ├── kb/                 # loaders.py, chunkers.py, vector_store.py, retriever.py, pipeline.py
│   │       ├── policies/           # loader.py, registry.py (versioned policy sets)
│   │       ├── tracing/            # tracer.py, store.py
│   │       └── evaluation/         # golden.py, offline.py, online.py, gate.py, mlflow_tracking.py
│   ├── prompts/                    # versioned prompt templates: triage.yaml, resolver.yaml, critic.yaml (name, version, template, changelog)
│   ├── policies/                   # versioned policy sets: shoplite_v1.yaml, ...
│   ├── configs/
│   │   ├── app.yaml                # active provider, thresholds, call caps, auto_send master switch
│   │   ├── taxonomy.yaml           # categories, priorities, routing, auto-send allowlist, confidence thresholds
│   │   ├── evals.yaml              # golden-set paths, gate thresholds
│   │   └── providers/              # ollama.yaml, gemini.yaml, bedrock.yaml
│   ├── data/
│   │   ├── kb/                     # ShopLite markdown articles (committed)
│   │   ├── fixtures/               # simulator tickets, red-team drafts (committed)
│   │   ├── lancedb/                # vector store files (GITIGNORED)
│   │   └── app.db                  # SQLite (GITIGNORED)
│   ├── eval/
│   │   ├── golden/                 # golden ticket datasets (committed, versioned)
│   │   └── reports/                # generated eval reports (GITIGNORED except .gitkeep)
│   ├── scripts/                    # seed_kb.py, simulate_tickets.py, run_eval.py, onboarding_gate.py, seed_demo.py
│   └── tests/                      # fakes.py + unit/integration tests
├── frontend/
│   ├── package.json
│   └── src/
│       ├── api/                    # generated types + typed client (openapi-typescript)
│       ├── components/             # TraceTimeline, DraftViewer, ViolationList, StatusBadge, ...
│       ├── pages/                  # Inbox, TicketDetail, ApprovalQueue, KBAdmin, Policies, EvalDashboard
│       └── hooks/
├── docs/
│   ├── learning/                   # phase1-*.md ... phase9-*.md  ← LEARNING CONTRACT
│   ├── question-bank/              # phase1-questions.md ... phase9-questions.md
│   └── architecture/               # decisions.md (ADR log), integration.md (how a business onboards), diagrams
├── docker-compose.yml              # Phase 9
├── .env.example                    # every env var documented; NEVER commit real .env
└── README.md
```

## 4. The agent graph (core architecture — implement exactly)

State machine per ticket (LangGraph `StateGraph`, SQLite checkpointer, one thread per ticket):

```
new ticket
  → TRIAGE node          — classify {category, priority, sentiment, churn_risk, confidence}
                           structured output; on parse/validation failure retry with the
                           error fed back (max 2 retries) → still failing: status=untriaged,
                           route to human queue, STOP
  → confidence gate      — triage confidence < taxonomy threshold → status=pending_review
                           (approval queue, with whatever triage produced), skip to END
  → RESOLVER node        — retrieve top-k KB chunks; if best score < relevance threshold:
                           flag kb_gap=true, status=pending_review, skip critic, END.
                           Else draft a reply that cites article ids like [KB-012]
  → CRITIC node          — check draft against the ACTIVE policy set version → structured
                           verdict: pass | revise | escalate, with violations[]
                           (policy_id, severity, span, reason)
  → revision loop        — verdict=revise AND revisions_used==0: resolver redoes the draft
                           with critic feedback appended, critic re-checks ONCE. Second
                           revise or escalate → pending_review
  → action gate          — auto-send ONLY IF: critic pass AND category is in taxonomy
                           auto-send allowlist AND global auto_send_enabled=true (the
                           onboarding eval gate flips this). Else: LangGraph INTERRUPT →
                           status=pending_review (approval queue)
  → human decision       — approve / edit / reject (+reason) resumes the graph thread
  → SEND                 — mock outbox table (no real email); record online feedback signal
```

Hard rules:
- **LLM call budget: ≤6 calls per ticket** (triage 1+≤2 retries, resolver 1+≤1 revision,
  critic 1+≤1 recheck). Enforce in code (a counter in graph state), not just in prose.
- Every node execution emits a **trace span** (Phase 5): node name, duration, model, token
  usage, prompt version, retry count, decision summary. No node is exempt.
- Prompts are never inline strings — always loaded from `backend/prompts/*.yaml`
  (name, version, template, changelog). The critic additionally loads the active policy set
  from `backend/policies/`. Versions of both are recorded on every trace and eval run.
- Nodes call `BaseLLM` from our provider abstraction. LangGraph never touches provider SDKs.

## 5. Provider abstraction (same pattern as my RAG project)

- `services/providers/base.py`: `BaseLLM.generate(messages, **kwargs) -> LLMResponse`
  (text, token usage, latency_ms, model_name) and `BaseEmbedder.embed_documents /
  embed_query` + `dimension` property. Retries with exponential backoff + per-request
  timeout live in the base layer.
- Implementations: Ollama (local HTTP, default), Gemini (`google-genai` SDK), Bedrock
  (boto3, fully written but marked untested-stub, lazy import).
- `factory.py` reads `configs/app.yaml` → `provider: ollama|gemini|bedrock` → loads
  `configs/providers/<name>.yaml` → returns instances. Switching = one YAML line. Test it.

## 6. Configuration & secrets

- All tunables live in `configs/*.yaml` — never hardcoded: provider+models, temperature,
  top_k, relevance threshold, confidence thresholds, retry caps, call budget,
  auto-send allowlist, gate thresholds.
- Secrets only in `.env` (gitignored), loaded via Pydantic Settings; `.env.example`
  documents every variable. Never print or log a secret.
- `auto_send_enabled` is a persisted runtime setting (DB), default **false** — only the
  onboarding eval gate (Phase 6) or an explicit admin action flips it.

## 7. THE LEARNING CONTRACT (non-negotiable, every phase)

I am learning agentic AI engineering through this build. Every phase, alongside the code,
you MUST produce/update:

1. **`docs/learning/phaseN-<topic>.md`**: Concepts (the theory behind what we built —
   e.g., what a state graph buys over a prompt chain, how LangGraph interrupts/checkpoints
   work, why structured-output retry loops beat "temperature 0 and pray", what
   precision/recall mean for a critic agent) explained for a strong SWE newer to agents,
   with small concrete examples; Decisions & tradeoffs (alternatives rejected and why);
   Business relevance (cost, trust, compliance, latency); a Mermaid diagram of the phase's
   data/control flow.
2. **`docs/question-bank/phaseN-questions.md`**: 12–15 interview questions WITH model
   answers — conceptual, decision-based, tradeoff, and scenario ("your auto-send rate
   dropped after a policy update — debug it"). Answers 4–8 sentences, specific, honest
   about tradeoffs.
3. **`docs/architecture/decisions.md`**: append an ADR (context → decision → consequences)
   for each significant choice.

Do NOT pause development for teaching sessions. Docs are a byproduct, written at phase end,
reflecting what was actually built.

## 8. Engineering conventions

Backend:
- Every module: docstring stating its responsibility; every public function: type hints +
  concise docstring. Pydantic models for all API bodies and inter-service data shapes.
- Custom exception hierarchy in `app/core/errors.py`; FastAPI handlers return structured
  error JSON; never bare `except:`.
- Structured logging (module, ticket_id, node, latency, token counts). Never log full
  customer message bodies at INFO, never log secrets at any level.
- Tests: every service module gets unit tests with fakes (`tests/fakes.py`: FakeLLM with a
  scriptable response queue — including malformed-then-valid sequences to exercise retry
  paths — and FakeEmbedder). Network-dependent tests marked `@pytest.mark.integration`,
  auto-skipped when Ollama/keys absent.

Frontend:
- TypeScript strict mode; no `any`. API types generated from OpenAPI — never hand-written
  duplicates of backend schemas.
- TanStack Query for ALL server state (no useEffect-fetch); polling via `refetchInterval`
  where the UI needs liveness. Local UI state stays in components; no global store unless
  genuinely needed (then Zustand, small).
- Components small and typed; loading/error/empty states for every data view (this is
  senior-signal territory — no blank screens).
- Accessibility basics: semantic HTML, labeled controls, keyboard-reachable actions.

Both:
- Commits small, conventional (`feat:`, `fix:`, `docs:`, `test:`); suggest the message, I
  run git. Windows-friendly: `pathlib`, no POSIX-only commands; everything must work in
  Git Bash on Windows 11.

## 9. Working style with me

- Each phase prompt has a "Questions to ask me FIRST" section. Ask those (plus anything
  genuinely blocking), wait for answers, then build. Don't re-ask what this spec fixes.
- Deliver in the numbered order listed; one-line status after each deliverable.
- These libraries move fast (langgraph, lancedb, mlflow, google-genai, openapi-typescript) —
  **check current versions on PyPI/npm before pinning**, and verify against the installed
  version's docs rather than guessing from training data.
- Acceptance criteria are MY checklist — make them pass before declaring done, and show me
  the commands to verify myself.
- If something can't work as specced (version conflict, API change, Windows quirk), STOP
  and present options with a recommendation — don't silently substitute.
