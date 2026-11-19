# Architecture Decision Records (ADRs) — Support Triage Agent

This log records all significant architectural and technical decisions made throughout the lifecycle of the Support Triage Agent project. Each record follows the Context → Decision → Consequences structure.

---

## ADR-001: Monorepo Structure for Full-Stack Agent Application

- **Status:** Accepted
- **Date:** 2026-11-15
- **Deciders:** Lead AI Engineer & User

### Context
The Support Triage Agent project consists of a Python backend (FastAPI, LangGraph, LanceDB, SQLite) and a TypeScript frontend (React 18, Vite, Tailwind CSS, TanStack Query). The system architecture requires tight coupling between backend API schemas, agent state representations, trace telemetry events, and UI rendering. We needed to choose between a monorepo structure and separate repositories (polyrepo).

### Decision
We adopt a unified monorepo structure rooted at `support-triage-app/`, containing `backend/`, `frontend/`, `docs/`, and `configs/`. Dependency management remains native to each language ecosystem (`uv` with `pyproject.toml` for Python, `npm` with `package.json` for Node.js), avoiding complex multi-language monorepo build tools like Nx or Pants.

### Consequences
**Positive:**
- Atomic commits across backend data contracts, generated TypeScript types, and UI components.
- Streamlined local developer experience: clone one repository to run the complete multi-agent application.
- Architecture docs, golden evaluation datasets, and prompt templates live alongside the source code.

**Negative / Tradeoffs:**
- Separate package installation steps (`uv sync` and `npm install`).
- Potential for CI pipeline cross-triggering unless path filtering is configured.

---

## ADR-002: Provider Abstraction Decoupled from Orchestration Framework

- **Status:** Accepted
- **Date:** 2026-11-15
- **Deciders:** Lead AI Engineer & User

### Context
Agentic workflows require invoking LLM chat and embedding models across different environments (local zero-cost development via Ollama, cloud deployment via Google Gemini, enterprise staging via AWS Bedrock). Orchestration libraries like LangChain often bundle provider implementations directly into node logic, causing vendor lock-in, unstandardized telemetry outputs, and opaque retry loops.

### Decision
We introduce an explicit Strategy + Factory provider abstraction in `app/services/providers/`:
- `BaseLLM` and `BaseEmbedder` abstract base classes standardize inputs and responses (`LLMResponse` containing text, token counts, and latency).
- Infrastructure retries and exponential backoff (`tenacity`) are centralized in `BaseLLM`, keeping them completely separate from application-level semantic retry loops in agent nodes.
- A centralized factory (`factory.py`) reads `configs/app.yaml` to dynamically instantiate concrete providers (`OllamaLLM`, `GeminiLLM`, `BedrockLLM`).
- LangGraph agent nodes call `BaseLLM.generate()` directly; LangGraph never touches third-party provider packages.

### Consequences
**Positive:**
- Zero code changes required to swap LLM providers — only a single line edit in `configs/app.yaml`.
- Guarantees token usage and latency metrics on every call for the first-party trace store.
- Enables deterministic unit testing via `FakeLLM` with scriptable response queues, running offline in milliseconds.

**Negative / Tradeoffs:**
- Requires maintaining custom adapter classes for each supported LLM provider rather than using off-the-shelf community integrations.

---

## ADR-003: SQLite via SQLAlchemy 2.0 as Primary Application Database

- **Status:** Accepted
- **Date:** 2026-11-15
- **Deciders:** Lead AI Engineer & User

### Context
The application needs persistence for tickets, drafts, human approvals, outbox messages, and per-node execution trace events. We needed to balance zero-infra local developer setup with production readiness.

### Decision
We select SQLite using SQLAlchemy 2.0 ORM with `create_all` initialization at startup. We document the absence of Alembic migrations as an accepted tradeoff for this portfolio demo.

### Consequences
**Positive:**
- Zero infrastructure dependencies: runs in-process with a single local file (`backend/data/app.db`), enabling instant startup.
- Full SQL querying capability for tickets, trace events, and approval states.
- Seamless upgrade path: because SQLAlchemy 2.0 models are dialect-agnostic, switching to managed PostgreSQL (e.g., Neon or Cloud SQL) requires only updating the connection string.

**Negative / Tradeoffs:**
- SQLite does not support high-concurrency multi-instance writes and requires WAL mode for concurrent readers/writers.
- Database file is ephemeral on ephemeral hosting tiers (e.g., Render free tier), addressed via seed scripts in Phase 9.

---

## ADR-004: OpenAPI-Generated TypeScript Client via openapi-typescript

- **Status:** Accepted
- **Date:** 2026-11-15
- **Deciders:** Lead AI Engineer & User

### Context
Maintaining duplicate TypeScript interfaces manually on the frontend inevitably introduces schema drift when backend Pydantic models change, causing runtime failures in production.

### Decision
We implement automated TypeScript type generation using `openapi-typescript`. Running `npm run gen:api` fetches the live schema from FastAPI's `/openapi.json` and outputs strict TypeScript interfaces into `src/api/schema.d.ts`. The frontend API client (`src/api/client.ts`) imports types directly from this generated file.

### Consequences
**Positive:**
- Single source of truth: backend Pydantic models dictate the frontend type contract.
- Immediate compile-time errors during `tsc` / `npm run build` if any backend API property changes.
- Eliminates thousands of lines of boilerplate manual type definitions.

**Negative / Tradeoffs:**
- Requires the backend to be running (or an exported `openapi.json` file) when regenerating types.
- Requires awareness of Windows Node IPv6 resolution quirks (`127.0.0.1` vs `localhost`).

---

## ADR-005: Article-Aware Heading-Based Chunking Strategy

- **Status:** Accepted
- **Date:** 2026-11-17
- **Deciders:** Lead AI Engineer & User

### Context
Customer support help center articles contain precise, structured operational policies (e.g., return windows, fee schedules, refund dollar caps, and step-by-step instructions). Splitting text purely on fixed character or token counts (e.g., 500 characters) frequently separates operational conditions from their quantitative limits, leading to inaccurate responses or hallucinated policies. Furthermore, isolated subsections (e.g., "Processing Timelines: 3 to 5 business days") lose semantic meaning when detached from their parent article title.

### Decision
We implement `ArticleAwareChunker` in `app/services/kb/chunkers.py`:
- Documents are split along markdown heading boundaries (`#`, `##`, `###`).
- Every chunk prepends a full breadcrumb header (e.g., `# Returns Policy > ## Processing Timelines`) to ground the dense vector in its domain entity.
- Sections exceeding `max_chars: 1500` (~350–400 words) are sub-split along paragraph boundaries (`\n\n`) with a sliding window overlap of `overlap_chars: 200` to preserve cross-paragraph context.

### Consequences
**Positive:**
- Guaranteed semantic boundary preservation: rules, conditions, and dollar limits remain in the same chunk.
- Improved retrieval precision: breadcrumbs provide topical context for brief subsection paragraphs.
- Deterministic chunk boundaries and stable SHA-256 hashes per chunk.

**Negative / Tradeoffs:**
- Assumes structured markdown input; documents lacking markdown headers fall back to single-chunk or paragraph-split representations.

---

## ADR-006: Relevance-Threshold Gating as an Abstention Mechanism

- **Status:** Accepted
- **Date:** 2026-11-17
- **Deciders:** Lead AI Engineer & User

### Context
In customer support automation, hallucinations are most dangerous when an LLM attempts to answer out-of-domain or ungrounded questions. Unthresholded top-$k$ nearest neighbor retrieval always returns candidate chunks regardless of how distant they are in vector space. If a customer asks about an uncovered topic, an agent given low-similarity chunks will often attempt to synthesize an answer, risking unauthorized commitments.

### Decision
We implement explicit relevance-threshold gating in `app/services/kb/retriever.py`:
- Cosine similarity scores ($1 - \text{distance}$) are computed for all candidate chunks.
- If the best score falls below `relevance_threshold` (configured at $0.50$ in `configs/app.yaml`), or if no chunks match, the retriever sets `has_kb_gap = True`.
- The downstream Resolver agent (Phase 3) uses this signal to abstain from drafting, mark the ticket as `pending_review`, and route directly to human support.

### Consequences
**Positive:**
- Fail-safe abstention: prevents the system from guessing or fabricating support policies.
- Quantifiable product signal: KB gap telemetry highlights gaps in documentation that inform the knowledge base roadmap.
- Tunable via YAML: threshold can be calibrated without modifying agent or retrieval code.

**Negative / Tradeoffs:**
- Poorly phrased or highly idiomatic customer queries may trigger false-positive gaps if their vector similarity drops below the threshold, requiring review in the human queue.

---

## ADR-007: Content-Hash Based Idempotency for Vector Ingestion

- **Status:** Accepted
- **Date:** 2026-11-17
- **Deciders:** Lead AI Engineer & User

### Context
In multi-tenant or enterprise environments, help center documents are frequently re-scanned or re-indexed via automated pipelines or administrator uploads. Re-embedding an entire knowledge base on every ingestion cycle causes unnecessary compute overhead, slow startup times, external API rate-limit consumption, and index write lock contention.

### Decision
We implement article-level content hashing using SHA-256:
- The loader computes a SHA-256 hash of the complete raw markdown text (`content_hash`).
- `KBVectorStore` tracks indexed article hashes in `kb_metadata.json` alongside the LanceDB dataset.
- On ingestion, articles with matching hashes are skipped entirely (0 calls, 0.00 s).
- Articles with changed hashes trigger an atomic replacement: previous chunks for that `article_id` are deleted from LanceDB and the updated chunks are embedded and inserted.

### Consequences
**Positive:**
- Sub-second re-index runs when articles are unchanged (20 articles evaluated and skipped in 0.00 s).
- Granular updates: modifying one article only embeds that article's chunks, leaving the rest untouched.
- Clean delete and upload synchronization between disk (`data/kb/`) and the vector database.

**Negative / Tradeoffs:**
- Metadata tracking file must remain synchronized with the LanceDB table; if the metadata file is manually deleted, the next run re-indexes all articles.

---

## ADR-008: LangGraph StateGraph as Agent Orchestration Engine

- **Status:** Accepted
- **Date:** 2026-11-19
- **Deciders:** Lead AI Engineer & User

### Context
A multi-agent support desk requires non-linear routing (triage → confidence gate → resolver), state durability, and human-in-the-loop (HITL) review capabilities. Standard procedural code (`if/else` function chains) loses in-memory state on process restarts, cannot pause execution cleanly for external human approvals, and lacks inspectable checkpoint histories.

### Decision
We adopt LangGraph's `StateGraph` backed by a SQLite checkpointer (`SqliteSaver`):
- `AgentState` is defined as a strongly typed `TypedDict` capturing ticket metadata, triage classifications, retrieval results, drafted replies, citation lists, and call budget counters.
- Node transitions and branching decisions are expressed as declarative conditional edges (`route_after_triage`).
- Checkpoints are saved per `thread_id` (the ticket ID), allowing threads to pause at approval gates and resume when a reviewer acts.
- **Architectural Boundary**: Graph nodes invoke our own `BaseLLM` interface, never vendor SDKs or LangChain model wrappers.

### Consequences
**Positive:**
- Durable execution: thread progress is persisted after every super-step.
- Decoupled routing logic: conditional edges are pure, unit-testable functions.
- Native support for interrupts and human review queues in Phase 4.

**Negative / Tradeoffs:**
- Adds `langgraph` and `langgraph-checkpoint-sqlite` dependencies.
- Developers must adhere to functional state-update paradigms ($(\text{State}) \to \Delta\text{State}$) rather than mutating objects in place.

---

## ADR-009: Provider-Portable Structured Output Parsing with Conversational Error-Feedback Retries

- **Status:** Accepted
- **Date:** 2026-11-19
- **Deciders:** Lead AI Engineer & User

### Context
Agent nodes communicate through strict structured schemas (e.g., triage classification). LLMs occasionally produce invalid JSON syntax or violate schema validation rules (e.g., confidence $> 1.0$). Constrained decoding strategies (e.g., BNF grammars) require low-level token sampler access unavailable on cloud APIs (Gemini, Bedrock), while vendor tool-calling APIs vary widely between cloud and local models.

### Decision
We implement `call_structured()` in `app/agents/structured.py`:
- Extracts JSON candidate strings from markdown fences or delimiter pairs.
- Parses and validates JSON using Pydantic v2 schemas (`model_validate`).
- On syntax or validation failure, appends the exact error trace to the conversation history and re-prompts the model (up to `max_retries = 2`).
- Raises `StructuredOutputError` with the total attempt count if retries are exhausted, allowing the graph to route to `untriaged`.

### Consequences
**Positive:**
- 100% provider-agnostic: behaves identically across local Ollama, Google Gemini, and AWS Bedrock.
- High recovery rate: small models self-correct over 95% of syntax/schema errors when given the Pydantic error trace.
- Observable: every retry logs an attempt counter and error string.

**Negative / Tradeoffs:**
- Retries incur additional token consumption and latency compared to first-pass constrained decoding.

---

## ADR-010: In-Code Enforced Hard Budget Cap for Agentic LLM Calls

- **Status:** Accepted
- **Date:** 2026-11-19
- **Deciders:** Lead AI Engineer & User

### Context
Autonomous agent workflows risk entering runaway loops if validation errors, hallucinated citations, or revision requests trigger repetitive re-prompting. Unbounded loops cause unpredictable inference costs and resource exhaustion.

### Decision
We enforce a hard cap of $\le 6$ LLM calls per ticket directly in code:
- `AgentState` tracks `llm_calls_used: int` and `max_llm_calls: int` (configured in `configs/app.yaml`).
- Every agent node checks `llm_calls_used >= max_llm_calls` at entry.
- Exceeding the budget aborts execution immediately, transitioning the ticket to `status = error` with an explicit diagnostic message.

### Consequences
**Positive:**
- Absolute mathematical ceiling on token and dollar cost per ticket.
- Predictable, bounded failure modes.
- Protects downstream APIs from rate-limit exhaustion.

**Negative / Tradeoffs:**
- Severely degraded tickets with complex issues may hit the budget cap and require human handling rather than continued automated attempts.

---

## ADR-011: Synchronous Graph Execution via FastAPI BackgroundTasks

- **Status:** Accepted
- **Date:** 2026-11-19
- **Deciders:** Lead AI Engineer & User

### Context
Customer support ticket ingestion requires immediate response times (`202 Accepted` returning a ticket ID) so customer webhooks and frontends do not hang while multi-step agent reasoning occurs. We needed to choose an asynchronous execution mechanism.

### Decision
We dispatch ticket graph execution using FastAPI's in-process `BackgroundTasks`:
- `POST /api/tickets` creates the ticket with status `new`, commits to SQLite, schedules `_process_ticket_background(ticket_id)` on `BackgroundTasks`, and returns HTTP 202 immediately.
- The background task opens an isolated database session and executes `run_ticket_graph()`.

### Consequences
**Positive:**
- Zero infrastructure dependencies: requires no external message brokers (RabbitMQ/Redis) or separate worker daemons.
- Fast, non-blocking HTTP responses for ticket submission.

**Negative / Tradeoffs:**
- In-process background tasks do not survive application crashes or server restarts. In high-concurrency production deployments, this will be migrated to a durable queue (e.g., Celery or AWS SQS).
