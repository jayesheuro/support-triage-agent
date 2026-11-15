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
