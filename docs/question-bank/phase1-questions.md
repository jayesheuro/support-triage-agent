# Phase 1 Question Bank — Monorepo Scaffold, Multi-Provider Architecture & Type Systems

Interview questions and model answers covering architectural decisions, provider abstractions, type-safety guarantees, and full-stack operational concerns.

---

### Q1: Why generate TypeScript types from OpenAPI instead of writing interfaces by hand?
**Answer:** Hand-authoring TypeScript interfaces to mirror backend schemas creates an inevitable divergence known as schema drift. In an agentic application where data models evolve rapidly — such as adding violation severity levels, token usage fields, or confidence scores — manual synchronization depends on developer discipline and easily fails during high-velocity iterations. Generating types directly from FastAPI's `/openapi.json` via tools like `openapi-typescript` makes the backend Pydantic models the single source of truth across the entire stack. If a backend engineer renames or removes a property, the frontend build fails at compile time (`tsc`), identifying the exact line in UI components needing remediation before runtime errors reach production. The minor tradeoff is needing a generation step in CI/CD and ensuring the backend schema is up to date before regenerating.

---

### Q2: Why choose a monorepo over a polyrepo for a two-tier agent application?
**Answer:** A monorepo ensures that backend logic, frontend UI, data taxonomy, and architecture documentation evolve together within atomic commits. In an agentic system where backend changes directly alter frontend representations (such as rendering new agent node trace timelines or human-in-the-loop interrupt payloads), separating the codebase into multiple repositories introduces git coordination overhead, version tagging friction, and desynchronized pull requests. Having backend and frontend code in a single repository simplifies end-to-end local development, allowing an engineer to clone one repo and run the full stack immediately. The accepted tradeoff is maintaining distinct dependency managers (`uv` for Python, `npm` for Node), which we resolve by keeping clear directory boundaries (`backend/` and `frontend/`) rather than forcing a heavy multi-language monorepo orchestrator.

---

### Q3: Where do retries belong — in the provider base class or at the agent call sites — and why?
**Answer:** Retries for transient infrastructure failures (such as HTTP timeouts, connection resets, and vendor rate limits) belong strictly in the provider base class (`BaseLLM`), whereas retries for semantic output failures belong at the agent node layer. By centralizing infrastructure retries in `BaseLLM` using an exponential backoff decorator (`tenacity`), every agent node (Triage, Resolver, Critic) automatically gains resilience against intermittent network blips without duplicating retry loops across the codebase. Conversely, when an LLM produces malformed JSON or violates an output schema, retrying the identical prompt blindly is often futile. Those semantic failures must propagate to the agent node, which constructs a new prompt feeding the validation error back to the model for calibrated correction. Conflating infrastructure retries with semantic error-feedback loops is an anti-pattern that leads to uncoordinated retry storms.

---

### Q4: How would you add a fourth LLM provider (e.g., Anthropic Claude direct API or Azure OpenAI) to this codebase?
**Answer:** Adding a fourth provider requires four discrete steps with zero changes to existing agent or graph code. First, create a new module in `app/services/providers/` (e.g., `anthropic.py`) subclassing `BaseLLM` and `BaseEmbedder`, implementing `_generate_impl` and `_embed_documents_impl`. Second, register the provider name in `configs/providers/anthropic.yaml` with its specific model identifiers, token limits, and timeouts, and add any required API secrets to `.env.example` and `app/core/config.py`. Third, add a builder function to `_BUILDERS` in `app/services/providers/factory.py` mapping the configuration string to the new class instances. Finally, set `provider: anthropic` in `configs/app.yaml`. Because all agent nodes interact exclusively with the `BaseLLM` contract, the entire multi-agent graph immediately runs on the new provider.

---

### Q5: Why decouple the LLM provider layer from orchestration frameworks like LangGraph?
**Answer:** Orchestration frameworks frequently bundle provider integrations (e.g., `langchain-openai` or `langchain-google-genai`), which creates vendor lock-in to the framework's internal abstractions and release cycles. By placing our own `BaseLLM` strategy pattern between LangGraph nodes and underlying LLM APIs, we maintain complete control over raw HTTP transport, request timeouts, token usage tracking, and retry backoff policies. LangGraph manages the state machine, checkpoint transitions, and conditional edge routing, while our provider abstraction handles raw inference. This separation allows us to swap, mock, or instrument the LLM layer without breaking graph topology, and ensures that testing nodes with scripted fake responses (`FakeLLM`) requires zero external network mocking or monkeypatching of third-party libraries.

---

### Q6: What is the benefit of using Pydantic Settings over standard environment variable access via `os.environ`?
**Answer:** `os.environ` provides unstructured string access without validation, fail-fast guarantees, or type coercion, leading to runtime failures deep inside execution paths when a key is missing or malformed. Pydantic Settings (`BaseSettings`) validates all required configuration and environment variables at startup, parsing datatypes (e.g., integer port numbers, float thresholds, boolean flags) and providing helpful error messages if required credentials are omitted. Furthermore, it cleanly decouples runtime secrets stored in `.env` from structural application settings loaded from committed YAML files (`configs/app.yaml`). Centralizing this in a typed `Settings` object allows dependency injection throughout the app and makes unit testing trivial by passing isolated mock settings objects without mutating global process state.

---

### Q7: Why use TanStack Query instead of native React `useEffect` and `useState` for server data fetching?
**Answer:** Native `useEffect` fetching treats remote server state like local component state, forcing developers to manually implement loading spinners, error boundaries, request deduplication, cache invalidation, and race-condition guards. TanStack Query treats server data as an asynchronous cache owned by the backend. It provides out-of-the-box support for declarative polling (`refetchInterval`), background revalidation on window focus, cached response deduplication across multiple subscriber components, and explicit status flags (`isLoading`, `isError`, `isFetching`). In our Health monitor, this allows the UI to poll the backend every 10 seconds, render immediate graceful degradation banners when the API goes offline, and automatically recover when the backend restarts — all without writing a single line of timer or cleanup code.

---

### Q8: How does the application prevent accidental data leakage or secret exposure in logs?
**Answer:** We implement defensive logging practices across multiple architectural boundaries. First, our structured logger (`app/core/logging.py`) formats messages and metadata into single-line JSON records while explicitly omitting customer message bodies and raw prompts from `INFO`-level logs. Second, provider API keys (such as `GOOGLE_API_KEY` and AWS credentials) are strictly loaded via Pydantic Settings from `.env` and are never attached to logging dictionaries, trace spans, or error payloads returned by FastAPI exception handlers. Third, `.env` and runtime database files are strictly listed in root `.gitignore`, verified by automated tests to ensure sensitive configuration files cannot be accidentally committed to source control.

---

### Q9: What are the latency and operational tradeoffs between local Ollama inference and cloud APIs like Google Gemini?
**Answer:** Local Ollama inference guarantees zero API monetary costs, total data privacy, and offline functionality, making it ideal for continuous development, unit testing, and budget-conscious evaluation runs. However, local inference on consumer hardware (such as 8 GB RAM on CPU) is bounded by hardware throughput, exhibiting generation latencies of 2 to 6 seconds per call and lower reasoning capacity on smaller quantized models like `llama3.2:3b`. Cloud providers like `gemini-2.0-flash` provide sub-second time-to-first-token, superior contextual reasoning, and higher concurrency, but introduce per-token costs, rate quotas, and network dependencies. Our provider abstraction allows development and regression testing to run locally for free on Ollama, with an instant switch to Gemini for production deployment and high-throughput evaluation gates.

---

### Q10: Why is SQLite an acceptable choice for the application database and trace store in this portfolio project?
**Answer:** SQLite requires zero infrastructure setup, runs embedded in-process, and persists reliably to a single local file (`backend/data/app.db`), making the application entirely self-contained without needing Docker or a managed cloud database for local execution. For an e-commerce support triage system operating as a single-node service, SQLite easily handles hundreds of transactions per second, which comfortably exceeds typical multi-agent support desk workloads. We document SQLite as an accepted architectural tradeoff: while it lacks horizontal scaling and requires WAL mode for concurrent write operations, SQLAlchemy 2.0 abstracts the ORM layer, allowing seamless migration to PostgreSQL (such as Neon or Cloud SQL) for enterprise deployments with zero application code changes.

---

### Q11: How does `FakeLLM` in `tests/fakes.py` enable rigorous unit testing of agent failure recovery loops?
**Answer:** `FakeLLM` implements the `BaseLLM` interface but replaces network inference with a scriptable FIFO response queue that can return pre-canned strings or raise specific exceptions. This enables deterministic testing of retry paths: by enqueuing a malformed JSON string followed by a schema-compliant string, a unit test can verify that an agent node correctly detects the parsing error, attaches the validation failure to the context, and re-invokes the LLM to successfully recover. Because `FakeLLM` operates entirely in-memory with zero network latency or API dependency, the entire unit test suite runs in under two seconds in CI environments without requiring live Ollama instances or cloud credentials.

---

### Q12: Scenario: Your team switches `provider: gemini` in `configs/app.yaml`, but the health endpoint reports status "ok" while agent calls fail. How do you diagnose this?
**Answer:** The `/api/health` endpoint is designed as a lightweight liveness and configuration check: it validates that the YAML configuration files can be parsed and that active model strings are recognized, but deliberately avoids making an external inference call on every health check to prevent rate-limit consumption and latency spikes. If agent calls fail while health is "ok", the issue is at the provider credential or transport boundary — most commonly a missing or invalid `GOOGLE_API_KEY` in `.env`, a network firewall blocking outbound requests to Google APIs, or an exhausted API quota. The diagnosis procedure is to run `python scripts/smoke_test.py`, which executes an active end-to-end `generate()` and `embed_query()` call, surfacing the exact `ProviderAuthError` or `ProviderRateLimitError` immediately.

---

### Q13: Why is `auto_send_enabled` hardcoded to `false` by default in `configs/app.yaml`?
**Answer:** In an autonomous agent system with real-world side effects (such as emailing customers or issuing store credits), safety and trust machinery must be defensive by default. Setting `auto_send_enabled: false` ensures that all generated drafts are paused at the human-in-the-loop approval gate (`pending_review`), regardless of the agent's confidence score or critic evaluation. As defined in our product vision, auto-send automation can only be enabled dynamically after the system executes an offline evaluation run against a versioned golden ticket dataset and successfully clears strict precision and recall thresholds. Defaulting to `false` prevents unverified models from communicating autonomously with end users.

---

### Q14: How does the application handle Windows-specific runtime quirks (such as path separators and IPv6 resolution)?
**Answer:** We enforce strict cross-platform engineering practices. All filesystem manipulations use Python's standard `pathlib.Path`, which handles Windows backslashes and POSIX forward slashes transparently, avoiding brittle hardcoded path strings. In the console and CLI scripts, `sys.stdout` is explicitly reconfigured to UTF-8 to prevent `UnicodeEncodeError` failures on Windows code page 1252 when rendering status indicators. Additionally, in Node.js 20+ environments on Windows, `localhost` often resolves to IPv6 `::1` before IPv4 `127.0.0.1`, which causes `ECONNREFUSED` if the ASGI server binds only to IPv4. We explicitly configure the frontend proxy and API generation scripts to target `http://127.0.0.1:8000`, eliminating connection drops across Windows developer environments.
