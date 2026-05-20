# Multi-LLM Provider Factory

A production-grade, asynchronous Python unified interface layer built to orchestrate communication across multiple LLM provider native APIs (OpenAI, Google GenAI, and AWS Bedrock). 

This system completely avoids the heavy dependency bloat and rigid abstractions of large orchestration frameworks by leveraging pure object-oriented design patterns. By using the **Strategy** and **Factory** patterns, the application core remains entirely decoupled from underlying vendor SDK changes, providing absolute structural autonomy, raw API performance, and clean debugging stacks.

---

## 🚀 Core Features

*   **Design-Pattern Driven:** Uses the Strategy pattern for provider isolation and the Factory pattern for clean runtime lifecycle routing.
*   **Dual Runtime Support:** Natively supports both high-throughput asynchronous (`agenerate`) and standard synchronous (`generate`) execution pipelines.
*   **Production Resiliency:** Built-in exponential backoff retry mechanisms utilizing the `tenacity` engine to seamlessly handle rate limits and transient network dropouts.
*   **Type Safe Telemetry:** Leverages **Pydantic V2** to enforce strict input data validation, handle structured JSON output generation, and provide structured metric readouts (prompt tokens, completion tokens, latency) per execution loop.
*   **Highly Extensible:** Features a pluggable runtime registry wrapper blueprint, allowing developers to connect custom drivers (e.g., local Ollama or specialized endpoints) without altering library files.

---

## 📂 Project Directory Structure

```text
multi_llm_factory/
│
├── core/                       # Core engine source module
│   ├── __init__.py             # Factory pattern engine & runtime registry
│   ├── config.py               # Pydantic V2 structural configuration & telemetry schemas
│   └── wrappers.py             # Provider Strategy implementation drivers
│
├── .env                        # Local infrastructure credentials (git ignored)
├── .gitignore                  # Git tracking exclusion filters
├── main.py                     # CLI Orchestration pipeline entry-point script
└── requirements.txt            # Explicit third-party platform dependencies