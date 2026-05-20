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
```

🚀 Setup & Installation

    Clone & Navigate:
    Bash

    git clone your-repo-url-here
    cd llm-optimization-engine


2. **Environment Configuration:**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here

    Install Dependencies:
    Bash

    pip install -r requirements.txt


---

## 💻 CLI Usage Guide

The utility provides two core execution paths through `src/main.py`.

### 1. Default Interactive Mode
Pass a direct text prompt via the command line arguments.
```bash
python src/main.py --prompt "Analyze our system logs for anomalies" --provider openai --model gpt-4o

2. File Ingestion Mode

Process long text payloads directly from an external file target (e.g., system trace logs, documentation files).
Bash

python src/main.py --file path/to/logs.txt --provider anthropic --model claude-3-5-sonnet-latest

CLI Argument Reference:

    --prompt: The raw text query string (Required if --file is absent).

    --file: File path containing target content to process (Required if --prompt is absent).

    --provider: Model vendor router tier (openai, anthropic).

    --model: Target string Identifier corresponding to the native vendor catalog.

🛠️ Extending the Project (Adding New Model Wrappers)

Follow this 3-step sequence to introduce a brand-new model provider layer.
Step 1: Subclass the Base Provider

Create a new driver file under src/providers/ (e.g., cohere_provider.py) and implement the abstract LLMBaseWrapper.
Python

from src.providers.base import LLMBaseWrapper

class CohereNativeWrapper(LLMBaseWrapper):
    def initialize_client(self) -> None:
        # Initialize vendor SDK client using environment variables
        pass

    def generate(self, prompt: str, model_id: str, **kwargs) -> str:
        # Execute query against vendor endpoint and return raw text string
        return "Response from Cohere"

Step 2: Register with the Global Factory

Open src/factory.py and import your newly created driver class, then register it to the factory router layout mapping.
Python

from src.providers.cohere_provider import CohereNativeWrapper

# Inside factory registration map or initialization loop:
LLMModelFactory.register_provider("cohere", CohereNativeWrapper)

Step 3: Run Validation & Commit

Verify integration locally using the interactive CLI flag layout:
Bash

python src/main.py --prompt "Testing new driver integration" --provider cohere --model command-r
