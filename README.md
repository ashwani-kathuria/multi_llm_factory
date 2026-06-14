# Multi-LLM Provider Factory

A production-grade, asynchronous Python unified interface layer built to orchestrate
communication across multiple LLM provider native APIs (OpenAI, Google GenAI, and AWS
Bedrock).

This system completely avoids the heavy dependency bloat and rigid abstractions of large
orchestration frameworks by leveraging pure object-oriented design patterns. By using the
**Strategy** and **Factory** patterns, the application core remains entirely decoupled
from underlying vendor SDK changes, providing absolute structural autonomy, raw API
performance, and clean debugging stacks.

---

## 🚀 Core Features

- **Design-Pattern Driven:** Uses the Strategy pattern for provider isolation and the
  Factory pattern for clean runtime lifecycle routing.
- **Dual Runtime Support:** Natively supports both high-throughput asynchronous
  (`agenerate`) and standard synchronous (`generate`) execution pipelines.
- **Production Resiliency:** Built-in exponential backoff retry mechanisms via the
  `tenacity` engine to handle rate limits and transient network dropouts.
- **Type-Safe Telemetry:** Leverages **Pydantic V2** to enforce strict input validation,
  handle structured JSON output, and provide per-execution metrics (tokens, latency).
- **Reasoning Trace Analysis:** Integrated `proximity_metric` module performs
  Topic-Aware Self-Correction Proximity analysis on chain-of-thought reasoning traces,
  quantifying how much uncertainty was resolved through self-correction.
- **Highly Extensible:** Pluggable runtime registry — add custom drivers (e.g., local
  Ollama or specialised endpoints) without altering library files.

---

## 📂 Project Structure

```text
multi_llm_factory/
│
├── core/                        # Core engine source module
│   ├── __init__.py              # Factory pattern engine & runtime registry
│   ├── config.py                # Pydantic V2 config & telemetry schemas
│   └── wrappers.py              # Provider Strategy implementation drivers
│
├── proximity_metric.py          # Topic-Aware Self-Correction Proximity module
├── main.py                      # CLI orchestration pipeline entry-point
├── requirements.txt             # Third-party platform dependencies
└── .env                         # Local credentials (git-ignored)
```

---

## ⚙️ Setup & Installation

### 1. Clone & Navigate

```bash
git clone <your-repo-url>
cd multi_llm_factory
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the spaCy Language Model

Required once for the proximity metric's subject extraction:

```bash
python -m spacy download en_core_web_sm
```

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```env
# Provider API Keys
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here

# AWS credentials (if using Bedrock)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1

# Optional: override the embedding model used by proximity_metric
# PROXIMITY_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

---

## 💻 CLI Usage

### Basic Usage

```bash
python main.py -p <provider> -m <model-id> [options]
```

### Run with a Prompt File

```bash
python main.py -p google -m gemini-2.5-flash -f prompt.txt
```

### Run with Default Prompt

```bash
python main.py -p openai -m gpt-4o
```

### Full CLI Reference

| Flag | Long Form | Default | Description |
|------|-----------|---------|-------------|
| `-p` | `--provider` | *(required)* | Provider: `openai`, `google`, `aws_bedrock` |
| `-m` | `--model` | *(required)* | Model ID e.g. `gpt-4o`, `gemini-2.5-flash` |
| `-t` | `--temperature` | `0.0` | Sampling temperature (0.0 – 2.0) |
| `-f` | `--prompt-file` | `None` | Path to a `.txt` file containing the prompt |
| `-o` | `--output-file` | `output.txt` | Path to save the model's text response |
| `-k` | `--max-tokens` | `16384` | Maximum output token budget |

---

## 📊 Proximity Metric — Uncertainty Quantification

After each LLM call the reasoning trace (chain-of-thought) is automatically analysed by
`proximity_metric.py` using the **Topic-Aware Self-Correction Proximity** algorithm.

### What It Does

Modern reasoning models frequently express uncertainty and then resolve it:

> *"Wait, the equation might be wrong. Let me recalculate. The correct answer is 4."*

Naïve hedge-word counting would penalise this trace even though the uncertainty was
immediately resolved. The proximity metric detects this pattern and reduces the
uncertainty weight of resolved hedges proportionally to match quality.

### 8-Step Pipeline

| Step | Description |
|------|-------------|
| 1 | **Hedge detection** — find sentences with uncertainty markers (`maybe`, `might`, `not sure`, `seems`, …) |
| 2 | **Verification detection** — find sentences with resolution markers (`verify`, `confirm`, `therefore`, …) |
| 3 | **Subject extraction** — spaCy dependency parsing extracts the uncertainty *target* (e.g. `boundary condition`) |
| 4 | **Embedding similarity** — SentenceTransformer cosine similarity between hedge & verification subjects |
| 5 | **Proximity score** — `exp(−gap / 50)`, rewards verifications that follow quickly |
| 6 | **Match score** — `0.7 × similarity + 0.3 × proximity` |
| 7 | **Resolution** — best-match verification per hedge; resolved if `match_score ≥ 0.60` |
| 8 | **Weight adjustment** — resolved: `eff_weight = 1 − 0.8 × match_score`; unresolved: `1.0` |

### Public API

```python
from proximity_metric import calculate_proximity_metrics

result = calculate_proximity_metrics(
    reasoning_text,
    embedding_model="all-MiniLM-L6-v2",   # configurable
    similarity_threshold=0.60,
    proximity_decay=50.0,
    match_weights=(0.7, 0.3),
    resolution_weight_reduction=0.8,
)
```

### Output Schema

```json
{
  "total_hedges": 3,
  "resolved_hedges": 1,
  "unresolved_hedges": 2,
  "trur": 0.3333,
  "weighted_trur": 0.2651,
  "matches": [
    {
      "id": "H1",
      "subject": "the equation",
      "position": 0,
      "matched_verification": "V1",
      "subject_similarity": 1.0,
      "proximity_score": 0.9802,
      "match_score": 0.9941,
      "resolved": true,
      "effective_weight": 0.2047
    },
    {
      "id": "H2",
      "subject": "the boundary condition",
      "position": 3,
      "matched_verification": null,
      "resolved": false,
      "effective_weight": 1.0
    }
  ]
}
```

| Field | Description |
|-------|-------------|
| `trur` | Topic-Resolved Uncertainty Ratio — `resolved / total` |
| `weighted_trur` | Mean weight reduction across all hedges |
| `effective_weight` | Per-hedge uncertainty contribution after resolution adjustment |

### Supported Embedding Models

Set `PROXIMITY_EMBEDDING_MODEL` in `.env` or pass `embedding_model=` directly:

```env
PROXIMITY_EMBEDDING_MODEL=all-mpnet-base-v2
```

| Model | Speed | Quality |
|-------|-------|---------|
| `all-MiniLM-L6-v2` *(default)* | ⚡ Fast | Good |
| `all-mpnet-base-v2` | Medium | High |
| `multi-qa-MiniLM-L6-cos-v1` | ⚡ Fast | Good (QA-tuned) |
| `bge-small-en-v1.5` | ⚡ Fast | Good |
| `bge-base-en-v1.5` | Medium | High |
| `e5-small-v2` | ⚡ Fast | Good |
| `e5-base-v2` | Medium | High |

### Standalone Smoke Test

```bash
python proximity_metric.py
```

Runs the spec example trace and prints full JSON output. Model weights are cached after
the first run — subsequent calls complete in under one second.

---

## 🛠️ Extending the Project

### Adding a New LLM Provider

**Step 1 — Subclass `BaseLLMWrapper`** in `core/wrappers.py` (or a new file):

```python
from core.config import LLMModelConfig, LLMUnifiedResponse
from core.wrappers import BaseLLMWrapper

class CohereWrapper(BaseLLMWrapper):
    def _initialize_client(self):
        import cohere
        return cohere.Client(api_key=self.config.api_key.get_secret_value())

    def generate(self, prompt, system_instruction=None, response_schema=None):
        # ... synchronous implementation
        pass

    async def agenerate(self, prompt, system_instruction=None, response_schema=None):
        # ... async implementation
        pass
```

**Step 2 — Register with the Factory** (in `core/__init__.py` or at startup):

```python
from core import LLMModelFactory
from my_providers.cohere_wrapper import CohereWrapper

LLMModelFactory.register_provider("cohere", CohereWrapper)
```

**Step 3 — Run & Validate:**

```bash
python main.py -p cohere -m command-r -f prompt.txt
```

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `pydantic>=2.7.0` | Config validation & structured output schemas |
| `tenacity>=8.3.0` | Exponential backoff retry logic |
| `openai>=1.30.0` | OpenAI SDK |
| `google-genai>=0.1.1` | Google GenAI SDK |
| `boto3>=1.34.120` | AWS Bedrock SDK |
| `python-dotenv>=1.0.1` | `.env` credential loading |
| `httpx>=0.27.0` | Async HTTP transport |
| `sentence-transformers>=2.7.0` | Embedding-based semantic similarity (proximity metric) |
| `spacy>=3.7.0` | NLP subject extraction via dependency parsing (proximity metric) |
