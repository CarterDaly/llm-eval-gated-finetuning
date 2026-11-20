# llm-eval-gated-finetuning

**Eval-gated continuous fine-tuning for small LLMs.** Production feedback becomes training data, and a candidate model is promoted only when a calibrated evaluation suite shows it is no worse than the current champion.

> **Status: early.** The service registers and serves both classical (pickled scikit-learn style) models and LLMs (a base model plus an optional LoRA adapter, served by vLLM), and logs every request. The evaluation, dataset and training parts of the loop are not built yet; see [Roadmap](#roadmap).

## The loop (target design)

```
1 Serve + log → 2 Score + feedback → 3 Curate dataset → 4 Train candidate
      ↑                                                        ↓
7 Promote / rollback ← 6 Shadow → canary ← (pass) ← 5 Offline eval gate
      ↑                                                        ↓ (fail)
      └──────────────────── 8 Monitor + trigger ←── error analysis
```

1. **Serve + log.** Each request records prompt, output, model/adapter version, latency, cost, and a prompt embedding.
2. **Score + feedback.** Programmatic verifiers on every output, a calibrated LLM judge on a sample, and human feedback (accept/reject, corrections, preferences).
3. **Curate.** Versioned SFT/DPO datasets built from feedback, deduplicated and decontaminated against the held-out eval sets.
4. **Train.** LoRA SFT / DPO (later GRPO with verifier rewards) on a small open model; every run is tied to its dataset hash in MLflow.
5. **Gate.** Paired comparison against the champion with bootstrap confidence intervals, per-slice regression checks, and latency/cost budgets.
6. **Shadow → canary.** Mirrored traffic first, then a small live share, with automatic rollback.
7. **Promote.** The champion only moves forward; the previous one is kept for rollback.
8. **Monitor.** Embedding drift, verifier pass-rate drops, or enough new feedback trigger the next iteration.

## What exists today

| Component | Where |
|---|---|
| Model registry, each model linked to its MLflow run | `POST /models`, `GET /models` |
| LLM registry: base model pinned to a Hugging Face commit, optional LoRA adapter (URI + SHA-256), lineage to the parent model | `POST /models` (JSON) |
| LLM generation through vLLM, each request logged with messages, tools, decoding params, output, tokens, latency and estimated cost | `POST /generations/model/<name>/version/<version>`, `GET /generations/<id>` |
| Prompt embeddings for every LLM request, computed in the background (for drift monitoring and failure clustering) | Celery task, `BAAI/bge-small-en-v1.5` via fastembed on CPU |
| Prediction serving and logging (with SHAP values and a UMAP embedding) | `POST /predictions/model/<name>/version/<version>`, `GET /predictions/<id>` |
| Feedback capture (ground truth, single and batch) | `PATCH /predictions/<id>/actual`, `PATCH /predictions/actuals` |
| Plain-language explanation of a prediction by a local LLM (Ollama) | `GET /predictions/<id>/explainer` |
| Async predictions | Celery + Redis, `/tasks/...` |
| Monitoring | Prometheus `/metrics`; optional Arize |
| Schema migrations | Flask-Migrate / Alembic, `migrations/` |
| API key auth | `X-API-Key` header when `API_KEY` is set |

## Quickstart (Docker)

```bash
cp .env.example .env
docker compose up --build
curl localhost:5000/          # {"status": "ok"}
```

This starts the API on :5000, the MLflow UI on :5001, Postgres on :5432 and Redis on :6379, plus a Celery worker. The API container applies database migrations before it starts serving.

## Local development

Requires Python 3.12.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env                 # SQLite database, read by the flask CLI
docker compose up -d mlflow redis    # tracking server and broker
flask db upgrade                     # create the schema
flask run                            # API on :5000
celery -A gated_finetune.worker worker --pool=solo --loglevel=info   # /tasks and prompt embeddings
```

Checks (the same ones CI runs):

```bash
flake8
pytest            # tests never touch the network; external services are mocked
```

**Why `--pool=solo` on the Mac:** Celery's default prefork pool fails every task on macOS (`ValueError: not enough values to unpack`), because macOS starts pool processes with `spawn` and Celery only sets up its per-process state for `fork`. The solo pool runs tasks one at a time in the worker process, which is fine for development. The Docker worker runs on Linux, which forks, and keeps the prefork pool.

**Changing the schema:** edit the models in `gated_finetune/services/database/`, then run `flask db migrate -m "describe the change"`, review the generated file in `migrations/versions/`, and commit it. `tests/test_migrations.py` fails if the models and migrations disagree.

**Changing dependencies:** edit `requirements.in` or `requirements-dev.in`, then regenerate the pinned lockfiles with the commands at the top of `requirements.in`.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Health check (always public) |
| GET | `/metrics` | Prometheus metrics (always public) |
| POST | `/models` | Register a classical model: multipart `file` (pickle), `name`, `version`, `features`, `mlflow_run_id`, optional `type` |
| POST | `/models` | Register an LLM: JSON `{"kind": "llm", "name", "version", "base_model", "base_model_revision"}`, optional `adapter_uri` + `adapter_sha256`, `mlflow_run_id`, `parent_model_id`, `generation_config` |
| GET | `/models` | List registered models |
| POST | `/predictions/model/<name>/version/<version>` | Predict; body `{"features": {...}}`, optional `tags`, `actual`, `threshold`, `umap` |
| GET | `/predictions/<id>` | Fetch a stored prediction |
| GET | `/predictions/<id>/explainer` | LLM summary of the prediction and its nearest neighbours (needs Ollama) |
| PATCH | `/predictions/<id>/actual` | Record the ground truth for one prediction |
| PATCH | `/predictions/actuals` | Record ground truth in bulk: `{"updates": [{"prediction_id", "actual"}]}` |
| POST | `/generations/model/<name>/version/<version>` | Generate; body `{"messages": [...]}`, optional `tools`, `params`, `source` (`live`/`eval`/`replay`), `example_id`, `tags`. Backend errors return 502 and are still logged |
| GET | `/generations/<id>` | Fetch a stored generation, including its messages and tools |
| POST | `/tasks/predictions/model/<name>/version/<version>` | Queue a prediction; `?task_type=short\|long`, `?wait=true` to block |
| GET | `/tasks/<task_id>` | Status and result of a queued prediction |

**Registering an LLM, an example:**

```bash
curl -X POST localhost:5000/models -H 'Content-Type: application/json' -d '{
  "kind": "llm", "name": "fc-bot", "version": "v1",
  "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
  "base_model_revision": "<40-character commit hash from the Hugging Face repo>"
}'
```

The revision must be a commit hash, not a branch, so a model version always means the same weights and chat template. A fine-tuned version adds `adapter_uri`, `adapter_sha256` and `parent_model_id`; the parent must use the same base model and revision. Decoding defaults to greedy (`temperature` 0), so evaluation runs repeat.

**Serving LLMs with vLLM.** vLLM runs on a Linux GPU machine (on Windows, inside WSL2), separately from this service. Start it with the same base model and revision you registered, for example `vllm serve Qwen/Qwen2.5-1.5B-Instruct --revision <hash>`, and point `VLLM_BASE_URL` at it. A model with an adapter is requested from vLLM under its model id (e.g. `model-fc-bot-v2`), so load the adapter under that name (`--enable-lora --lora-modules model-fc-bot-v2=<path>`). Don't pass `--chat-template`: vLLM must use the base model's own template, the one training uses. Tools are sent with `tool_choice: "none"`, so the raw output, `<tool_call>` tags included, is stored and parsed by this service's code checks rather than by vLLM.

**Prompt embeddings.** After a generation is stored, a Celery task embeds the request's user messages (system prompts are the same for every request and would only add noise) and saves the 384-dimension vector with the model's name. The model is fixed in `prompt_embedding_service.py`, because vectors from different models can't be compared. fastembed runs a quantized ONNX copy of the model on the CPU and downloads it (about 70 MB) on first use into `FASTEMBED_CACHE_PATH`. If the broker is down the generation still succeeds; its `prompt_embedding` stays empty.

**Security note:** `POST /models` accepts pickle files, and unpickling runs code from the file. Only register models you trust, and set `API_KEY` whenever the API is reachable by anyone else.

## Configuration

All settings are environment variables; `.env.example` lists them with defaults.

| Variable | Purpose |
|---|---|
| `FLASK_APP` | `gated_finetune.app`, for the `flask` CLI (set in the Docker image) |
| `DATABASE_URL` | SQLAlchemy URL (default `sqlite:///data.db`) |
| `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND` | Redis URLs for async tasks |
| `MLFLOW_TRACKING_URI` | MLflow tracking server; required for predictions (no silent local fallback) |
| `MODELS_DIR` | Where uploaded model pickles are stored (default `models`) |
| `API_KEY` | Optional; when set, every request except `/` and `/metrics` needs a matching `X-API-Key` |
| `ARIZE_ENABLED`, `ARIZE_API_KEY`, `ARIZE_SPACE_ID`, `ARIZE_ENVIRONMENT` | Optional Arize logging, off by default |
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | Local LLM for the explainer endpoint |
| `LLM_BACKEND` | Serving backend for `/generations`: `vllm` (default) |
| `VLLM_BASE_URL`, `VLLM_API_KEY`, `VLLM_TIMEOUT` | vLLM server URL (default `http://localhost:8000`), its `--api-key` if set, request timeout in seconds (default 60) |
| `FASTEMBED_CACHE_PATH` | Where the prompt embedding model is cached (default: the system temp directory; set and volume-mounted in Docker) |
| `LLM_COST_PER_1K_TOKENS` | Optional USD rate used to estimate `cost_usd` per generation; unset leaves it empty |
| `GUNICORN_BIND`, `GUNICORN_WORKERS` | Optional overrides for `gunicorn.conf.py` |

## Roadmap

- [ ] LLM model type: base models + LoRA adapters, vLLM serving and prompt embeddings (done); local transformers backend (next)
- [ ] Eval service: golden + vault sets, verifiers, judge calibration (Cohen's κ), bootstrap gate
- [ ] Dataset builder: feedback → SFT/DPO, dedup, decontamination, MLflow lineage
- [ ] Training jobs: TRL LoRA SFT → DPO on rented GPUs, adapters registered as candidates
- [ ] Traffic replay simulator with staged distribution shift
- [ ] Shadow / canary routing with automatic rollback
- [ ] GRPO with verifier rewards

## Origin

Started as a course project (DATA 5580) and is being extended into a standalone platform. Course history is preserved in the commit log.

## License

[MIT](LICENSE)
