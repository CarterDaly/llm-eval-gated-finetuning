# llm-eval-gated-finetuning

**Eval-gated continuous fine-tuning for small LLMs.** Production feedback becomes training data, and a candidate model is promoted only when a calibrated evaluation suite shows it is no worse than the current champion.

> **Status: foundation only.** The serving, logging, feedback-capture and monitoring service below works today, currently for classical (pickled scikit-learn style) models. The LLM fine-tuning loop is not built yet; see [Roadmap](#roadmap).

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
celery -A gated_finetune.worker worker --loglevel=info   # only for /tasks
```

Checks (the same ones CI runs):

```bash
flake8
pytest            # tests never touch the network; external services are mocked
```

**Changing the schema:** edit the models in `gated_finetune/services/database/`, then run `flask db migrate -m "describe the change"`, review the generated file in `migrations/versions/`, and commit it. `tests/test_migrations.py` fails if the models and migrations disagree.

**Changing dependencies:** edit `requirements.in` or `requirements-dev.in`, then regenerate the pinned lockfiles with the commands at the top of `requirements.in`.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Health check (always public) |
| GET | `/metrics` | Prometheus metrics (always public) |
| POST | `/models` | Register a model: multipart `file` (pickle), `name`, `version`, `features`, `mlflow_run_id`, optional `type` |
| GET | `/models` | List registered models |
| POST | `/predictions/model/<name>/version/<version>` | Predict; body `{"features": {...}}`, optional `tags`, `actual`, `threshold`, `umap` |
| GET | `/predictions/<id>` | Fetch a stored prediction |
| GET | `/predictions/<id>/explainer` | LLM summary of the prediction and its nearest neighbours (needs Ollama) |
| PATCH | `/predictions/<id>/actual` | Record the ground truth for one prediction |
| PATCH | `/predictions/actuals` | Record ground truth in bulk: `{"updates": [{"prediction_id", "actual"}]}` |
| POST | `/tasks/predictions/model/<name>/version/<version>` | Queue a prediction; `?task_type=short\|long`, `?wait=true` to block |
| GET | `/tasks/<task_id>` | Status and result of a queued prediction |

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
| `GUNICORN_BIND`, `GUNICORN_WORKERS` | Optional overrides for `gunicorn.conf.py` |

## Roadmap

- [ ] LLM model type: base models + LoRA adapters, vLLM serving (Ollama / transformers for local dev)
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
