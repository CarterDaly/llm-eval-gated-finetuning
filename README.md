# llm-eval-gated-finetuning

**Eval-gated continuous fine-tuning for small LLMs.** Production feedback becomes training data, and a candidate model is promoted only when a calibrated evaluation suite shows it is no worse than the current champion.

> **Status:** the serving, logging, feedback, and monitoring foundation is in place (below). The LLM fine-tuning loop is under construction; see [Roadmap](#roadmap).

## The loop

```
1 Serve + log → 2 Score + feedback → 3 Curate dataset → 4 Train candidate
      ↑                                                        ↓
7 Promote / rollback ← 6 Shadow → canary ← (pass) ← 5 Offline eval gate
      ↑                                                        ↓ (fail)
      └──────────────────── 8 Monitor + trigger ←── error analysis
```

1. **Serve + log.** Every request records prompt, output, model/adapter version, latency, cost, and a prompt embedding.
2. **Score + feedback.** Programmatic verifiers on every output, a calibrated LLM judge on a sample, and human feedback (accept/reject, corrections, preferences).
3. **Curate.** Build versioned SFT/DPO datasets from feedback, deduplicated and decontaminated against the held-out eval sets.
4. **Train.** LoRA SFT / DPO (later GRPO with verifier rewards) on a small open model; every run is tied to the dataset hash in MLflow.
5. **Gate.** Paired comparison against the champion with bootstrap confidence intervals, per-slice regression checks, and latency/cost budgets.
6. **Shadow → canary.** Mirrored traffic first, then a small live share, with automatic rollback.
7. **Promote.** The champion only moves forward; the bar rises with every promotion.
8. **Monitor.** Embedding drift, verifier pass-rate drops, or enough new feedback trigger the next iteration.

## What exists today

| Component | Where |
|---|---|
| Model registry with MLflow run linkage | `POST/GET /models` |
| Prediction serving and logging | `POST /predictions/model/<name>/version/<version>`, `GET /predictions/<id>` |
| Feedback capture (ground truth, single and batch) | `PATCH /predictions/<id>/actual`, `PATCH /predictions/actuals` |
| Explanations (SHAP) and embeddings (UMAP) | `GET /predictions/<id>/explainer`, `gated_finetune/services/` |
| Async jobs | Celery + Redis (`/tasks/...`) |
| Monitoring | Prometheus (`/metrics`), Arize |
| LLM explanation service | `gated_finetune/services/llm_service.py` (Ollama) |
| API key auth | `X-API-Key` header when `API_KEY` is set |

## Quickstart

```bash
cp .env.example .env          # or create .env with the variables below
docker compose up --build     # API :5000, MLflow :5001, Postgres :5432, Redis :6379
```

Local development:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Environment variables: `DATABASE_URL`, `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`, `MLFLOW_TRACKING_URI`, `API_KEY` (optional), `ARIZE_API_KEY` / `ARIZE_SPACE_KEY` / `ARIZE_ENABLED` (optional), `OLLAMA_BASE_URL` / `OLLAMA_MODEL` (optional).

MLflow runs as a tracking server with proxied artifacts, so runs are not tied to any local path.

## Roadmap

- [ ] LLM model type: vLLM serving of base models + LoRA adapters (Ollama for local dev)
- [ ] Eval service: golden + vault sets, verifiers, judge calibration (Cohen's κ), bootstrap gate
- [ ] Dataset builder: feedback → SFT/DPO, dedup, decontamination, MLflow lineage
- [ ] Training jobs: TRL LoRA SFT → DPO on rented GPUs, adapters registered as candidates
- [ ] Traffic replay simulator with staged distribution shift
- [ ] Shadow / canary routing with automatic rollback
- [ ] GRPO with verifier rewards

## Origin

Started as a course project (DATA 5580) and is being extended into a standalone platform. Course history is preserved in the commit log.
