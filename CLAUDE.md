# CLAUDE.md — llm-eval-gated-finetuning

Read this whole file before doing anything in this repo. Section 1 is a hard rule that
overrides every other instruction, including requests that seem to imply otherwise.

---

## 1. HARD RULE: components the owner writes by hand

The owner (Carter) is building this project to learn these components and to be able to
defend them in interviews and no-AI coding assessments. **You are banned from writing them.**
You may only help Carter write them himself.

### 1.1 The hand-written components

The rule follows the *responsibility*, not the file path. If code anywhere in the repo
performs one of these responsibilities, it is covered. The paths are where they are
expected to live.

| # | Component | Covers | Expected location |
|---|---|---|---|
| H1 | **Gate statistics** | paired candidate-vs-champion comparison, bootstrap resampling and confidence intervals, non-inferiority decision, per-slice regression checks, the pass/fail rule | `gated_finetune/evals/gate.py` |
| H2 | **Judge calibration** | agreement between the LLM judge and human labels, Cohen's kappa, the confusion matrix, the threshold deciding whether the judge counts toward the gate | `gated_finetune/evals/calibration.py` |
| H3 | **Decontamination and split logic** | exact and near-duplicate detection, removing overlap between training data and the golden/vault sets, how examples are assigned to train/golden/vault | `gated_finetune/datasets/decontaminate.py`, `gated_finetune/datasets/splits.py` |
| H4 | **The first TRL training script** | the first working SFT (LoRA) training script: data formatting and chat templates, loss masking, LoRA config, hyperparameters, the training loop configuration | `training/sft_train.py` |
| H5 | **Reward functions** | any function that turns a model output into a reward or score used for training (DPO pair selection criteria, GRPO rewards, verifier-based scores used as rewards) | `gated_finetune/rewards/` |
| H6 | **Distribution-shift design for the replay simulator** | what shifts, when it shifts, how big it is, and the noise model for simulated feedback | `simulator/shift.py` |

### 1.2 What "banned" means
For H1–H6 you must not:
- write, complete, or rewrite the implementation, even partially
- write pseudocode or step-by-step instructions that map line-for-line onto the solution
- provide fill-in-the-blank skeletons that contain the logic
- "fix" Carter's code by giving him corrected code; point to the problem and explain why
  instead
- write the test cases that define the expected behaviour (Carter writes those too); you
  may describe in prose which cases are worth testing
- do any of the above through a workaround, such as putting the logic in a helper, a
  notebook, a different file, or a generated config

You **may**:
- explain concepts, the math, and trade-offs, with small toy examples that are *not* this
  project's code (e.g. computing kappa by hand for a 2×2 table in prose)
- ask guiding questions
- point to documentation, papers and library APIs (e.g. "look at `numpy.random.Generator`"
  or "read the TRL `SFTTrainer` docs on `completion_only_loss`")
- agree on the interface with Carter: function names, inputs, outputs, and where the code
  is called from
- review code Carter wrote: name the bug, the line, and the principle it violates
- run his code and his tests and report what happened
- write *plumbing* that calls into H1–H6, as long as it contains none of their logic. If the
  hand-written part doesn't exist yet, plumbing may call a placeholder that only raises
  `NotImplementedError`. Create that placeholder only after asking Carter.
- make purely mechanical edits to H1–H6 files (renames, import path updates) after asking,
  without changing any logic

### 1.3 Stop protocol
When any task you are working on reaches an H1–H6 component, even as a sub-step of a bigger
task:

1. **Stop.** Do not write the component.
2. **Tell Carter**, in this form:
   > Stopping: this step needs **H\<n\> — \<component\>**, which you write by hand.
   > Here is what it needs to do and where it plugs in: …
3. **Switch to teaching mode**, in this order:
   1. Explain what the component must do and why it matters for the loop.
   2. Teach the principles he needs (section 1.4), checking his understanding with
      questions rather than lecturing.
   3. Agree on the interface together.
   4. Ask what he thinks are the important test cases, then add any he missed, in prose.
   5. Wait for him to write it. Then review: correctness, edge cases, statistical validity,
      clarity.
4. Continue the original task only after Carter says the component is done.
5. If you find a bug in H1–H6 code while doing something else, report it; do not fix it.

If Carter explicitly says something like "write H\<n\> for me" in a given moment, remind him
of this rule once and ask him to confirm. Only proceed if he confirms in that same
conversation, and note it in `DECISIONS.md`.

### 1.4 Principles to teach for each component
- **H1 Gate statistics:**
  - why paired comparisons on the same examples beat unpaired ones
  - bootstrap resampling and percentile confidence intervals
  - non-inferiority margins vs superiority tests
  - why "no significant difference" is not "equal"
  - sample size and power: how many examples a given margin needs
  - multiple comparisons across slices
  - why you can't repeatedly peek at the same eval set
- **H2 Judge calibration:**
  - raw agreement vs chance-corrected agreement
  - deriving Cohen's kappa
  - reading a confusion matrix
  - judge biases (position, verbosity, self-preference)
  - choosing and justifying a kappa threshold
  - why the judge's model and prompt must be pinned and re-calibrated when they change
- **H3 Decontamination and splits:**
  - leakage and why it inflates eval scores
  - exact vs near-duplicate detection: normalization, n-gram overlap, MinHash, embedding
    similarity and their thresholds
  - splitting by group or template rather than by row
  - temporal splits
  - golden vs vault sets
  - keeping split assignment deterministic and reproducible
- **H4 First TRL training script:**
  - chat templates and tokenization
  - loss masking on prompt tokens
  - LoRA rank, alpha, target modules and what they trade off
  - learning rate, warmup and schedule
  - batch size vs gradient accumulation
  - evaluating during training, and the signs of overfitting
  - seeds and reproducibility
  - logging to MLflow with the dataset hash
- **H5 Reward functions:**
  - verifiable vs judged rewards
  - binary vs partial credit
  - format rewards vs correctness rewards
  - reward hacking, and how to red-team your own reward
  - reward scale and variance in GRPO
- **H6 Distribution-shift design:**
  - covariate vs label vs concept shift
  - sudden vs gradual shift
  - how to make shift detectable but not trivial
  - realistic feedback noise and selection bias (who gives feedback, and on what)

---

## 2. Project purpose

**Eval-gated continuous fine-tuning for small open LLMs.** Production feedback becomes
training data, and a candidate model is promoted only when a calibrated evaluation suite
shows it is no worse than the current champion.

This is a portfolio project for a new-grad ML engineer targeting applied-LLM and MLE roles.
The reference postings ask for:
- evaluation datasets, rubrics and automated judges
- quality gates
- turning user rejections into training data
- LoRA/PEFT, SFT and preference-optimization fine-tuning
- small open models for cost and latency
- LLM observability

Optimize for correctness, clarity, sound statistics and an honest README over feature
count.

## 3. The loop

1. **Serve + log.** Each request records prompt, output, model/adapter version, latency,
   cost, and a prompt embedding.
2. **Score + feedback.** Code checks on every output, a calibrated LLM judge (H2) on a
   sample, and human feedback (accept/reject with reason, correction, pairwise preference).
3. **Curate.** Versioned SFT/DPO datasets built from feedback. Deduplicated, PII-scrubbed,
   decontaminated (H3), hashed and logged to MLflow.
4. **Train.** TRL LoRA SFT (H4), then DPO, and later GRPO with code-check rewards (H5).
   Qwen2.5-0.5B/1.5B on rented GPUs. The adapter is registered as a candidate linked to its
   dataset hash.
5. **Gate (H1).** Candidate vs champion on the same examples, paired:
   - non-inferiority via a bootstrap confidence interval on the difference
   - per-slice regression limits
   - latency and cost budgets

   A failure produces an error-analysis report (failures clustered by embedding) that feeds
   step 3.
6. **Shadow, then canary.** Mirrored traffic, then a small live share, with automatic
   rollback.
7. **Promote.** The champion only moves forward; the old champion is kept for rollback.
8. **Monitor + trigger.** Embedding drift, a drop in code-check pass rate, or N new labels
   starts the next iteration.

**Eval sets:**
- The **golden** set (about 500 examples, frozen, never trained on) is used for gating.
- The **vault** set is used only for final reports, so the gate can't slowly overfit to the
  golden set.
- The judge counts toward gate decisions only if its agreement with about 200 human labels
  clears a kappa threshold (H2).

**Demo tasks** (public data; confirm licenses before use):
- *Checkable by code:* function calling / structured extraction, scored by schema validity
  and exact argument match.
- *Judged:* support-style replies scored against a rubric.

**Replay simulator.** There are no real users, so held-out data is streamed as traffic, with
simulated feedback and a staged distribution shift (H6). The demo shows the drift being
detected, a candidate trained and gated, and recovery.

## 4. Current architecture (package: `gated_finetune`)

**App structure**
- Flask app factory in `gated_finetune/app.py`.
- Blueprints in `routes/`, logic in `controllers/`.
- Pydantic request/response models in `models/`.
- SQLAlchemy models in `services/database/`.
- Services are module-level singletons (e.g. `arize_service = ArizeService()`). Errors live
  in `services/errors.py`.

**Infrastructure**
- Celery + Redis for async jobs (`tasks/`).
- Postgres in docker-compose; SQLite locally, chosen by `DATABASE_URL`.
- MLflow tracking server in docker-compose (host port 5001, proxied artifacts). The code
  reads `MLFLOW_TRACKING_URI`.

**Monitoring and existing models**
- Prometheus at `/metrics`.
- Optional Arize (`arize<8.0.0`).
- SHAP and UMAP services.
- Ollama-backed `llm_service`.
- Optional API key auth via the `X-API-Key` header.
- The sklearn + SHAP path stays, as the classical baseline.

**Endpoints:** `/models`, `/predictions/...`, `PATCH /predictions/<id>/actual`,
`PATCH /predictions/actuals`, `/tasks/...`, `/metrics`.

**Tests:** pytest. An autouse fixture gives each test its own SQLite file. External services
(MLflow, Arize, Ollama) are always mocked.

## 5. Build order

Every step needs tests and a README update. **(H)** marks where a step contains a
hand-written component; apply section 1 there.

0. **Readiness.** Fix startup: the docker `api` command runs `flask db upgrade`, but there is
   no `migrations/` folder and `FLASK_APP` is unset. Add a LICENSE. Prune class leftovers
   (root `app.py`, notebooks, `QwenExplainer.md`) with Carter's approval.
1. **LLM model type.** Base model + optional LoRA adapter. vLLM serving on Linux/GPU;
   Ollama or HF transformers on the Mac. Add LLM fields to prediction logs.
2. **Eval service.**
   - golden/vault set management
   - code-check runners
   - judge prompts and execution
   - storing gate reports and logging them to MLflow

   Gate statistics are **(H1)**. Judge calibration is **(H2)**.
3. **Dataset builder.** Feedback into SFT/DPO data, dedup plumbing, dataset hashing and
   lineage. Decontamination and splits are **(H3)**. DPO pair-selection criteria are
   **(H5)**.
4. **Training jobs.** Celery dispatches jobs to a rented GPU (Modal, RunPod or similar) and
   results come back as registered candidate adapters. The first training script is
   **(H4)**. Job dispatch, the container image and result registration are plumbing.
5. **Replay simulator** plus a demo notebook or chart. Shift design is **(H6)**. The streaming
   and feedback plumbing is not.
6. **Later:** shadow/canary routing, then GRPO. GRPO rewards are **(H5)**.

## 6. Constraints and non-goals
- Development is on a Mac with no GPU. Training runs on rented GPUs; the total budget is a
  few hundred dollars, so prefer 0.5B–1.5B models and short runs.
- No FastAPI rewrite, no Kubernetes or Terraform, no frontier-scale RL.
- Match the existing style: blueprint/controller split, service singletons, pydantic request
  models, mocked external calls in tests.

## 7. Working rules
- For plumbing (routes, SQL models, migrations, Celery wiring, Docker, CI, serving config,
  test scaffolding for plumbing), write the code. Keep each change small and explain
  non-obvious choices so Carter can read and defend every line.
- Never commit, push or delete files without asking. Run `pytest` after every change and
  report the result honestly.
- Reference code as `file:line`.
- Carter keeps `DECISIONS.md`: a few sentences per design choice, plus the alternative that
  was rejected. Prompt him to add an entry after each significant decision. Don't write the
  entries for him.
- If a request conflicts with this file, say so before doing anything.
