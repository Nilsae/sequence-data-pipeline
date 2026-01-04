# MountainCar-v0 data generation and processing. 




## Goal

The goal of this project is to demonstrate an end-to-end ML data pipeline:
- generating sequential telemetry data from a simulation environment
- storing it in a relational database with integrity constraints
- validating data quality using SQL
- extracting features and training a baseline ML model to predict episode outcomes early

The project is intentionally scoped to be simple, reproducible, and extensible.

## Data model overview

The dataset is stored in two relational tables:

- `sessions`: one row per episode (run)
  - environment ID, policy type, random seed
  - success label, episode length, total return

- `events`: one row per timestep within a session
  - state variables (position, velocity)
  - action taken and reward
  - termination flags

Each `session` has many ordered `events` linked by `session_id`.


## Action space (MountainCar-v0)

| Action code | Meaning        | Effect on the car |
| ----------- | -------------- | ----------------- |
| `0`         | Push **left**  | Accelerate left   |
| `1`         | **No push**    | Coast             |
| `2`         | Push **right** | Accelerate right  |

### Success label definition

A session is labeled as `success = true` if the environment terminates
because the goal condition is reached (`terminated = true`).

Time-limit terminations (`truncated = true`) are treated as failures.


# Project Setup and Commands Summary

This document summarizes the key commands used so far and the essential details behind them.

---

## Environment setup (Python)

Create and activate a local virtual environment inside the repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
which python
python --version
```

Install pinned dependencies and verify imports:

```bash
pip install --upgrade pip
pip install -r requirements.txt
python -c "import gymnasium, numpy, psycopg2; print('ok')"
python -c "import gymnasium; print(gymnasium.__version__)"
```

Freeze exact versions for reproducibility:

```bash
pip freeze > requirements.lock.txt
python --version > python-version.txt
```

**Essential details**

* `.venv/` and `.env` are excluded from version control via `.gitignore`
* `requirements.txt` defines intended dependencies
* `requirements.lock.txt` captures the exact environment snapshot

---

## PostgreSQL setup (macOS / Homebrew)

Start the PostgreSQL service:

```bash
brew services start postgresql@16
```

Verify `psql` availability:

```bash
which psql
psql --version
```

Check the active user and database:

```bash
psql -d postgres -c "SELECT current_user, current_database();"
```

Create the project database:

```bash
createdb sequence_pipeline
psql -d postgres -c "\\l"
```

Enter and exit interactive `psql`:

```bash
psql -d sequence_pipeline
\\q
```

---

## Database schema and seed data

Apply the schema (creates `sessions` and `events` tables and indexes):

```bash
psql -d sequence_pipeline -f sql/schema.sql
```

Insert a small seed dataset for smoke testing:

```bash
psql -d sequence_pipeline -f sql/seed.sql
```

Verify contents:

```bash
psql -d sequence_pipeline -c "SELECT COUNT(*) FROM sessions;"
psql -d sequence_pipeline -c "SELECT COUNT(*) FROM events;"
psql -d sequence_pipeline -c "SELECT session_id, success, length_steps, return_sum FROM sessions ORDER BY session_id;"
psql -d sequence_pipeline -c "SELECT session_id, t, position, velocity, action, terminated, truncated FROM events ORDER BY session_id, t;"
```

---

## Data validation (SQL)

Run data quality checks:

```bash
psql -d sequence_pipeline -f sql/validation.sql
```

**Validation checks include**

* duplicate `(session_id, t)` rows
* mismatches between `sessions.length_steps` and actual event counts
* missing timestep `t = 0` or timestep gaps
* exactly one terminal row per session
* basic numeric sanity checks for position and velocity

---

## Real data generation (Gymnasium → PostgreSQL)

Generate real MountainCar-v0 episodes and log telemetry:

```bash
python pipeline/generate_data.py
```

Generate epsilon-noisy heuristic episodes:

```bash
EPS=0.40 python pipeline/generate_data.py
```

This produces:

* episode-level rows in `sessions`
* step-level telemetry rows in `events`

---

## Inspect label distribution

Check success vs failure counts by policy:

```bash
psql -d sequence_pipeline -c "
SELECT policy, success, COUNT(*)
FROM sessions
GROUP BY policy, success
ORDER BY policy, success;
"
```

This confirmed that the `epsilon_heuristic` policy yields both successes and failures, making the dataset suitable for supervised ML.

---

## PostgreSQL sequence reset (important troubleshooting)

If manual inserts desynchronize the auto-increment sequence, you may see:

```
duplicate key value violates unique constraint "sessions_pkey"
```

Fix by resetting the sequence:

```bash
psql -d sequence_pipeline -c "
SELECT setval(
  pg_get_serial_sequence('sessions', 'session_id'),
  COALESCE((SELECT MAX(session_id) FROM sessions), 1)
);
"
```
## Baseline ML experiment

As a first baseline, a logistic regression classifier is trained to predict
episode success using only early telemetry.

- Input: first 50 timesteps of `(position, velocity)`
- Label: `sessions.success`
- Model: Logistic Regression with standardization
- Metric: ROC AUC, accuracy

This demonstrates that early dynamics contain strong predictive signals,
and provides a reference point for evaluating more expressive models.


Baseline results (K=50):
- ROC AUC: ~0.88
- Accuracy: ~0.82

Linear baselines were later compared against nonlinear models to assess representational limits.

## Feature engineering experiments

To evaluate how additional behavioral information affects early outcome prediction,
multiple feature representations were tested using the first K = 50 timesteps.

### Feature sets evaluated

1. **State-only (baseline)**
   - `(position, velocity)`
   - 2 × K features

2. **State + action + reward**
   - `(position, velocity, action, reward)`
   - 4 × K features

3. **State + dynamics + action (one-hot)**
   - `(position, velocity)`
   - first-order differences `(Δposition, Δvelocity)`
   - action encoded as one-hot `(left, none, right)`
   - 7 × K features

### Findings

- Adding raw `action` and `reward` did **not** improve performance for a linear model
- Action encoded as ordinal values degraded performance
- Derived dynamics (`Δposition`, `Δvelocity`) were more informative than raw rewards
- Overall, a linear classifier saturated quickly, suggesting nonlinear structure in the data

These results motivated evaluating a nonlinear baseline model.

## Nonlinear baseline: Random Forest

To test whether the predictive signal is nonlinear, a RandomForest classifier was trained
using the enriched feature representation (state, dynamics, one-hot actions).

### Model
- RandomForestClassifier
- 400 trees
- class-weighted to address label imbalance
- trained on early trajectory windows (K = 50)

### Results (K = 50)

- ROC AUC: **~0.94**
- Accuracy: **~0.91**
- Failure (negative class) precision improved substantially

This confirms that early trajectory dynamics contain **strong nonlinear structure**
that is not captured by linear models.

### Key insight

> Early success/failure prediction in MountainCar is best modeled with nonlinear decision boundaries,
even when using only the first few dozen timesteps.


## Model comparison summary (K = 50)

| Model              | Features                          | ROC AUC | Accuracy |
|--------------------|-----------------------------------|---------|----------|
| Logistic Regression| position, velocity                | ~0.88   | ~0.82    |
| Logistic Regression| state + action + dynamics         | ~0.84   | ~0.79    |
| Random Forest      | state + dynamics + one-hot action | ~0.94   | ~0.91    |



**Key insight**

* PostgreSQL auto-increment uses a separate sequence object
* Manually inserting primary keys does not advance the sequence automatically


## Possible extensions

- Add richer features (actions, rewards, windowed statistics)
- Train sequence models (MLP, 1D CNN, RNN)
- Predict time-to-failure instead of binary success
- Replace MountainCar with other Gymnasium environments
- Run large-scale generation on a compute cluster

---

## Project summary

This project sets up a reproducible Python environment and a local PostgreSQL database, defines a relational schema for episode-level (`sessions`) and step-level (`events`) telemetry, and adds SQL validation checks for data integrity. A Gymnasium-based data generator runs MountainCar-v0 episodes under different policies and logs telemetry into Postgres. The resulting dataset is validated and inspected via SQL to produce a clean, labeled dataset suitable for downstream machine learning.
