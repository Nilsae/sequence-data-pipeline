import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import psycopg2
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

load_dotenv()


@dataclass
class Config:
    policy: str = "epsilon_heuristic"
    K: int = 50                       # number of early steps to use
    test_size: float = 0.2
    random_state: int = 42
    min_steps: int = 50               # require at least K steps


def get_conn():
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        dbname=os.getenv("PGDATABASE", "sequence_pipeline"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD") or None,
    )


def fetch_episode_features(cur, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build one feature vector per session using the first K steps:
      features = [pos_0, vel_0, pos_1, vel_1, ..., pos_{K-1}, vel_{K-1}]
      label = sessions.success

    This is intentionally simple and fast.
    """
    cur.execute(
        """
        SELECT s.session_id, s.success
        FROM sessions s
        WHERE s.policy = %s AND s.length_steps >= %s
        ORDER BY s.session_id;
        """,
        (cfg.policy, cfg.min_steps),
    )
    sessions = cur.fetchall()  # [(session_id, success), ...]

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for session_id, success in sessions:
        cur.execute(
            """
            SELECT t, position, velocity
            FROM events
            WHERE session_id = %s AND t < %s
            ORDER BY t;
            """,
            (session_id, cfg.K),
        )
        rows = cur.fetchall()

        # Safety: ensure we got exactly K rows
        if len(rows) != cfg.K:
            continue

        feat = np.empty(cfg.K * 2, dtype=np.float32)
        for i, (_t, pos, vel) in enumerate(rows):
            feat[2 * i] = float(pos)
            feat[2 * i + 1] = float(vel)

        X_list.append(feat)
        y_list.append(1 if success else 0)

    X = np.vstack(X_list) if X_list else np.zeros((0, cfg.K * 2), dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def main():
    cfg = Config()

    with get_conn() as conn:
        with conn.cursor() as cur:
            X, y = fetch_episode_features(cur, cfg)

    if X.shape[0] == 0:
        raise RuntimeError("No training data found. Check policy name and min_steps/K.")

    print(f"Loaded dataset: X={X.shape}, y={y.shape}")
    print(f"Positive (success=1): {int(y.sum())} / {len(y)} ({y.mean():.3f})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    # StandardScaler + LogisticRegression baseline
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)

    print(f"\nBaseline results (K={cfg.K}, policy={cfg.policy})")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC:  {auc:.4f}\n")
    print(classification_report(y_test, pred, digits=4))

    # Optional: save model
    os.makedirs("artifacts", exist_ok=True)
    try:
        import joblib
        joblib.dump(model, "artifacts/logreg_baseline.joblib")
        print("Saved model to artifacts/logreg_baseline.joblib")
    except Exception:
        print("Tip: pip install joblib to save the model artifact.")


if __name__ == "__main__":
    main()
