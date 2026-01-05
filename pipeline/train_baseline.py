import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import psycopg2
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score #, classification_report
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline

load_dotenv()


@dataclass
class Config:
    policy: str = "epsilon_heuristic"
    K_list: tuple = (10, 25, 50, 100)
    test_size: float = 0.2
    random_state: int = 42


def get_conn():
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        dbname=os.getenv("PGDATABASE", "sequence_pipeline"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD") or None,
    )


def fetch_episode_features(cur, cfg: Config, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build one feature vector per session using the first K steps:
      features = per-step block of 7 values repeated K times:
        [pos, vel, dpos, dvel, a_left, a_none, a_right] for steps t=0..K-1


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
        (cfg.policy, K),
    )
    sessions = cur.fetchall()  # [(session_id, success), ...]

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for session_id, success in sessions:
        cur.execute(
            """
            SELECT t, position, velocity, action
            FROM events
            WHERE session_id = %s AND t < %s
            ORDER BY t;
            """,
            (session_id, K),
        )
        rows = cur.fetchall()

        # Safety: ensure we got exactly K rows
        if len(rows) != K:
            continue

        feat = np.empty(K * 7, dtype=np.float32)
        prev_pos = None
        prev_vel = None

        for i, (_t, pos, vel, act) in enumerate(rows):
            pos = float(pos)
            vel = float(vel)

            dpos = 0.0 if prev_pos is None else (pos - prev_pos)
            dvel = 0.0 if prev_vel is None else (vel - prev_vel)

            # one-hot for action (0/1/2)
            a0 = 1.0 if act == 0 else 0.0
            a1 = 1.0 if act == 1 else 0.0
            a2 = 1.0 if act == 2 else 0.0

            base = 7 * i
            feat[base + 0] = pos
            feat[base + 1] = vel
            feat[base + 2] = dpos
            feat[base + 3] = dvel
            feat[base + 4] = a0
            feat[base + 5] = a1
            feat[base + 6] = a2

            prev_pos = pos
            prev_vel = vel



        X_list.append(feat)
        y_list.append(1 if success else 0)

    X = np.vstack(X_list) if X_list else np.zeros((0, K * 7), dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def train_eval_rf(X: np.ndarray, y: np.ndarray, cfg: Config) -> Dict[str, float]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=cfg.random_state,
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)

    prob = rf.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)

    # failure metrics (class 0)
    from sklearn.metrics import precision_recall_fscore_support
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, labels=[0, 1])
    fail_prec, fail_rec, fail_f1 = prec[0], rec[0], f1[0]

    return {
        "acc": acc,
        "auc": auc,
        "fail_prec": fail_prec,
        "fail_rec": fail_rec,
        "fail_f1": fail_f1,
        "model": rf,
    }

def main():
    cfg = Config()
    results = []

    with get_conn() as conn:
        with conn.cursor() as cur:
            for K in cfg.K_list:
                X, y = fetch_episode_features(cur, cfg, K)
                if X.shape[0] == 0:
                    print(f"Skipping K={K}: no sessions found with length_steps >= {K}")
                    continue

                print(f"Loaded dataset for K={K}: X={X.shape}, y={y.shape} (pos rate={y.mean():.3f})")

                out = train_eval_rf(X, y, cfg)
                results.append((K, out["auc"], out["acc"], out["fail_prec"], out["fail_rec"], out["fail_f1"], out["model"]))
    
    
    if not results:
        raise RuntimeError("No results produced. Check data and queries.")

   # Print a clean comparison table
    print("\nRandomForest early-prediction sweep")
    print("K\tAUC\tACC\tFailPrec\tFailRec\tFailF1")
    for K, auc, acc, fp, fr, ff1, _ in results:
        print(f"{K}\t{auc:.4f}\t{acc:.4f}\t{fp:.4f}\t\t{fr:.4f}\t\t{ff1:.4f}")

    # Save the best model by AUC
    import json
    import joblib

    os.makedirs("artifacts", exist_ok=True)

    # Save sweep metrics (so the exact run is recorded)
    sweep = [
        {
            "K": K,
            "auc": float(auc),
            "acc": float(acc),
            "fail_prec": float(fp),
            "fail_rec": float(fr),
            "fail_f1": float(ff1),
        }
        for (K, auc, acc, fp, fr, ff1, _model) in results
    ]
    with open("artifacts/rf_sweep_results.json", "w") as f:
        json.dump(sweep, f, indent=2)

    # 1) Save best-by-AUC model
    best = max(results, key=lambda r: r[1])
    best_K, best_auc, best_acc, *_rest, best_model = best
    joblib.dump(best_model, f"artifacts/rf_best_auc_K{best_K}.joblib")
    print(f"\nSaved best-AUC RF model to artifacts/rf_best_auc_K{best_K}.joblib (AUC={best_auc:.4f}, ACC={best_acc:.4f})")

    # 2) Save a recommended operating point (prefer K=50 if present)
    preferred_K = 50
    available_K = [r[0] for r in results]
    if preferred_K in available_K:
        rec = next(r for r in results if r[0] == preferred_K)
    else:
        # fallback: closest K to 50
        rec = min(results, key=lambda r: abs(r[0] - preferred_K))

    rec_K, rec_auc, rec_acc, *_rest2, rec_model = rec
    joblib.dump(rec_model, f"artifacts/rf_recommended_K{rec_K}.joblib")
    print(f"Saved recommended RF model to artifacts/rf_recommended_K{rec_K}.joblib (AUC={rec_auc:.4f}, ACC={rec_acc:.4f})")
    print("Saved sweep metrics to artifacts/rf_sweep_results.json")


if __name__ == "__main__":
    main()
