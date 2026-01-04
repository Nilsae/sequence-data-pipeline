import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import gymnasium as gym
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()


@dataclass
class EpisodeResult:
    success: bool
    length_steps: int
    return_sum: float


def get_conn():
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        dbname=os.getenv("PGDATABASE", "sequence_pipeline"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD") or None,
    )


def insert_session(cur, env_id: str, seed: int, policy: str, success: bool, length_steps: int, return_sum: float) -> int:
    cur.execute(
        """
        INSERT INTO sessions (env_id, seed, policy, success, length_steps, return_sum)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING session_id;
        """,
        (env_id, seed, policy, success, length_steps, return_sum),
    )
    return cur.fetchone()[0]


def insert_events(cur, session_id: int, rows: List[Tuple[int, float, float, int, float, bool, bool]]):
    sql = """
        INSERT INTO events (session_id, t, position, velocity, action, reward, terminated, truncated)
        VALUES %s
    """
    values = [(session_id,) + r for r in rows]
    execute_values(cur, sql, values, page_size=2000)

def choose_action(policy: str, action_rng: np.random.Generator, position: float, velocity: float) -> int:
    if policy == "random":
        return int(action_rng.integers(0, 3))
    if policy == "heuristic":
        return 2 if velocity > 0 else 0
        # If the car is currently moving to the right, keep pushing right
        # If it’s moving to the left (or stationary), push left
    raise ValueError(f"Unknown policy: {policy}")

def run_episode(env, episode_seed: int, action_rng: np.random.Generator, policy: str):
    obs, _info = env.reset(seed=episode_seed)

    rows: List[Tuple[int, float, float, int, float, bool, bool]] = []
    return_sum = 0.0

    while True:
        t = len(rows)
        position, velocity = float(obs[0]), float(obs[1])

        action = choose_action(policy, action_rng, position, velocity)
        obs, reward, terminated, truncated, _info = env.step(action)

        reward_f = float(reward)
        return_sum += reward_f

        rows.append((t, position, velocity, action, reward_f, bool(terminated), bool(truncated)))

        if terminated or truncated:
            success = bool(terminated)
            return rows, EpisodeResult(success=success, length_steps=len(rows), return_sum=return_sum)


def main(
    n_episodes: int = 500,
    base_seed: int = 1000,
    action_seed: int = 999,
    env_id: str = "MountainCar-v0",
    # policy: str = "random",
    policy: str = "heuristic",
):
    # Gymnasium env
    env = gym.make(env_id)
    action_rng = np.random.default_rng(action_seed)

    conn = get_conn()
    conn.autocommit = False

    try:
        with conn.cursor() as cur:
            for i in range(n_episodes):
                episode_seed = base_seed + i
                rows, result = run_episode(env, episode_seed, action_rng, policy)

                session_id = insert_session(
                    cur,
                    env_id=env_id,
                    seed=episode_seed,
                    policy=policy,
                    success=result.success,
                    length_steps=result.length_steps,
                    return_sum=result.return_sum,
                )
                insert_events(cur, session_id, rows)

                if (i + 1) % 50 == 0:
                    conn.commit()
                    print(f"Inserted {i+1}/{n_episodes} episodes (latest session_id={session_id}, success={result.success})")

            conn.commit()

        print("Done.")

    finally:
        env.close()
        conn.close()


if __name__ == "__main__":
    main()
