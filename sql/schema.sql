-- sql/schema.sql
BEGIN;

DROP TABLE IF EXISTS events CASCADE;
DROP TABLE IF EXISTS sessions CASCADE;

CREATE TABLE sessions (
    session_id    BIGSERIAL PRIMARY KEY,
    env_id        TEXT NOT NULL,                          -- e.g., 'MountainCar-v0'
    seed          INTEGER NOT NULL,
    policy        TEXT NOT NULL DEFAULT 'random',          -- e.g., 'random'
    success       BOOLEAN NOT NULL,                        -- true if terminated, false if truncated
    length_steps  INTEGER NOT NULL CHECK (length_steps >= 1),
    return_sum    DOUBLE PRECISION NOT NULL,              -- sum of rewards (often -length_steps)
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE events (
    event_id     BIGSERIAL PRIMARY KEY,
    session_id   BIGINT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    t            INTEGER NOT NULL CHECK (t >= 0),

    position     DOUBLE PRECISION NOT NULL,
    velocity     DOUBLE PRECISION NOT NULL,

    action       SMALLINT NOT NULL CHECK (action IN (0, 1, 2)),
    reward       DOUBLE PRECISION NOT NULL,

    terminated   BOOLEAN NOT NULL,
    truncated    BOOLEAN NOT NULL,

    -- No duplicate timestep within a session
    CONSTRAINT uniq_session_t UNIQUE (session_id, t) --,

    -- Sanity: you should not have both true at the same time
    -- CONSTRAINT chk_terminal_flags CHECK (NOT (terminated AND truncated))
);

-- Indexes for the exact access pattern: load ordered sequences per session
CREATE INDEX idx_events_session_t ON events (session_id, t);
CREATE INDEX idx_sessions_env_policy ON sessions (env_id, policy);

COMMIT;
