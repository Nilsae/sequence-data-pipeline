-- sql/seed.sql
BEGIN;

-- Two tiny sessions for smoke testing
INSERT INTO sessions (session_id, env_id, seed, policy, success, length_steps, return_sum)
VALUES
  (1, 'MountainCar-v0', 123, 'random', FALSE, 4, -4.0),
  (2, 'MountainCar-v0', 456, 'random', TRUE,  4, -4.0);

-- Session 1: truncated (failure)
INSERT INTO events (session_id, t, position, velocity, action, reward, terminated, truncated)
VALUES
  (1, 0, -0.500,  0.000, 2, -1.0, FALSE, FALSE),
  (1, 1, -0.498,  0.002, 2, -1.0, FALSE, FALSE),
  (1, 2, -0.495,  0.003, 1, -1.0, FALSE, FALSE),
  (1, 3, -0.494,  0.001, 0, -1.0, FALSE, TRUE);

-- Session 2: terminated (success)
INSERT INTO events (session_id, t, position, velocity, action, reward, terminated, truncated)
VALUES
  (2, 0,  0.450,  0.020, 2, -1.0, FALSE, FALSE),
  (2, 1,  0.470,  0.025, 2, -1.0, FALSE, FALSE),
  (2, 2,  0.495,  0.030, 2, -1.0, FALSE, FALSE),
  (2, 3,  0.520,  0.035, 2, -1.0, TRUE,  FALSE);

COMMIT;
