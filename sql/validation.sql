-- sql/validation.sql
-- Purpose: data quality checks before exporting to ML training

-- 1) Duplicate timesteps per session (should be zero rows)
SELECT session_id, t, COUNT(*) AS n
FROM events
GROUP BY session_id, t
HAVING COUNT(*) > 1;

-- 2) Sessions whose declared length_steps doesn't match actual number of events
SELECT s.session_id, s.length_steps, COUNT(e.event_id) AS actual_steps
FROM sessions s
LEFT JOIN events e ON e.session_id = s.session_id
GROUP BY s.session_id, s.length_steps
HAVING COUNT(e.event_id) <> s.length_steps;

-- 3) Sessions missing timestep 0 or with gaps in timesteps
-- (requires events per session ordered by t)
WITH ordered AS (
  SELECT
    session_id,
    t,
    LAG(t) OVER (PARTITION BY session_id ORDER BY t) AS prev_t
  FROM events
),
gaps AS (
  SELECT session_id, t, prev_t
  FROM ordered
  WHERE prev_t IS NOT NULL AND t <> prev_t + 1
),
missing_zero AS (
  SELECT session_id
  FROM events
  GROUP BY session_id
  HAVING MIN(t) <> 0
)
SELECT 'gap' AS issue, session_id, prev_t, t
FROM gaps
UNION ALL
SELECT 'missing_t0' AS issue, session_id, NULL::INTEGER AS prev_t, NULL::INTEGER AS t
FROM missing_zero
ORDER BY session_id, issue;

-- 4) Terminal flag sanity: exactly one terminal step per session
-- In Gymnasium, the final step has either terminated OR truncated, and earlier steps are both false.
WITH terminal_counts AS (
  SELECT
    session_id,
    SUM(CASE WHEN terminated THEN 1 ELSE 0 END) AS n_terminated,
    SUM(CASE WHEN truncated THEN 1 ELSE 0 END) AS n_truncated,
    SUM(CASE WHEN terminated OR truncated THEN 1 ELSE 0 END) AS n_terminal_rows
  FROM events
  GROUP BY session_id
)
SELECT *
FROM terminal_counts
WHERE n_terminal_rows <> 1 OR (n_terminated + n_truncated) <> 1;

-- 5) Basic range sanity for MountainCar (soft check)
-- MountainCar typical bounds: position ~ [-1.2, 0.6], velocity ~ [-0.07, 0.07]
SELECT *
FROM events
WHERE position < -1.3 OR position > 0.7
   OR velocity < -0.08 OR velocity > 0.08;
