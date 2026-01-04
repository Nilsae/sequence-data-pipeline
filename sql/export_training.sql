-- sql/export_training.sql
-- Export step-level events joined with session labels for epsilon_heuristic only.

SELECT
  e.session_id,
  s.success,
  e.t,
  e.position,
  e.velocity,
  e.action,
  e.reward
FROM events e
JOIN sessions s ON s.session_id = e.session_id
WHERE s.policy = 'epsilon_heuristic'
ORDER BY e.session_id, e.t;
