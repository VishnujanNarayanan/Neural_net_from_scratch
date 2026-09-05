-- Analytical queries over the diagnosis run.
--
-- The notebook writes two tables into db/results.db after evaluating the network:
--
--   cases       one row per patient case: the 30 nucleus measurements, the true
--               diagnosis, and which split the case landed in
--   predictions one row per validation case: the model's P(benign) and the label
--               it assigned at the 0.50 threshold
--
-- Every claim the README makes in its Findings section is derived here rather than
-- typed by hand, so re-running the notebook re-derives the numbers instead of
-- letting the prose drift away from the model.
--
-- Run standalone:   sqlite3 db/results.db < db/queries.sql

-- name: class_balance
-- Does the stratified split actually preserve the 37/63 malignant/benign ratio?
SELECT
    split,
    diagnosis,
    COUNT(*)                                                    AS n,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY split), 1) AS pct_of_split
FROM cases
GROUP BY split, diagnosis
ORDER BY split, diagnosis;

-- name: feature_ranges
-- The standardisation rationale: which measurements span the widest raw range?
-- Answered against the training split only, which is the split the scaler saw.
SELECT
    'mean area'        AS feature, MIN(mean_area)        AS lo, MAX(mean_area)        AS hi FROM cases WHERE split = 'train'
UNION ALL
SELECT
    'mean smoothness', MIN(mean_smoothness), MAX(mean_smoothness) FROM cases WHERE split = 'train'
UNION ALL
SELECT
    'worst area',      MIN(worst_area),      MAX(worst_area)      FROM cases WHERE split = 'train';

-- name: misclassified
-- Every validation case the model got wrong, with the score it gave.
SELECT
    p.case_id,
    c.diagnosis                AS actual,
    p.predicted                AS predicted,
    ROUND(p.p_benign, 4)       AS p_benign
FROM predictions p
JOIN cases c ON c.case_id = p.case_id
WHERE c.diagnosis <> p.predicted
ORDER BY p.p_benign;

-- name: score_separation
-- How far apart are the two classes? The README claims median P(benign) of
-- roughly 0.01 for malignant cases and 0.95 for benign ones.
SELECT
    c.diagnosis,
    COUNT(*)                                             AS n,
    ROUND(MIN(p.p_benign), 4)                            AS min_p_benign,
    ROUND(AVG(p.p_benign), 4)                            AS mean_p_benign,
    ROUND(MAX(p.p_benign), 4)                            AS max_p_benign
FROM predictions p
JOIN cases c ON c.case_id = p.case_id
GROUP BY c.diagnosis;

-- name: threshold_sweep
-- Is the one missed malignancy recoverable by moving the decision threshold?
-- Counts, at each candidate threshold, how many malignant cases are called benign
-- (the clinically expensive error) against how many false alarms it costs.
WITH thresholds(t) AS (
    VALUES (0.30), (0.40), (0.50), (0.60), (0.70), (0.80), (0.90)
)
SELECT
    t.t                                                                        AS threshold,
    SUM(CASE WHEN c.diagnosis = 'malignant' AND p.p_benign >= t.t THEN 1 ELSE 0 END) AS malignant_called_benign,
    SUM(CASE WHEN c.diagnosis = 'benign'    AND p.p_benign <  t.t THEN 1 ELSE 0 END) AS benign_called_malignant,
    ROUND(100.0 * SUM(
        CASE WHEN (c.diagnosis = 'benign') = (p.p_benign >= t.t) THEN 1 ELSE 0 END
    ) / COUNT(*), 2)                                                           AS accuracy_pct
FROM thresholds t
CROSS JOIN predictions p
JOIN cases c ON c.case_id = p.case_id
GROUP BY t.t
ORDER BY t.t;
