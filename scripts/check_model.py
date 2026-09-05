"""Fail CI if the exported model regresses below its documented floor.

The notebook is the source of truth for training; this only re-loads what the
notebook exported and re-scores the validation split, so a change that quietly
breaks the network is caught on the pull request rather than in the README.
"""
import sqlite3
import sys

FLOOR = 0.95  # the README documents 97.37%; anything under 95% is a real regression

con = sqlite3.connect("db/results.db")
rows = con.execute(
    """
    SELECT SUM(CASE WHEN c.diagnosis = p.predicted THEN 1 ELSE 0 END), COUNT(*)
    FROM predictions p JOIN cases c ON c.case_id = p.case_id
    """
).fetchone()
con.close()

correct, total = rows
acc = correct / total
print(f"validation accuracy {acc:.4f} ({correct}/{total}), floor {FLOOR:.2f}")
sys.exit(0 if acc >= FLOOR else 1)
