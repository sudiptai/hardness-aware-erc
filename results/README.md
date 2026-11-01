# Result files

Every CSV has one row per (fold, seed) with `dev_WF1, test_WF1, test_MF1, test_UA`; policy runs add `kbar`, `tau`, `unresolvable_frac` and, for the hardness policy, the `curve_*` columns used for Fig. 2 (W-F1 vs mean context for hardness-k and random-k at the same k-bar, inference on the window-agnostic pass-1 model).

* `iemocap/` — final protocol (Sessions 1–4 train with 10 % of dialogues as dev, Session 5 test; `LOSO_*` hold out Sessions 1–4 in turn). Produced 28 Oct (fixed-k, LOSO, entropy), 30 Oct (hardness policy rev. 4 and ablations) and 1 Nov 2026 (random-k at the policy's k-bar).
* `emorynlp/` — text-only control, final protocol.
* `summary/` — aggregated tables as printed by notebook Section 10 (`summary_tables_II_IV.csv`), paired-bootstrap CIs vs the hardness-k default (`paired_bootstrap_CIs.csv`, delta = hardness − row), Table III quartile tables, oracle k* by quartile, Table VI calibration/disagreement.
* `old_policy_28oct/` — the superseded IEMOCAP policy runs in which the classifier was trained at k = 8 and evaluated at the predicted k (paper Sec. III-F). Kept for the audit trail; not used in any table.
* MELD per-seed files, `*_pooled.npz` predictions, `table_perclass_iemocap.csv`, `confusion_*`, and `text_only_pilot/` are in the Google Drive `hardness_erc/results/` folder and are added at release.
