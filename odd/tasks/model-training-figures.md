# Model training and evaluation figures (thesis, RUD5 style)

## Objective
Produce the evidence the thesis needs to document training and evaluation the way the
reference document does (console output of dataset processing, training progress per
iteration, final metrics, training curves, precision-confidence and recall-confidence
curves, confusion matrix), adapted to a TF-DF Gradient Boosted Trees classifier.

## Problem
- `build-dataset` prints scattered per-course log lines; there is no summary of what the
  processing produced (rows read, discarded and why, classes, gestiones).
- `train` silences TF-DF (`verbose=0`), never persists `training_logs`, and discards the
  out-of-fold probabilities (`train.py:133-134`), so no curve beyond the six existing
  figures can be drawn.
- GBT early stopping uses TF-DF's internal `validation_ratio=0.1` split **by row**, so the
  same student sits in training and in the early-stopping set: its validation curve is
  optimistic.

## Why
The thesis chapter is being rewritten in the reference style; every figure must show this
model's real process and every number must come from one run.

## Scope (authorized)
- Console summary for `build-dataset`.
- `train`: early-stopping validation grouped by student (option B, user decision
  2026-10-24), persisted training logs per fold and final model, persisted out-of-fold
  probabilities, per-fold and per-class console tables, optional verbose TF-DF output.
- `figuras`: four new figures (training results, precision-confidence,
  recall-confidence, normalized confusion matrix).
- Tests for the new pure logic; `reports/LAMINAS.md` updated.
- Retrain in WSL and regenerate all figures. Numbers WILL change (accepted by the user).

## Constraints
- AGENTS.md rules: missing data is `None`/`NaN`, never 0; no student names in logs;
  `contract.py` stays the single feature builder; splits grouped by student.
- TF-DF runs only in WSL (Ubuntu, `.venv-wsl`); figures must keep running on Windows
  without importing TensorFlow (they read JSON/parquet only).
- `data/`, `models/`, `reports/figures/` stay out of git.
- Back up the current model before retraining.

## Verified facts (2026-10-24, TF-DF 1.10 probe in WSL)
- `GradientBoostedTreesModel.fit(train, validation_data=valid)` is supported.
- `make_inspector().specialized_header().training_logs.entries[i]` exposes
  `number_of_trees`, `training_loss`, `training_secondary_metrics` (accuracy),
  `validation_loss`, `validation_secondary_metrics` (accuracy).

## Tasks
- [x] T1 Dataset processing summary printed by `build-dataset`.
- [x] T2 Training: grouped early-stopping validation, persisted training logs and
      out-of-fold probabilities, console tables.
- [x] T3 Figures: training results, precision-confidence, recall-confidence,
      normalized confusion matrix.
- [x] T4 Retrain in WSL, regenerate figures, record the new numbers.
- [x] T4b Class-weighting iteration (user decision 2026-10-24): four candidates, persisted
      selection rationale, RF out-of-bag training figure, mean line only where all classes
      measure, candidate comparison figure; retrain and LAMINAS numbers.
- [ ] T5 Thesis text for 3.4.6 and 4.2.4 in RUD5 style (delivered in chat).

## Route
- T1-T3: delegated direct (one writer; 3+ non-trivial files).
- T4: same writer, WSL run.
- T5: parent, in chat (user edits the .docx himself).

## Acceptance criteria
- `pytest` and `ruff check src tests` green on Windows `.venv`.
- `train` in WSL writes `registro_entrenamiento.json` and
  `predicciones_fuera_de_pliegue.parquet`; `figuras` writes 10 PNGs on Windows.
- No student overlap between training, early-stopping and validation sets (asserted).

## Progress
- 2026-10-24: feature document created; TF-DF API verified.
- 2026-10-24: T1-T3 implemented (writer, delegated). New: `evaluation/consola.py`
  (console tables), `cli metricas`, `train --detalle`; tests `test_resumen_dataset.py`,
  `test_entrenamiento.py`, `test_curvas_confianza.py`, `TestPromediosPorClase`. RED observed
  (import errors), then GREEN: pytest 170 passed; ruff check/format clean.
- 2026-10-24: T4 run. Backup at `models/tfdf_riesgo__previo_20261024`. Dataset 1092 rows /
  33 students (unchanged). With grouped early stopping GBT stops at 21-41 iterations per fold
  (kept 13-33); CV: GBT acc 0.637±0.029, macro F1 0.534±0.062, critical recall 0.062±0.064
  (4/5 folds); RF acc 0.660±0.062, macro F1 0.563±0.095, critical recall 0.125±0.178 (4/5).
  Macro F1 difference falls inside the tie band (max std 0.095), so `_elegir` picks
  random_forest by critical recall. The served model is now RF. `figuras` writes 9 PNGs;
  `resultados_entrenamiento.png` is skipped because RF logs no iterations.
- Next: user decides — keep RF (and draw the training figure from the GBT folds, or drop it),
  or keep GBT as the served model. Then update LAMINAS sections 1-6 numbers.
- 2026-10-24 (T4b): `entrenamiento.ponderacion_clases: [ninguna, balanceada]`; four
  candidates `gradient_boosted_trees`, `random_forest`, `*_ponderado`. Balanced weights from
  the fitting rows only, same weights on the GBT early-stopping set, weight column
  `peso_de_clase` (a leading `__` is stripped by `pd_dataframe_to_tf_dataset` -> KeyError).
  Results: GBT 0.534±0.062 macro F1 / 0.062 critical recall; RF 0.563±0.095 / 0.125;
  GBT pond. 0.536±0.069 / 0.161; RF pond. 0.555±0.079 / 0.223. Band 0.095 (RF) holds all four;
  winner `random_forest_ponderado` by critical recall. metadata: `modelo=random_forest`,
  `candidato`, `ponderacion_clases`, `pesos_clases`, `seleccion`. OOF: acc 0.634, macro F1
  0.534, critical recall 0.200 (10/50). RF logs: ~30 OOB checkpoints (num_trees, accuracy,
  loss, num_examples). Backup `models/tfdf_riesgo__iter4_rf_20261024`. figuras: 11 PNGs.
  Checks: pytest 188 passed; ruff check/format clean. Route: delegated writer.
- Next: T5 thesis text (parent).
- 2026-10-24: committed on `dev` (GGA PASSED on each, after fixing its findings):
  `09fa89d` training, `570f832` log entry without iteration, `c55e15f` figures,
  `69b302c` 0.5 threshold lookup, `8f00db3` console/CLI, `58b5ea3` privacy + n/d display,
  plus this docs commit. pytest 193 passed; ruff check/format clean. Not pushed.
