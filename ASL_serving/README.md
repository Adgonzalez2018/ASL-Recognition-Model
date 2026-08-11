# ASL_serving

Inference, capture, and delivery for the isolated ASL sign classifier trained in `ASL_training`.

This project is **planned but not implemented**. It currently contains documentation only. No source code, dependencies, or deployment configuration exist yet, and none should be added before the entry conditions in `docs/CURRENT_PHASE.md` are met.

## What this project does

Receives one short video clip containing one isolated ASL sign and returns either:

* a predicted gloss and a calibrated confidence score, or
* an explicit abstention, when confidence falls below the selected threshold

## What it consumes

Everything model-related arrives from `ASL_training` as one versioned bundle, produced by training Phase 9. This project does not train, does not reimplement preprocessing from memory, and does not redefine the label map. See `docs/BUNDLE_CONTRACT.md`.

## Documents

Read in this order:

1. `docs/PROJECT.md` — scope and success criteria
2. `docs/ARCHITECTURE.md` — layers and boundaries
3. `docs/ROADMAP.md` — phases S0 through S8
4. `docs/CURRENT_PHASE.md` — active phase, blockers, entry conditions
5. `docs/BUNDLE_CONTRACT.md` — what arrives from training
6. `docs/INFERENCE_CONTRACT.md` — clip to tensor to prediction
7. `docs/CAPTURE_CONTRACT.md` — clip acquisition and boundaries
8. `docs/CONFIDENCE_CONTRACT.md` — calibration and abstention
9. `docs/API_CONTRACT.md` — request and response shapes
10. `docs/DEPLOYMENT.md` — runtime and packaging
11. `docs/MONITORING_CONTRACT.md` — logging and drift
12. `docs/DECISIONS.md` — lasting decisions
13. `docs/OPEN_QUESTIONS.md` — what is still blocked, and on what

## Status

Waiting on `ASL_training` Phase 5 (baseline experiments) and Phase 9 (handoff package). Preprocessing, bundle, and API design work can proceed now; anything requiring a measured accuracy, latency, temperature, or threshold value cannot.
