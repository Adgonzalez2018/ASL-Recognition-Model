# Deployment

## Purpose

Defines how the service is packaged, configured, and run.

Most concrete numbers in this document are unset. They depend on the architecture selected in training Phase 5 and on latency measured in serving Phase S7. Placeholders are marked explicitly rather than guessed, because a plausible-looking wrong number tends to outlive the moment it was invented.

## Runtime Shape

Single process, model resident in memory, loaded once at startup.

```text
container
└── service process
    ├── bundle, mounted or fetched at startup
    ├── model in memory
    └── HTTP listener
```

## Startup Sequence

1. Read configuration.
2. Locate and load the bundle.
3. Verify the bundle per `BUNDLE_CONTRACT.md`.
4. Build the architecture, restore weights.
5. Run the parity smoke check against at least one reference fixture.
6. Mark ready.
7. Accept traffic.

Any failure in steps 2 through 5 aborts startup. The service must not accept traffic in a degraded state, and must not fall back to a default or previous bundle.

Requests arriving before step 6 receive an explicit not-ready response.

### On the startup parity check

Running one reference fixture at startup costs a fraction of a second and catches the class of failure that is otherwise invisible: a bundle that loads cleanly but has been paired with the wrong preprocessing or a mismatched decoder version in the image.

## Configuration

Supplied by environment:

| Setting | Purpose |
|---|---|
| bundle location | Path or fetch URI |
| device | CPU or GPU selection |
| host and port | |
| request limits | Upload size, timeout, concurrency |
| clip validity bounds | Duration and resolution, until the bundle carries them |
| log destination | |
| retention policy | Default off |

Never supplied by configuration: frame count, resolution, normalization, label map, temperature, threshold. Those come from the bundle. See `ARCHITECTURE.md`.

No hardcoded absolute paths in the source. Bundle location is always configured.

## Dependencies

Pinned independently of `ASL_training`.

Serving needs inference, not training: no optimizer, no scheduler, no dataset tooling, no notebook stack. The runtime dependency set should be the minimum that builds the architecture, loads weights, decodes video, and serves HTTP.

Two dependencies affect numerics and therefore parity, and must be pinned deliberately and recorded:

* the tensor library
* the video decoder

A decoder upgrade can change frame-seek behavior. A tensor library upgrade can change resize or reduction numerics. Both must trigger a full parity run before release.

## Device

CPU and GPU must both be supported.

CPU is the realistic default for a low-concurrency, single-user practice workload, and it removes an entire class of deployment complexity. GPU becomes justified only if measured latency on CPU proves unacceptable.

Device selection must not change results beyond the documented parity tolerance. Where it does, that is a finding to record, not to absorb.

## Resource Budget

**Unset.** To be filled from measurement in Phase S7.

| Quantity | Value | Source |
|---|---|---|
| Model memory | unset | Depends on selected architecture |
| Peak inference memory | unset | Measure at S7 |
| Per-clip latency, CPU | unset | Measure at S7 |
| Per-clip latency, GPU | unset | Measure at S7 |
| Decode share of latency | unset | Expected to be significant |
| Container image size | unset | |

Training preflight found that decode dominated step time on a 4-core Kaggle host, at roughly half of every step even with the GPU work reduced. Serving decodes one clip per request rather than a batch, but the same lesson applies: **decode is likely to be a meaningful share of serving latency, possibly the majority.** Optimizing model inference while ignoring decode would be the same mistake in a new setting.

## Image Contents

Must contain: application code, pinned dependencies, decoder.

Must not contain: model weights, bundles, dataset material, recorded clips, secrets, tokens.

The bundle is mounted or fetched at runtime. Baking weights into the image conflates two versioning schemes and makes a model change require an image rebuild.

## Health and Readiness

* Readiness is false until the model is loaded and the parity smoke check passes.
* Liveness must not be a bare process check; a process with a failed model is not alive in any useful sense.
* Health output carries bundle identity, so the deployed model version is observable without inspecting the filesystem.

## Rollout

A bundle change is a deployment event.

* Bundle version must be recorded at rollout.
* The previous bundle must remain retrievable for reproducing earlier predictions.
* Rollback means reverting to a previous bundle version, which must remain a supported operation.

Serving two bundles simultaneously for comparison is prohibited; see `BUNDLE_CONTRACT.md`.

## Optimization

Deferred until measured.

ONNX export, quantization, fused kernels, and batching all change numerics or timing behavior. Each requires:

1. A measured latency problem it addresses.
2. A full parity re-run.
3. A recorded decision, including any tolerance change.

Given that decode may dominate, model-side optimization could deliver a small fraction of its apparent benefit. Measure first.

## Environments

| Environment | Purpose |
|---|---|
| Local development | Stub model, no real bundle required |
| CI | Lint, tests, parity suite against fixtures |
| Staging | Real bundle, real clips, latency measurement |
| Production | Deferred until staging results exist |

Local development must work without a real bundle. Requiring one would block all S0 through S4 work on training completion, which is exactly the coupling this project is structured to avoid.
