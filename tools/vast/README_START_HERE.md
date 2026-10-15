# COMP8851 | Pre-rental readiness package (v2)

**Status: PREPARED FOR PROVISIONING; NOT YET READY FOR CANONICAL TRAINING.** No GPU has been rented, no datasets have been copied and no model has been trained using this package. The package contains reproducibility guards and scaffolding—not invented training artifacts.

## What we are preserving from Assessment 2

The six datasets (YelpChi, Amazon, T-Finance, Elliptic, FDCompCN and T-Social), frozen split seed **2**, final training seeds **2/42/72**, four training ratios **TR40/TR30/TR20/TR10**, TR40-only **12-trial** validation-AUPRC hyperparameter selection, frozen winning settings for all lower ratios, separate final test evaluation, and seven metrics. Training uses max **100 epochs** and patience **20 where applicable**, with model-specific schedule exceptions approved by Venus. Benchmark hardware remains **1× RTX A6000 48 GB**. Evidence must include hashes, raw per-run metrics, model code revision, exact configuration, and separate timing/memory measurements.

The **10-model Ass2 evidence closure** is higher priority than 3 new models. Existing evidence is *never overwritten*. Read `guides/PRIOR_WORK_AND_UNRESOLVED.md`.

**v3 improvements (checked 2026-10-15):** fetches exact locked commit even if branch tip moves; adds local self-check; expands step-by-step first-30-minute guide; captures historical host/disk constraints. Package checks never launch a paid instance. Source locks have been rechecked against the GitHub commit API.

## Before clicking Rent

1. Consult `guides/FIRST_30_MINUTES.md` and `configs/vast_offer_checklist.json` first. Prefer **Romania host 64430 / m:10958** if still available: approximately 144.6 GB disk, 6 allocated CPU, 48 GB system RAM, A6000. Host **45423** and host **250721** offered only ~60 GB storage in screenshots. Do not attempt the full T-Social and three-env workflow on 60 GB without a documented storage plan. Kansas **73215** has 64 GB RAM and slower storage, but was connected with a previous blocked GAAP allocation; check stability.
2. **Do not assume the offer ID equals a host ID or machine ID**. Inspect the offer in Vast.ai; launch with the actual offer ID. Verify real available disk *and* the allocation field, download bandwidth, reliability, expiry, and hourly total. Disk cannot be resized after creation.
3. Confirm you can recover **canonical asset files** from the old Vast volumes or your local backups. The team GitHub branch does not contain the actual split registry in `shared/splits` (only `.gitkeep` at inspection time).
4. If a previous instance remains accessible, run `bash scripts/08_find_prior_assets_READONLY.sh /workspace` there after copying this single script. It locates exact T-Social/split/evidence candidates without deleting or moving anything; transfer only checksum-verified source assets.
5. Check spare cost budget for setup, idle while transferring, and storage charges after stopping. Set a reminder to stop the instance when not in use.
6. Review `configs/assets_manifest.json`. Two historical T-Social checksums are entered; paths are **not yet available**. All other hashes are deliberately blank.
7. Read the compatibility blockers in `guides/MODEL_SPECIFIC_BLOCKERS.md`. Do not run upstream default jobs as canonical benchmark runs.

### FIRST STEP locally: verify downloaded archive (NO RENT)

```bash
bash scripts/09_local_selfcheck.sh
```

Works on macOS/Linux with `python3` (no GNU `sha256sum` required). Expected: all package checks pass. The canonical data are **not** provided and must still be transferred.

### Optional local dry-run preview — DOES NOT RENT

```bash
bash scripts/00_offer_dry_run.sh OFFER_ID 120
```

This prints the command for a 120 GB allocation. It does not execute it. For a ~60 GB offer you could preview a 50 GB disk, **but that offer is not recommended** for full benchmark work.

## Right after renting (first five minutes)

Transfer this entire folder or zip via Vast CLI, then on the GPU instance:

```bash
cd /workspace
unzip COMP8851_PreRent_Ready_v3.zip
cd COMP8851_PreRent_Ready_v3
bash scripts/09_local_selfcheck.sh
bash scripts/01_host_preflight.sh | tee host_preflight.log
bash scripts/02_bootstrap_repositories.sh | tee bootstrap.log
python3 scripts/03_verify_assets.py --manifest configs/assets_manifest.json --workspace /workspace/comp8851
python3 scripts/04_plan_experiments.py --output /workspace/comp8851/manifests/blocked_job_plan.csv
```

**Expected asset verifier result before copying assets: FAIL/BLOCKED. This is correct.** It prevents an accidental run on incorrect datasets or splits. The bootstrap clones exact locked Git revisions and does **not** install model environments or train.

### After you have copied verified assets

Make the `source_path`, `split_path` and SHA256 values in `configs/assets_manifest.json` refer to exact canonical data. Run the verifier for each dataset again. Never use newly generated splits just to pass a gate. A proposed change must be documented and approved.

Then create model environments *one at a time* using `guides/ENVIRONMENT_SETUP.md`, run **author-mode smoke on Amazon**, and independently adapt and test the frozen shared split and validation-only evaluator. The package does not claim that those model-specific adapters exist yet. See `guides/CANONICAL_ADAPTER_CONTRACT.md`.

## How the project should run

```
ASSET HASH GATE -> ENV REPRO GATE -> 2-EPOCH SMOKE WITHOUT TEST
 -> TR40 12-TRIAL VAL-AUPRC TUNING -> FREEZE WINNER
 -> TR40/30/20/10 x SEEDS 2/42/72 -> FINAL TEST ONCE PER RUN
 -> PER-SEED CSV + SUMMARY + MANIFEST + SHA256 + COST EVIDENCE
```

Use `scripts/05_create_run_manifest.py` **before every run** and `scripts/06_audit_evidence.py` **afterward**. These are evidence scaffolding and checks, not trainer implementations.

## Upstream sources (pinned October 2026)

`configs/source_lock.json` pins four source repositories (GitHub revisions rechecked on 2026-10-15). The bootstrap fetches and detaches to the exact source commit even if the branch has advanced. It stops if local changes are present; no author code is edited. GitHub repository code can have bugs or mismatched defaults: freeze and validate each one.

## What this package does NOT do

- Does not transfer canonical datasets, checkpoints, historical evidence or complete source code from your retired Vast instances.
- Does not install incompatible Python environments into one environment.
- Does not repair RGTAN, SAWGAD or EOGFD in a way that changes their models.
- Does not bypass the strict **validation-only** selection or allow test peeking.
- Does not automatically rent, tune or start model training.

The next genuinely blocked step is **recovering and verifying the canonical assets**, especially the T-Social graph and split, plus collecting the latest PMP/GAAP evidence from any remaining instance/backups.
