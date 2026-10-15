# Before and after renting: precise sequence

## A. On your computer — BEFORE renting

1. Download `COMP8851_PreRent_Ready_v3.zip` and `COMP8851_PreRent_Ready_v3_README.md`.
2. Unzip the ZIP and open `README_START_HERE.md` inside the new folder.
3. Run `bash scripts/09_local_selfcheck.sh` inside the extracted folder. Expected: all package files PASS, exactly 432 **BLOCKED** planned new-model fits.
4. Confirm an accessible copy of each **canonical source graph and frozen split** from the previous BWGNN/PMP/GAAP project. The GitHub checkout **does not contain those canonical split files**. If you cannot locate them, prepare for recovery before tuning.
5. Check Vast.ai candidate offer using its visible ID and host ID **separately**. Select A6000 48 GB, on-demand, SSH/Direct and adequate disk. Host `64430` previously listed ~144 GB; hosts `45423`/`250721` showed only ~60 GB; screenshots are historical, not live availability.
6. Disk allocation is **fixed on creation**, storage bills even while stopped. Prefer ~120 GB on a confirmed ~145 GB offer, leaving headroom; do **not** request 120 GB on a ~60 GB offer. For 60 GB restrict to small smoke tests and dataset streaming; don't expect full T-Social and all envs to fit.
7. Before payment: examine compute, storage and bandwidth rates; ensure you can back up outputs. There is no cost estimate in this package for storage or transfer.
8. UI rental is fine. A no-rent preview from CLI is `bash scripts/00_offer_dry_run.sh YOUR_OFFER_ID 120` and does not create an instance. `YOUR_OFFER_ID` is not the machine/host number.

## B. FIRST 10 MINUTES on the newly rented instance

**Nothing below runs official experiments.** Upload ZIP using Vast UI file uploader or Vast CLI `vastai copy local:./COMP8851_PreRent_Ready_v3.zip INSTANCE_ID:/workspace/` (check `vastai copy --help` for your installed version).

```bash
cd /workspace
unzip COMP8851_PreRent_Ready_v3.zip
cd COMP8851_PreRent_Ready_v3
bash scripts/09_local_selfcheck.sh
bash scripts/01_host_preflight.sh | tee host_preflight.log
bash scripts/02_bootstrap_repositories.sh | tee bootstrap.log
```

Checkpoint: see `SOURCE PINS ALL PASS`, correct A6000 48 GB, enough RAM/free disk. If anything fails, **stop** rather than blindly installing packages.

## C. Data recovery and source-of-truth checks

1. From a previously accessible instance, run `bash scripts/08_find_prior_assets_READONLY.sh /workspace` there and save its output. It only locates files, reads two candidate SHA256s, and does not modify old storage.
2. Transfer source graphs and `seed2` split files through an authenticated supported transfer method. Avoid publicly exposing original data. Keep raw originals unchanged.
3. Copy the template to a **mutable** manifest, leave historical data untouched:

```bash
cd /workspace/COMP8851_PreRent_Ready_v3
cp configs/assets_manifest.json /workspace/comp8851/manifests/assets_manifest_working.json
# Edit the working JSON with actual source_path and split_path and independently verified SHA256 values.
python3 scripts/03_verify_assets.py --manifest /workspace/comp8851/manifests/assets_manifest_working.json --workspace /workspace/comp8851 --dataset T-Social
```

Expected: `FILE_SHA_GATE=BLOCKED` until the files are present. Do not fabricate hashes to bypass it. After file SHA verification, still conduct the manual graph structure and split-semantics audit.

## D. Environment and author-mode smoke

1. Read `guides/ENVIRONMENT_SETUP.md` for model-specific versions. Install **separate** RGTAN, SAWGAD and EOGFD environments, one at a time.
2. Use Amazon as an initial native smoke dataset, then assess canonical split eligibility. Native reproduction is **not** a canonical result.
3. Preserve SAWGAD's synthetic training mechanism and flag weakly-supervised 100+350 schedule for Venus; RGTAN risk-feature labels and 5 folds need vetting; EOGFD's released `EOGFD` import mismatch and test-in-epoch code must be fixed in an audited runner before canonical work.
4. No official tuning before source/split checks, supervisor decisions and validation-only adapter pass.

## E. Only after approval and audit

```bash
python3 scripts/04_plan_experiments.py --output /workspace/comp8851/manifests/blocked_job_plan.csv
python3 scripts/05_create_run_manifest.py --model RGTAN --dataset Amazon --ratio .4 --seed 2 --phase SMOKE --output /workspace/comp8851/manifests/RGTAN_Amazon_smoke_TEMPLATE.json
```

These scripts create **plans and blocked evidence headers**, not real trainers. The complete missing piece is an independently checked adapter+runner for each new model against the canonical data. Do not start the full 432 fits without them.

## F. Preserve evidence and stop billing

Before stopping or destroying an instance, export code lock files, dataset hashes, run logs, configs, per-seed scores, environment manifests and failures off-instance. Verify the copied checksums. Stopping releases compute but **storage may continue billing**; destroying ends container storage charges and irreversibly deletes data.
