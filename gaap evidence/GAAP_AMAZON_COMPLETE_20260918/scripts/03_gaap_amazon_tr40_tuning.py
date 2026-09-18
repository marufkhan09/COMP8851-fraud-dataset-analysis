#!/usr/bin/env python3
from pathlib import Path
import json, subprocess, sys, re

ROOT = Path("/workspace/gaap_vast")
BASE = ROOT / "scripts/02_gaap_amazon_unified_tr40_smoke.py"
LAUNCHER = ROOT / "gaap_python.sh"
OUT = ROOT / "unified/amazon/tuning"
TRIALS = OUT / "trials"
WINNER = OUT / "GAAP_AMAZON_FROZEN_WINNER.json"
SUMMARY = OUT / "GAAP_AMAZON_TR40_TUNING_SUMMARY.json"
REPO = ROOT / "repo/GAAP"
EXPECTED_COMMIT = "6a7dbb0447c4897504525de49e41a0526ee777f8"

OUT.mkdir(parents=True, exist_ok=True)
TRIALS.mkdir(parents=True, exist_ok=True)

grid = []
for hidden in (64, 128):
    for dropout in (0.1, 0.2):
        for lr in (0.001, 0.002, 0.0005):
            grid.append({"d_hidden": hidden, "gnn_dropout": dropout, "lr": lr})
assert len(grid) == 12
assert grid[0] == {"d_hidden": 64, "gnn_dropout": 0.1, "lr": 0.001}

base = BASE.read_text()
anchors = [
    'assert abs(float(cfg["lr"]) - 0.001) < 1e-15',
    'cfg["max_epochs"] = 2',
    'max_epochs=2,',
    'monitor="val_aps"',
    'patience=20',
    'TEST_EVALUATED=NO',
    'TEST_LABEL_ISOLATION=PASS',
    'GAAP_AMAZON_UNIFIED_2EPOCH_SMOKE=PASS',
]
missing = [x for x in anchors if x not in base]
assert not missing, f"BASE_ANCHORS_MISSING={missing}"

commit = subprocess.check_output(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
).strip()
assert commit == EXPECTED_COMMIT
dirty = subprocess.check_output(
    ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
    text=True,
).strip()
assert dirty == "", dirty

print("===== GAAP x AMAZON TR40 TUNING =====", flush=True)
print("BASE_RUNNER_GATE=PASS", flush=True)
print("GAAP_AUTHOR_REPO_CLEAN=PASS", flush=True)
print("TRIALS=12 | SEED=2 | MAX_EPOCHS=100 | PATIENCE=20", flush=True)
print("SELECTION=VALIDATION_AUPRC_ONLY", flush=True)
print("TEST_EVALUATED=NO", flush=True)
print("TEST_LABEL_ISOLATION=PASS", flush=True)

def replace_out_assignment(text, outdir):
    lines = text.splitlines()
    hits = [i for i, line in enumerate(lines) if re.match(r"^\s*OUT\s*=", line)]
    assert len(hits) == 1, f"OUT_ASSIGNMENTS={hits}"
    i = hits[0]
    lines[i] = f'OUT = Path("{outdir}")'
    return "\n".join(lines) + "\n"

def make_trial(text, trial_no, cfgv, outdir):
    s = replace_out_assignment(text, outdir)

    anchor = 'assert abs(float(cfg["lr"]) - 0.001) < 1e-15'
    insert = (
        anchor + "\n"
        + f'cfg["d_hidden"] = {int(cfgv["d_hidden"])}\n'
        + f'cfg["gnn_dropout"] = {float(cfgv["gnn_dropout"])}\n'
        + f'cfg["lr"] = {float(cfgv["lr"])}\n'
        + f'print("TUNING_TRIAL={trial_no:02d}", flush=True)\n'
        + f'print("TUNING_CONFIG=d_hidden={int(cfgv["d_hidden"])}|gnn_dropout={float(cfgv["gnn_dropout"])}|lr={float(cfgv["lr"])}", flush=True)'
    )
    assert s.count(anchor) == 1
    s = s.replace(anchor, insert, 1)

    assert s.count('cfg["max_epochs"] = 2') == 1
    s = s.replace('cfg["max_epochs"] = 2', 'cfg["max_epochs"] = 100', 1)

    assert s.count('max_epochs=2,') == 1
    s = s.replace('max_epochs=2,', 'max_epochs=100,', 1)

    s = s.replace("run=SMOKE | TR40 seed=2 | ", f"run=T{trial_no:02d} | TR40 seed=2 | ")

    assert s.count('"dataset": "YelpChi"') == 1
    s = s.replace('"dataset": "YelpChi"', '"dataset": "Amazon"', 1)

    assert s.count('"features": 32') == 1
    s = s.replace('"features": 32', '"features": 25', 1)

    if '"mode": "unified_2epoch_smoke"' in s:
        s = s.replace('"mode": "unified_2epoch_smoke"', '"mode": "tr40_tuning_trial"', 1)

    s = s.replace(
        'print("GAAP_AMAZON_UNIFIED_2EPOCH_SMOKE=PASS", flush=True)',
        f'print("GAAP_AMAZON_TUNING_TRIAL_{trial_no:02d}=PASS", flush=True)',
        1,
    )

    assert 'cfg["max_epochs"] = 2' not in s
    assert 'max_epochs=2,' not in s
    assert 'monitor="val_aps"' in s
    assert 'patience=20' in s
    assert 'TEST_EVALUATED=NO' in s
    assert 'TEST_LABEL_ISOLATION=PASS' in s
    return s

results = []

for trial_no, cfgv in enumerate(grid, start=1):
    tdir = TRIALS / f"trial_{trial_no:02d}"
    tdir.mkdir(parents=True, exist_ok=True)
    result_path = tdir / "trial_result.json"
    child_result = tdir / "smoke_result.json"
    child_script = tdir / f"trial_{trial_no:02d}.py"
    child_log = tdir / f"trial_{trial_no:02d}.log"

    if result_path.exists():
        old = json.loads(result_path.read_text())
        if (
            old.get("status") == "PASS"
            and old.get("trial") == trial_no
            and old.get("config") == cfgv
            and old.get("test_evaluated") is False
            and old.get("test_labels_hidden_during_fit") is True
            and old.get("repository_commit") == EXPECTED_COMMIT
        ):
            print(
                f"TRIAL_{trial_no:02d}=RESUME_PASS | "
                f"VAL_AUPRC={float(old['best_val_auprc']):.6f}",
                flush=True,
            )
            results.append(old)
            continue

    child_text = make_trial(base, trial_no, cfgv, tdir)
    child_script.write_text(child_text)

    subprocess.run(
        [str(LAUNCHER), "-m", "py_compile", str(child_script)],
        check=True,
    )

    print("", flush=True)
    print(f"===== TRIAL {trial_no:02d}/12 =====", flush=True)
    print("CONFIG=" + json.dumps(cfgv, sort_keys=True), flush=True)

    with child_log.open("w") as lf:
        proc = subprocess.Popen(
            [str(LAUNCHER), str(child_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            lf.flush()
        rc = proc.wait()

    if rc != 0:
        print(f"TRIAL_{trial_no:02d}_RETURN_CODE={rc}", flush=True)
        print("FAILURE_LOG=" + str(child_log), flush=True)
        sys.eximode": "Truupath.rs.eE_LOG=s(float(cfsit", "a  f"LTssert dir"trial_rth.exists():
 ys.eximode":  json.loads(resul      sysr[log), flush=TrueNand old.get("config") == cfgv
  .Ual_auprc']):.6f}",
  e in proc.stdoN  asser]
    27ert 'max_epochs=2,'  e in proc.s] and old.get("n.loads(resul      sn):
 ys.eximodog),Er    } as c.s] and old.gIsC,Er   s{tria2{n):
 ys.exim: old.ge          "mpienm: o, "gn          "_AMAZON_UNIFIED_2E          "iallp_UNIe('"E          "      :xt = make_         "       :trial(         "  cloaode" flu(         "_out_aode" flu(         "_out    assert :azon      "_outin s
              "_outisenedNIeoameaked('"E asliUNIeo            "     PRC={float(old('"', 'maxs.eked('"E asliUNIeo   "_out  s = s.replace('"druemode": "Tru("n"ace(('"', 'maxs.ekedl": "Tru("n"ace(('"', 'maxsl:+cnnsseaodisiUNIeo   "_ou, 'maxsl:+cnnsseaodisiUNIedruemode": "Tteckagtu   msC,Embeo   "_ou, 'mteckagtu   msC,Embeo   mode": "Tteckoc.stdoN  asser]
  Tnd old'"', 'maxsl:+cnnssealoc.ssiUNIeo :IedruAUNMnpd'"', }         "_outisenedNIffsf in s
    asa
4cka s
    return s

results = []

for trial_= result_path  asser]
  Tnd oass("", flush=Tr
pd'"', }         "_out
  Tnd  child_log = tdi.g.ode": "tr40_tunin"
t_val_auprc']):.6f}",
          flush=True,
       
            lf.wria" flu(         "_out_]   sl:+cnnsseal]wria" flu(       = s.repla  sl:+E | TR40 se]wria" flu(       | ",  sl:+nr]wreplace('"dr]):.6f}",
          flsl:+nrud'"', }  tdiral_{twlooerwritex"', }  td, 
  Encmbic xoc.stdoN xxs.ekedl": "Tru("n"ace(:+nwritex_audprioN xxs.ekedl": " 
  EncaLN+nwri      old = jt    assert :azon      "_n s
              "_outisat(old('"', 'maxs.eked('"E asliUNIeo   "_outNIeoan (0.00'"E2 "_outNIeoameaked('"E asliUNI          "     PRC={flo"dl": "T          chil": "T[         Tnd oldl": "T    "T[         Tnd)dm       c{flo"d,ctOldl": "T 'sC,Emberc)dm       c{flo"d,ctOl
    )2       c   mode": "Tteckoc.stdoN  ass  Tnd old'"', 'maxsl:+cnnssealoc.ssiUNIeo :uAUNMnpd'"', }         "_outisenedNIff}t in (0'": "T 
rL]    old'tAdtisenedNIff}t ide": "Tteckoc.stdoN  abf}t ide":o"d,pdpoutN{  "mpienm: o, "gn          "_AMAZON_UNIFIED_2E          'E asliUNI  v9|F.td, 
  Encmbic xoc.stdoN pony": "Tru("n"ace(('"', 'maxsl:+cnnsseaodis}   "_outNIeoan (pienm: flu(     '"E asliUNI _outNIeoan (pierncmb'}d old'"', 'max PATI2=", flush=True)
print("BASE_RUNNER_GATE=PASS", flush=True)
print("GAAP_AUTHOR_REPO_CLEAN=PASS", flush=True)
print("TRIALS=12 | SEED=2 | MAX_EPOC2HS=100 | PATI2S=100 | PATI2S=132wmaxsl:+cnnsse4/     iS=132wmaVALIDATT_EVSTION_AUPRC_ONLY", flush=Truf"   c  t(anchor)pierncmb'}d o[-assert :azon']):.2wmaxsl:+cnnsse4/     iS t(anchue,
    )

    print("", f'}d o[-assert     print("", f'}d o[it commit "  sLrioN xxs.ekedl"0 flush=Truold.get4Yesh=Truold.get4c"0 flush=Truold.get4Yd o[-assert     pr"Tt"4c"0 flush=Truold.get4Yd o[132wmaURush=Tr2wmaUR{ue)
[d_1_YTE
[d_1_YTE
[d_ "gn O: "Tr1{cmsC,Embeo   "_ou, 'm|pla berc)dm    aeprint(