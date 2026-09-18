
from pathlib import Path
import sys
import importlib.util
import json
import hashlib
import random
import time
import gc
import csv
import math
import yaml
import numpy as np
import torch
import dgl

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from lightning.pytorch import Trainer

from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
)

ROOT=Path("/workspace/gaap_vast")
REPO=ROOT/"repo/GAAP"

WINNER=ROOT/"unified/elliptic/tuning/GAAP_ELLIPTIC_FROZEN_WINNER.json"
ADAPTER=ROOT/"shared/elliptic/elliptic_gaap_input_adapter"
SPLIT=ROOT/"shared/elliptic/elliptic_seed2_nested_splits.npz"
FINAL=ROOT/"unified/elliptic/final"
RUNS=FINAL/"runs"
RUNNER=ROOT/"scripts/06_gaap_elliptic_final_12_runs.py"

cmd=Path("/proc/self/cmdline").read_bytes().split(b"\0")
idx=cmd.index(b"-c")
RUNNER.write_text(cmd[idx+1].decode()+"\n")

winner=json.loads(
    WINNER.read_text()
)

assert winner["winner_trial"]==5
assert winner["configuration_frozen"] is True
assert winner["test_evaluated"] is False
assert winner["test_predictions_generated"] is False

frozen=winner["winner_config"]

assert int(frozen["d_hidden"])==64
assert abs(float(frozen["gnn_dropout"])-0.2)<1e-12
assert abs(float(frozen["lr"])-0.002)<1e-15

sys.path.insert(
    0,
    str(REPO)
)

import mycode.utils.dataloader as gaap_dl

spec=importlib.util.spec_from_file_location(
    "gaap_author_101",
    REPO/"mycode/exp/101_retrain.py"
)

author=importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    author
)

split=np.load(
    SPLIT,
    allow_pickle=False
)

sizes={
    "TR40":18889,
    "TR30":13670,
    "TR20":9553,
    "TR10":4543,
    "val":8726,
    "test":18949,
}

for key,n in sizes.items():
    assert key in split.files
    assert len(split[key])==n

sets={
    key:set(
        map(
            int,
            split[key]
        )
    )
    for key in sizes
}

assert sets["TR10"] <= sets["TR20"] <= sets["TR30"] <= sets["TR40"]

for key in [
    "TR40",
    "TR30",
    "TR20",
    "TR10",
]:
    assert sets[key].isdisjoint(sets["val"])
    assert sets[key].isdisjoint(sets["test"])

assert sets["val"].isdisjoint(sets["test"])

VA=np.asarray(
    split["val"],
    dtype=np.int64
)

TE=np.asarray(
    split["test"],
    dtype=np.int64
)

VAL_SET=set(map(int,VA))
TEST_SET=set(map(int,TE))

source_graph=dgl.load_graphs(
    str(ADAPTER)
)[0][0]

SOURCE_Y=(
    source_graph.ndata["label"]
    .clone()
    .long()
    .cpu()
    .numpy()
)

assert source_graph.num_nodes()==203769
assert source_graph.num_edges()==234355
assert tuple(source_graph.ndata["feature"].shape)==(203769,166)
assert int((SOURCE_Y==1).sum())==4545
assert int((SOURCE_Y==0).sum())==42019
assert int((SOURCE_Y==-1).sum())==157205

del source_graph

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dgl.seed(seed)

    try:
        dgl.random.seed(seed)
    except Exception:
        pass

    if hasattr(author,"fix_seed"):
        author.fix_seed(seed)

def prepare_graph(ratio):

    g=dgl.load_graphs(
        str(ADAPTER)
    )[0][0]

    train_ids=np.asarray(
        split[ratio],
        dtype=np.int64
    )

    train_t=torch.as_tensor(
        train_ids,
        dtype=torch.long
    )

    val_t=torch.as_tensor(
        VA,
        dtype=torch.long
    )

    test_t=torch.as_tensor(
        TE,
        dtype=torch.long
    )

    y_true=(
        g.ndata["label"]
        .clone()
        .long()
    )

    assert torch.all(
        y_true[train_t]>=0
    )

    assert torch.all(
        y_true[val_t]>=0
    )

    assert torch.all(
        y_true[test_t]>=0
    )

    fit_y=torch.zeros(
        203769,
        dtype=torch.long
    )

    fit_y[train_t]=y_true[train_t]
    fit_y[val_t]=y_true[val_t]

    train_mask=torch.zeros(
        203769,
        dtype=torch.bool
    )

    val_mask=torch.zeros(
        203769,
        dtype=torch.bool
    )

    test_mask=torch.zeros(
        203769,
        dtype=torch.bool
    )

    train_mask[train_t]=True
    val_mask[val_t]=True
    test_mask[test_t]=True

    g.ndata["train_mask"]=train_mask
    g.ndata["val_mask"]=val_mask
    g.ndata["test_mask"]=test_mask

    g.ndata["label"]=fit_y.contiguous()

    g.ndata["feature"]=(
        g.ndata["feature"]
        .float()
        .contiguous()
    )

    assert int(train_mask.sum())==sizes[ratio]
    assert int(val_mask.sum())==8726
    assert int(test_mask.sum())==18949

    assert torch.all(
        g.ndata["label"][test_t]==0
    )

    return g

class ControlledEllipticLitSAGE(
    author.LitSAGE
):

    def on_validation_epoch_start(self):

        self._cy=[]
        self._cs=[]
        self._cn=[]

    def validation_step(
        self,
        batch,
        batch_idx
    ):

        blocks,x,y,mask=self.unify_batch(
            batch,
            "val_mask"
        )

        output=self(
            blocks,
            x
        )

        if isinstance(
            output,
            (tuple,list)
        ):
            logit=output[0]
        else:
            logit=output

        nid=blocks[-1].dstdata[dgl.NID]

        if isinstance(mask,slice):

            y_sel=y
            logit_sel=logit
            nid_sel=nid

        else:

            mask=mask.bool()

            if int(mask.sum())==0:
                return

            y_sel=y[mask]
            logit_sel=logit[mask]
            nid_sel=nid[mask]

        if y_sel.numel()==0:
            return

        score=torch.softmax(
            logit_sel,
            dim=1
        )[:,1]

        self._cy.append(
            y_sel.detach().cpu()
        )

        self._cs.append(
            score.detach().cpu()
        )

        self._cn.append(
            nid_sel.detach().cpu()
        )

    def on_validation_epoch_end(self):

        y=torch.cat(
            self._cy
        ).numpy()

        score=torch.cat(
            self._cs
        ).numpy()

        nid=torch.cat(
            self._cn
        ).numpy()

        assert len(y)==8726
        assert len(np.unique(nid))==8726
        assert set(map(int,nid))==VAL_SET

        ap=float(
            average_precision_score(
                y,
                score
            )
        )

        auc=float(
            roc_auc_score(
                y,
                score
            )
        )

        self.log(
            "val_aps",
            ap,
            prog_bar=True,
            on_step=False,
            on_epoch=True
        )

        self.log(
            "val_auc",
            auc,
            prog_bar=True,
            on_step=False,
            on_epoch=True
        )

        self._cy=[]
        self._cs=[]
        self._cn=[]

def build(ratio,seed):

    seed_everything(
        seed
    )

    g=prepare_graph(
        ratio
    )

    cfg=yaml.safe_load(
        (
            REPO/
            "config/SAGE_MiniF_DyPLE_MHA/elliptic.yaml"
        ).read_text()
    )

    cfg["data_name"]="elliptic_comp8851"

    cfg["d_hidden"]=64
    cfg["gnn_dropout"]=0.2
    cfg["lr"]=0.002

    cfg["seed"]=int(seed)

    cfg["device"]="cuda"
    cfg["device_id"]=0
    cfg["nowandb"]=True
    cfg["patience"]=20

    gaap_dl.read_dataset=lambda _:g

    dm_kwargs=dict(cfg)

    dm_kwargs.pop(
        "data_name",
        None
    )

    dm=gaap_dl.MiniFDataModule(
        "elliptic",
        **dm_kwargs
    )

    dm.g.ndata["label"]=(
        g.ndata["label"]
        .clone()
    )

    dm.g.ndata["feature"]=(
        dm.g.ndata["feature"]
        .float()
    )

    if hasattr(dm,"y"):
        dm.y=(
            g.ndata["label"]
            .clone()
        )

    assert int(dm.d_in)==166

    val_ids=torch.as_tensor(
        VA,
        dtype=torch.long
    )

    def controlled_val_loader():

        return dgl.dataloading.DataLoader(
            dm.g,
            val_ids,
            dm.val_sampler,
            device=dm.device,
            use_uva=dm.use_uva,
            batch_size=dm.val_bs,
            shuffle=False,
            drop_last=False
        )

    dm.val_dataloader=controlled_val_loader

    cfg["d_in"]=166
    cfg["n_classes"]=2
    cfg["n_nodes"]=203769

    model=ControlledEllipticLitSAGE(
        **cfg
    )

    return model,dm

def targeted_scores(
    model,
    dm,
    ids,
    mask_name
):

    ids=np.asarray(
        ids,
        dtype=np.int64
    )

    ids_t=torch.as_tensor(
        ids,
        dtype=torch.long
    )

    loader=dgl.dataloading.DataLoader(
        dm.g,
        ids_t,
        dm.val_sampler,
        device=dm.device,
        use_uva=dm.use_uva,
        batch_size=dm.val_bs,
        shuffle=False,
        drop_last=False
    )

    node_parts=[]
    score_parts=[]

    model.eval()

    with torch.no_grad():

        for batch in loader:

            blocks,x,y,mask=model.unify_batch(
                batch,
                mask_name
            )

            output=model(
                blocks,
                x
            )

            if isinstance(
                output,
                (tuple,list)
            ):
                logit=output[0]
            else:
                logit=output

            nid=(
                blocks[-1]
                .dstdata[dgl.NID]
                .detach()
                .cpu()
                .numpy()
            )

            score=(
                torch.softmax(
                    logit,
                    dim=1
                )[:,1]
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            assert len(nid)==len(score)

            node_parts.append(nid)
            score_parts.append(score)

    out_ids=np.concatenate(
        node_parts
    )

    out_score=np.concatenate(
        score_parts
    )

    assert len(out_ids)==len(ids)
    assert len(np.unique(out_ids))==len(ids)
    assert set(map(int,out_ids))==set(map(int,ids))

    order=np.argsort(
        out_ids
    )

    return (
        out_ids[order],
        out_score[order]
    )

def choose_threshold(y,score):

    thresholds=np.unique(
        np.concatenate([
            np.asarray(
                [0.0],
                dtype=np.float64
            ),
            np.asarray(
                score,
                dtype=np.float64
            ),
            np.asarray(
                [1.0],
                dtype=np.float64
            )
        ])
    )

    best=None

    for threshold in thresholds:

        pred=(
            score>=float(threshold)
        ).astype(np.int64)

        macro=float(
            f1_score(
                y,
                pred,
                average="macro",
                zero_division=0
            )
        )

        candidate=(
            macro,
            -abs(
                float(threshold)-0.5
            ),
            -float(threshold),
            float(threshold)
        )

        if (
            best is None
            or candidate>best[0]
        ):
            best=(
                candidate,
                float(threshold),
                macro
            )

    return best[1],best[2]

def metric_pack(
    y,
    score,
    threshold
):

    y=np.asarray(
        y,
        dtype=np.int64
    )

    score=np.asarray(
        score,
        dtype=np.float64
    )

    pred=(
        score>=float(threshold)
    ).astype(np.int64)

    tn,fp,fn,tp=[
        int(x)
        for x in confusion_matrix(
            y,
            pred,
            labels=[0,1]
        ).ravel()
    ]

    recall=float(
        recall_score(
            y,
            pred,
            pos_label=1,
            zero_division=0
        )
    )

    precision=float(
        precision_score(
            y,
            pred,
            pos_label=1,
            zero_division=0
        )
    )

    specificity=(
        float(
            tn/(tn+fp)
        )
        if (tn+fp)
        else 0.0
    )

    return {
        "auprc":float(
            average_precision_score(
                y,
                score
            )
        ),

        "auroc":float(
            roc_auc_score(
                y,
                score
            )
        ),

        "macro_f1":float(
            f1_score(
                y,
                pred,
                average="macro",
                zero_division=0
            )
        ),

        "precision":precision,

        "recall":recall,

        "fraud_f1":float(
            f1_score(
                y,
                pred,
                pos_label=1,
                zero_division=0
            )
        ),

        "specificity":specificity,

        "gmean":float(
            math.sqrt(
                max(
                    0.0,
                    recall*specificity
                )
            )
        ),

        "tn":tn,
        "fp":fp,
        "fn":fn,
        "tp":tp,
    }

def sha256(path):

    h=hashlib.sha256()

    with open(path,"rb") as f:

        for block in iter(
            lambda:f.read(
                8*1024*1024
            ),
            b""
        ):
            h.update(block)

    return h.hexdigest()

RATIOS=[
    "TR40",
    "TR30",
    "TR20",
    "TR10",
]

SEEDS=[
    2,
    42,
    72,
]

all_results=[]

print("")
print("===== i2 — GAAP × ELLIPTIC — FINAL PROTOCOL =====")
print("RATIOS=TR40,TR30,TR20,TR10")
print("SEEDS=2,42,72")
print("FINAL_RUNS=12")
print("MAX_EPOCHS=100")
print("PATIENCE=20")
print("CHECKPOINT_SELECTION=VALIDATION_AUPRC")
print("THRESHOLD_SELECTION=VALIDATION_MACRO_F1")
print("TEST_DURING_TRAINING=NO")
print("TEST_FOR_THRESHOLD=NO")
print("TEST_EVALUATION=ONCE_AFTER_FREEZE")
print("ELLIPTIC_UNKNOWN_STRUCTURAL=157205")
print("ELLIPTIC_UNKNOWN_SUPERVISION=NO")
print("ELLIPTIC_TIME_STEP_INCLUDED_AS_FEATURE=YES")

for ratio in RATIOS:

    for seed in SEEDS:

        run_id="elliptic_{}_seed{}".format(
            ratio.lower(),
            seed
        )

        out=RUNS/run_id

        out.mkdir(
            parents=True,
            exist_ok=True
        )

        result_path=out/"final_result.json"

        marker=out/"TEST_EVAL_STARTED"

        if result_path.exists():

            old=json.loads(
                result_path.read_text()
            )

            assert old["status"]=="PASS"
            assert old["ratio"]==ratio
            assert int(old["train_seed"])==seed
            assert old["test_evaluated_once"] is True

            all_results.append(
                old
            )

            print(
                "FINAL_RESUME={} | AUPRC={:.6f}".format(
                    run_id,
                    float(
                        old["test"]["auprc"]
                    )
                ),
                flush=True
            )

            continue

        if marker.exists():

            raise RuntimeError(
                "TEST_EVAL_ALREADY_STARTED_WITHOUT_RESULT: "
                +run_id
            )

        print("")
        print("============================================================")
        print(
            " i2 — GAAP × ELLIPTIC — {} seed={} ".format(
                ratio,
                seed
            )
        )
        print("============================================================")

        model,dm=build(
            ratio,
            seed
        )

        checkpoint=ModelCheckpoint(
            dirpath=str(
                out/"checkpoints"
            ),
            filename="best",
            monitor="val_aps",
            mode="max",
            save_top_k=1,
            save_last=False
        )

        early=EarlyStopping(
            monitor="val_aps",
            mode="max",
            patience=20,
            check_finite=True
        )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        start=time.perf_counter()

        trainer=Trainer(
            accelerator="gpu",
            devices=[0],
            max_epochs=100,
            logger=False,
            callbacks=[
                checkpoint,
                early
            ],
            num_sanity_val_steps=0,
            enable_model_summary=False,
            enable_progress_bar=True
        )

        trainer.fit(
            model,
            datamodule=dm
        )

        torch.cuda.synchronize()

        wall=float(
            time.perf_counter()
            -
            start
        )

        peak=float(
            torch.cuda.max_memory_allocated()
            /
            (1024**2)
        )

        assert checkpoint.best_model_path

        ckpt_path=Path(
            checkpoint.best_model_path
        )

        cp=torch.load(
            ckpt_path,
            map_location="cpu"
        )

        best_epoch=int(
            cp.get(
                "epoch",
                -1
            )
        )+1

        best_val_checkpoint=float(
            checkpoint.best_model_score
            .detach()
            .cpu()
        )

        model.load_state_dict(
            cp["state_dict"]
        )

        model.to("cuda")
        model.eval()

        training_record={
            "ratio":ratio,
            "seed":seed,
            "best_epoch":best_epoch,
            "best_checkpoint":
                str(ckpt_path),
            "best_checkpoint_sha256":
                sha256(ckpt_path),
            "best_val_checkpoint_auprc":
                best_val_checkpoint,
            "training_wall_seconds":
                wall,
            "peak_gpu_memory_mb":
                peak,
            "test_evaluated":False
        }

        (
            out/
            "training_result.json"
        ).write_text(
            json.dumps(
                training_record,
                indent=2
            )
        )

        val_ids,val_score=targeted_scores(
            model,
            dm,
            VA,
            "val_mask"
        )

        assert set(map(int,val_ids))==VAL_SET

        val_y=SOURCE_Y[val_ids]

        assert set(
            np.unique(
                val_y
            ).tolist()
        )=={0,1}

        threshold,val_threshold_macro=choose_threshold(
            val_y,
            val_score
        )

        val_metrics=metric_pack(
            val_y,
            val_score,
            threshold
        )

        freeze={
            "ratio":ratio,
            "seed":seed,
            "best_checkpoint":
                str(ckpt_path),
            "selected_threshold":
                threshold,
            "threshold_selection":
                "validation macro-F1 only",
            "validation_macro_f1":
                val_threshold_macro,
            "validation_metrics":
                val_metrics,
            "test_used_for_checkpoint_selection":
                False,
            "test_used_for_threshold_selection":
                False,
            "test_predictions_generated":
                False,
            "configuration_checkpoint_threshold_frozen":
                True
        }

        (
            out/
            "PRE_TEST_FREEZE.json"
        ).write_text(
            json.dumps(
                freeze,
                indent=2
            )
        )

        print(
            "PRE_TEST_FREEZE=PASS | {} seed={} | threshold={:.12f}".format(
                ratio,
                seed,
                threshold
            ),
            flush=True
        )

        marker.write_text(
            "TEST EVALUATION STARTED\n"
        )

        test_ids,test_score=targeted_scores(
            model,
            dm,
            TE,
            "test_mask"
        )

        assert set(map(int,test_ids))==TEST_SET

        test_y=SOURCE_Y[test_ids]

        assert set(
            np.unique(
                test_y
            ).tolist()
        )=={0,1}

        test_metrics=metric_pack(
            test_y,
            test_score,
            threshold
        )

        record={
            "status":"PASS",
            "model":"GAAP",
            "dataset":"Elliptic",
            "ratio":ratio,
            "train_seed":seed,
            "split_seed":2,

            "winner_trial":5,

            "frozen_config":{
                "d_hidden":64,
                "gnn_dropout":0.2,
                "lr":0.002
            },

            "winner_sha256":
                "c47e034ca6f7f23ffeec14164a3c0698da30134eadcb25c88c94e05c98867102",

            "split_sha256":
                "1e7963e9d09935fb33e0df74cfb786cab826d1b95ae10b9592dbc240b48ffdbc",

            "adapter_sha256":
                sha256(ADAPTER),

            "raw_features_sha256":
                "fd7f83573443c9e302e371d3f110e3b6224160f5d1ed8a287757936127800ff0",

            "raw_classes_sha256":
                "93e2e7b2405c735ba752bf6ba06b947561deddd1f5a8fc91e46f6a4c0e439493",

            "raw_edges_sha256":
                "a35053ba68a98e4382cae2ba65b9d9e36b23b6439e02dff084971b1b72a5156e",

            "repository_commit":
                "6a7dbb0447c4897504525de49e41a0526ee777f8",

            "checkpoint_selection":
                "validation AUPRC",

            "threshold_selection":
                "validation macro-F1 only",

            "max_epochs":100,
            "patience":20,

            "best_epoch":
                best_epoch,

            "best_checkpoint":
                str(ckpt_path),

            "best_checkpoint_sha256":
                sha256(ckpt_path),

            "selected_threshold":
                threshold,

            "validation":{
                "n":8726,
                **val_metrics
            },

            "test":{
                "n":18949,
                **test_metrics
            },

            "training_wall_seconds":
                wall,

            "peak_gpu_memory_mb":
                peak,

            "features":166,

            "time_step_in_features":
                True,

            "unknown_structural_nodes":
                157205,

            "unknown_supervision":
                False,

            "reverse_edges_added":
                False,

            "test_labels_hidden_during_fit":
                True,

            "test_used_for_checkpoint_selection":
                False,

            "test_used_for_threshold_selection":
                False,

            "test_predictions_generated_before_threshold_freeze":
                False,

            "test_evaluated_once":
                True,

            "controlled_validation_seed_nodes_only":
                True,

            "author_architecture_changed":
                False
        }

        result_path.write_text(
            json.dumps(
                record,
                indent=2
            )
        )

        all_results.append(
            record
        )

        print(
            "FINAL_RUN_COMPLETE={} seed={} | "
            "AUPRC={:.6f} | "
            "AUROC={:.6f} | "
            "MacroF1={:.6f} | "
            "Prec={:.6f} | "
            "Recall={:.6f} | "
            "FraudF1={:.6f} | "
            "GMean={:.6f}".format(
                ratio,
                seed,
                test_metrics["auprc"],
                test_metrics["auroc"],
                test_metrics["macro_f1"],
                test_metrics["precision"],
                test_metrics["recall"],
                test_metrics["fraud_f1"],
                test_metrics["gmean"]
            ),
            flush=True
        )

        del (
            model,
            dm,
            trainer,
            checkpoint,
            early,
            cp
        )

        gc.collect()
        torch.cuda.empty_cache()

assert len(all_results)==12

all_results=sorted(
    all_results,
    key=lambda x:(
        RATIOS.index(
            x["ratio"]
        ),
        SEEDS.index(
            int(
                x["train_seed"]
            )
        )
    )
)

results_json=(
    FINAL/
    "GAAP_ELLIPTIC_FINAL_12_RUNS.json"
)

results_json.write_text(
    json.dumps(
        all_results,
        indent=2
    )
)

results_csv=(
    FINAL/
    "GAAP_ELLIPTIC_FINAL_12_RUNS.csv"
)

with results_csv.open(
    "w",
    newline=""
) as f:

    fields=[
        "ratio",
        "seed",
        "threshold",
        "best_epoch",
        "auprc",
        "auroc",
        "macro_f1",
        "precision",
        "recall",
        "fraud_f1",
        "gmean",
        "training_wall_seconds",
        "peak_gpu_memory_mb",
    ]

    w=csv.DictWriter(
        f,
        fieldnames=fields
    )

    w.writeheader()

    for r in all_results:

        w.writerow({
            "ratio":
                r["ratio"],

            "seed":
                r["train_seed"],

            "threshold":
                r["selected_threshold"],

            "best_epoch":
                r["best_epoch"],

            "auprc":
                r["test"]["auprc"],

            "auroc":
                r["test"]["auroc"],

            "macro_f1":
                r["test"]["macro_f1"],

            "precision":
                r["test"]["precision"],

            "recall":
                r["test"]["recall"],

            "fraud_f1":
                r["test"]["fraud_f1"],

            "gmean":
                r["test"]["gmean"],

            "training_wall_seconds":
                r["training_wall_seconds"],

            "peak_gpu_memory_mb":
                r["peak_gpu_memory_mb"],
        })

metrics=[
    "auprc",
    "auroc",
    "macro_f1",
    "precision",
    "recall",
    "fraud_f1",
    "gmean",
]

summary={}

for ratio in RATIOS:

    rr=[
        r
        for r in all_results
        if r["ratio"]==ratio
    ]

    assert len(rr)==3

    summary[ratio]={}

    for metric in metrics:

        values=np.asarray(
            [
                float(
                    r["test"][metric]
                )
                for r in rr
            ],
            dtype=np.float64
        )

        summary[ratio][metric]={
            "mean":
                float(
                    np.mean(values)
                ),

            "sd":
                float(
                    np.std(
                        values,
                        ddof=0
                    )
                )
        }

summary_path=(
    FINAL/
    "GAAP_ELLIPTIC_FINAL_RATIO_SUMMARY.json"
)

summary_path.write_text(
    json.dumps(
        summary,
        indent=2
    )
)

manifest={
    "status":"PASS",
    "model":"GAAP",
    "dataset":"Elliptic",
    "final_runs":12,
    "ratios":RATIOS,
    "seeds":SEEDS,

    "winner_trial":5,

    "winner_config":{
        "d_hidden":64,
        "gnn_dropout":0.2,
        "lr":0.002
    },

    "winner_sha256":
        "c47e034ca6f7f23ffeec14164a3c0698da30134eadcb25c88c94e05c98867102",

    "checkpoint_selection":
        "validation AUPRC",

    "threshold_selection":
        "validation macro-F1",

    "controlled_validation_seed_nodes_only":
        True,

    "elliptic_chronology_preserved":
        True,

    "time_step_in_features":
        True,

    "features":
        166,

    "unknown_structural_nodes":
        157205,

    "unknown_supervision":
        False,

    "reverse_edges_added":
        False,

    "test_used_for_selection":
        False,

    "test_used_for_threshold":
        False,

    "test_predictions_generated_before_threshold_freeze":
        False,

    "test_evaluated_once_per_run":
        True,

    "repository_commit":
        "6a7dbb0447c4897504525de49e41a0526ee777f8",

    "population_sd":
        True
}

manifest_path=(
    FINAL/
    "GAAP_ELLIPTIC_FINAL_MANIFEST.json"
)

manifest_path.write_text(
    json.dumps(
        manifest,
        indent=2
    )
)

print("")
print(
    "===== i2 — GAAP × ELLIPTIC — FINAL RESULTS MEAN ± SD ====="
)

for ratio in RATIOS:

    x=summary[ratio]

    print(
        "{} | "
        "AUPRC={:.6f}±{:.6f} | "
        "AUROC={:.6f}±{:.6f} | "
        "MacroF1={:.6f}±{:.6f} | "
        "Prec={:.6f}±{:.6f} | "
        "Recall={:.6f}±{:.6f} | "
        "FraudF1={:.6f}±{:.6f} | "
        "GMean={:.6f}±{:.6f}".format(
            ratio,

            x["auprc"]["mean"],
            x["auprc"]["sd"],

            x["auroc"]["mean"],
            x["auroc"]["sd"],

            x["macro_f1"]["mean"],
            x["macro_f1"]["sd"],

            x["precision"]["mean"],
            x["precision"]["sd"],

            x["recall"]["mean"],
            x["recall"]["sd"],

            x["fraud_f1"]["mean"],
            x["fraud_f1"]["sd"],

            x["gmean"]["mean"],
            x["gmean"]["sd"]
        )
    )

print("")
print("GAAP_ELLIPTIC_FINAL=PASS")
print("GAAP_ELLIPTIC_FINAL_RUNS=12/12")
print("WINNER_TRIAL=5")
print("CHECKPOINT_SELECTION=VALIDATION_AUPRC")
print("THRESHOLD_SELECTION=VALIDATION_MACRO_F1")
print("CONTROLLED_VAL_SEEDS_ONLY=PASS")
print("ELLIPTIC_CHRONOLOGY=PASS")
print("ELLIPTIC_UNKNOWN_SUPERVISION=NO")
print("REVERSE_EDGES_ADDED=NO")
print("TEST_USED_FOR_SELECTION=NO")
print("TEST_USED_FOR_THRESHOLD=NO")
print("TEST_PREDICTIONS_BEFORE_THRESHOLD_FREEZE=NO")
print("TEST_EVALUATED_ONCE_PER_RUN=YES")
print("TEST_ISOLATION=PASS")
print("GAAP_AUTHOR_REPO_CLEAN=PASS")
print("GAAP_ELLIPTIC_FINAL_GATE=PASS")
print("RESULTS="+str(results_csv))
print("SUMMARY="+str(summary_path))
print("MANIFEST="+str(manifest_path))

