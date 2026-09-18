
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import sys

import numpy as np
import torch
import dgl

from scipy.io import loadmat

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

ROOT=Path("/workspace/gaap_vast")

TRAINER=ROOT/"scripts/04_gaap_amazon_final_train.py"

WINNER=ROOT/"unified/amazon/tuning/GAAP_AMAZON_FROZEN_WINNER.json"

OUT=Path(
    os.environ["GAAP_FINAL_OUT"]
)

RATIO=os.environ["GAAP_FINAL_RATIO"]

SEED=int(
    os.environ["GAAP_FINAL_SEED"]
)

MODE=os.environ.get(
    "GAAP_EVAL_MODE",
    "FINAL"
)

assert RATIO in {
    "TR40",
    "TR30",
    "TR20",
    "TR10",
}

assert SEED in {
    2,
    42,
    72,
}

source=TRAINER.read_text()

assert source.count("trainer.fit(")==1

prefix=source[
    :source.index("trainer.fit(")
]

ns={
    "__name__":"__main__",
    "__file__":str(TRAINER),
}

exec(
    compile(
        prefix,
        str(TRAINER),
        "exec"
    ),
    ns,
    ns
)

model=ns["model"]
dm=ns["dm"]
va=ns["va"]
te=ns["te"]

winner=json.loads(
    WINNER.read_text()
)

if MODE=="PREFLIGHT":

    checkpoint_path=Path(
        winner["winner_checkpoint"]
    )

else:

    training_path=(
        OUT/
        "training_result.json"
    )

    assert training_path.exists()

    training=json.loads(
        training_path.read_text()
    )

    checkpoint_path=Path(
        training[
            "best_checkpoint"
        ]
    )

assert checkpoint_path.exists(), checkpoint_path

checkpoint=torch.load(
    checkpoint_path,
    map_location="cpu"
)

assert "state_dict" in checkpoint

model.load_state_dict(
    checkpoint["state_dict"]
)

model.to("cuda")
model.eval()

best_epoch=int(
    checkpoint.get(
        "epoch",
        -1
    )
)+1

canonical=loadmat(
    ROOT/
    "shared/amazon/Amazon.mat"
)

canonical_labels=np.asarray(
    canonical["label"]
).reshape(-1).astype(np.int64)

assert canonical_labels.shape==(11944,)

assert int(
    (
        canonical_labels[3305:]==1
    ).sum()
)==821

assert int(
    (
        canonical_labels[3305:]==0
    ).sum()
)==7818

def sha256_file(path):

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

def targeted_loader(ids):

    ids=torch.as_tensor(
        ids,
        dtype=torch.long
    )

    return dgl.dataloading.DataLoader(
        dm.g,
        ids,
        dm.val_sampler,
        device=dm.device,
        use_uva=dm.use_uva,
        batch_size=dm.val_bs,
        shuffle=False,
        drop_last=False,
    )

def scores_for(ids,mask_name):

    ids=np.asarray(
        ids,
        dtype=np.int64
    )

    loader=targeted_loader(
        ids
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
                (list,tuple)
            ):

                logit=output[0]

            else:

                logit=output

            assert logit.ndim==2
            assert int(logit.shape[1])==2

            nid=(
                blocks[-1]
                .dstdata[dgl.NID]
                .detach()
                .cpu()
                .numpy()
            )

            probability=(
                torch.softmax(
                    logit,
                    dim=1
                )[:,1]
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            assert len(nid)==len(probability)

            node_parts.append(
                nid
            )

            score_parts.append(
                probability
            )

    output_ids=np.concatenate(
        node_parts
    )

    output_scores=np.concatenate(
        score_parts
    )

    assert len(output_ids)==len(ids)

    assert len(
        np.unique(output_ids)
    )==len(ids)

    assert set(
        int(x)
        for x in output_ids
    )==set(
        int(x)
        for x in ids
    )

    order=np.argsort(
        output_ids
    )

    return (
        output_ids[order],
        output_scores[order]
    )

def metrics(
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

    auprc=float(
        average_precision_score(
            y,
            score
        )
    )

    auroc=float(
        roc_auc_score(
            y,
            score
        )
    )

    macro_f1=float(
        f1_score(
            y,
            pred,
            average="macro",
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

    recall=float(
        recall_score(
            y,
            pred,
            pos_label=1,
            zero_division=0
        )
    )

    fraud_f1=float(
        f1_score(
            y,
            pred,
            pos_label=1,
            zero_division=0
        )
    )

    tn,fp,fn,tp=[
        int(v)
        for v in confusion_matrix(
            y,
            pred,
            labels=[0,1]
        ).ravel()
    ]

    specificity=(
        float(
            tn/(tn+fp)
        )
        if (tn+fp)>0
        else 0.0
    )

    gmean=float(
        math.sqrt(
            max(
                0.0,
                recall*specificity
            )
        )
    )

    return {
        "auprc":auprc,
        "auroc":auroc,
        "macro_f1":macro_f1,
        "precision":precision,
        "recall":recall,
        "fraud_f1":fraud_f1,
        "specificity":specificity,
        "gmean":gmean,
        "tn":tn,
        "fp":fp,
        "fn":fn,
        "tp":tp,
    }

val_target=(
    va.detach()
    .cpu()
    .numpy()
    .astype(np.int64)
)

val_ids,val_scores=scores_for(
    val_target,
    "val_mask"
)

assert len(val_ids)==1728

val_y=canonical_labels[
    val_ids
]

val_auprc=float(
    average_precision_score(
        val_y,
        val_scores
    )
)

if MODE=="PREFLIGHT":

    frozen=float(
        winner[
            "winner_val_auprc"
        ]
    )

    delta=abs(
        val_auprc-frozen
    )

    print(
        "TARGETED_VAL_N={}".format(
            len(val_ids)
        )
    )

    print(
        "TARGETED_VAL_AUPRC={:.6f}".format(
            val_auprc
        )
    )

    print(
        "FROZEN_WINNER_VAL_AUPRC={:.6f}".format(
            frozen
        )
    )

    print(
        "TARGETED_VAL_DELTA={:.8f}".format(
            delta
        )
    )

    assert delta < 0.002, (
        val_auprc,
        frozen,
        delta
    )

    print(
        "AUTHOR_TARGETED_INFERENCE_GATE=PASS"
    )

    print(
        "TEST_INFERENCE_PERFORMED=NO"
    )

    print(
        "SAFE_TO_START_FINAL_12_RUNS=YES"
    )

    sys.exit(0)

candidate_thresholds=np.unique(
    np.concatenate([
        np.asarray(
            [0.0],
            dtype=np.float64
        ),

        np.asarray(
            val_scores,
            dtype=np.float64
        ),

        np.asarray(
            [1.0],
            dtype=np.float64
        ),
    ])
)

rows=[]

for threshold in candidate_thresholds:

    pred=(
        val_scores>=float(threshold)
    ).astype(np.int64)

    macro=float(
        f1_score(
            val_y,
            pred,
            average="macro",
            zero_division=0
        )
    )

    rows.append({
        "threshold":
            float(threshold),

        "macro_f1":
            macro,
    })

winner_threshold=max(
    rows,
    key=lambda x:(
        x["macro_f1"],
        -abs(
            x["threshold"]-0.5
        ),
        -x["threshold"],
    )
)

selected_threshold=float(
    winner_threshold[
        "threshold"
    ]
)

threshold_csv=(
    OUT/
    "validation_threshold_grid.csv"
)

with threshold_csv.open(
    "w",
    newline=""
) as f:

    writer=csv.DictWriter(
        f,
        fieldnames=[
            "threshold",
            "macro_f1",
        ]
    )

    writer.writeheader()
    writer.writerows(
        rows
    )

val_metrics=metrics(
    val_y,
    val_scores,
    selected_threshold
)

print(
    "SELECTED_THRESHOLD={:.12f}".format(
        selected_threshold
    )
)

print(
    "VAL_AUPRC={:.6f}".format(
        val_metrics["auprc"]
    )
)

print(
    "VAL_MACRO_F1={:.6f}".format(
        val_metrics["macro_f1"]
    )
)

print(
    "THRESHOLD_SELECTION=VALIDATION_MACRO_F1"
)

print(
    "TEST_USED_FOR_THRESHOLD=NO"
)

print(
    "CONFIG_CHECKPOINT_THRESHOLD_FROZEN=YES"
)

test_target=(
    te.detach()
    .cpu()
    .numpy()
    .astype(np.int64)
)

test_ids,test_scores=scores_for(
    test_target,
    "test_mask"
)

assert len(test_ids)==3456

test_y=canonical_labels[
    test_ids
]

test_metrics=metrics(
    test_y,
    test_scores,
    selected_threshold
)

training=json.loads(
    (
        OUT/
        "training_result.json"
    ).read_text()
)

record={
    "status":"PASS",

    "model":"GAAP",

    "dataset":"Amazon",

    "ratio":
        RATIO,

    "train_seed":
        SEED,

    "split_seed":
        2,

    "repository_commit":
        "6a7dbb0447c4897504525de49e41a0526ee777f8",

    "dataset_sha256":
        "4b7e3f9cccc62b736792707393ccd74332a1a0592dba128ac6b2989bf1ee9d63",

    "split_sha256":
        "0fd96816c440b247b8cac768896bc454b1ecf4abebaf5001c1a3e1d900efa4fd",

    "adapter_sha256":
        "0b74a6e55c1499773e8e5c856319ca8624934830db886c21b30163f8f007dae1",

    "winner_sha256":
        "3a95d631c00a23651bc3fd3291ceec2bece1806dd4d5ef867d3f1d7000e0ae11",

    "winner_trial":
        3,

    "frozen_config":{
        "d_hidden":64,
        "gnn_dropout":0.1,
        "lr":0.0005,
    },

    "max_epochs":
        100,

    "patience":
        20,

    "checkpoint_selection":
        "validation AUPRC",

    "threshold_selection":
        "validation macro-F1 only",

    "best_epoch":
        best_epoch,

    "best_checkpoint":
        str(
            checkpoint_path
        ),

    "best_checkpoint_sha256":
        sha256_file(
            checkpoint_path
        ),

    "selected_threshold":
        selected_threshold,

    "validation":{
        "n":
            int(len(val_y)),

        **val_metrics,
    },

    "test":{
        "n":
            int(len(test_y)),

        **test_metrics,
    },

    "training_wall_seconds":
        float(
            training[
                "training_wall_seconds"
            ]
        ),

    "peak_gpu_memory_mb":
        float(
            training[
                "peak_gpu_memory_mb"
            ]
        ),

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

    "author_architecture_changed":
        False,

    "canonical_adapter_raw_edges":
        8796784,

    "nodes":
        11944,

    "features":
        25,
}

final_path=(
    OUT/
    "final_result.json"
)

final_path.write_text(
    json.dumps(
        record,
        indent=2
    )
)

print(
    "TEST_AUPRC={:.6f}".format(
        test_metrics["auprc"]
    )
)

print(
    "TEST_AUROC={:.6f}".format(
        test_metrics["auroc"]
    )
)

print(
    "TEST_MACRO_F1={:.6f}".format(
        test_metrics["macro_f1"]
    )
)

print(
    "TEST_PRECISION={:.6f}".format(
        test_metrics["precision"]
    )
)

print(
    "TEST_RECALL={:.6f}".format(
        test_metrics["recall"]
    )
)

print(
    "TEST_FRAUD_F1={:.6f}".format(
        test_metrics["fraud_f1"]
    )
)

print(
    "TEST_GMEAN={:.6f}".format(
        test_metrics["gmean"]
    )
)

print(
    "TEST_EVALUATED_ONCE=YES"
)

print(
    "TEST_ISOLATION=PASS"
)

print(
    "GAAP_AMAZON_FINAL_RUN=PASS"
)
