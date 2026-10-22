#!/usr/bin/env python3
"""Read-only Stage6: offline audit saved T-Finance test scores and assemble public evidence.
Never calls model.forward, changes frozen runs, includes .pt checkpoints, or re-tests.
"""
from __future__ import annotations
import hashlib, importlib.util, io, json, os, subprocess, sys, traceback, zipfile
from pathlib import Path, PurePosixPath
import numpy as np
from dgl.data.utils import load_graphs
import torch

W=Path('/workspace');R=W/'comp8851'
S3=R/'results/eogfd/tfinance/stage3_tr40_tuning'
S4=R/'results/eogfd/tfinance/stage4_frozen_finals'
S5=R/'results/eogfd/tfinance/stage5_heldout_test_once'
KIT=W/'EOGFD_TFINANCE_FINALIZE_I1_20261023'
PREFIX='EOGFD/T-Finance/'
OUT=W/'EOGFD_TFINANCE_I1_GITHUB_PUBLICATION_20261023.zip'
CHECKSUM=Path(str(OUT)+'.sha256')
RATIOS=('TR40','TR30','TR20','TR10');SEEDS=(2,42,72)
ALLOWED={'.md','.json','.txt','.csv','.py','.sh','.log','.npy'}

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def require(cond,why):
    if not cond:raise RuntimeError('STOP_'+str(why))
def jread(p):return json.loads(Path(p).read_text())
def jbytes(x):return (json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()

def main():
    require(sys.argv[1:]==['--export-public'],'USE_--export-public')
    require(not OUT.exists() and not CHECKSUM.exists(),'PUBLIC_ZIP_ALREADY_EXISTS_DO_NOT_OVERWRITE')
    sp=KIT/'i1_stage5_once.py'
    require(sp.exists(),'MISSING_STAGE5_CODE')
    spec=importlib.util.spec_from_file_location('stage5_export_audit',sp)
    s5=importlib.util.module_from_spec(spec);sys.modules[spec.name]=s5;spec.loader.exec_module(s5)
    frozen,a=s5.audit_frozen()
    require(a['completed_finals']==12,'FROZEN_12')
    summary=jread(S5/'EOGFD_TFinance_Stage5_AllRatios_Summary.json')
    require(summary['status']=='EOGFD_TFINANCE_12_ONCE_ONLY_HELDOUT_TESTS_COMPLETE','STAGE5_COMPLETE')
    require(summary['test_evaluations']==12 and summary['test_inference_claims_per_final']==1,'ONCE_ONLY_COUNT')
    require(summary['winner_trial']==8 and summary['frozen_config']==frozen['winner_config'],'FROZEN_WINNER_MATCH')
    with np.load(s5.SPLIT,allow_pickle=False) as z:test_ix=np.asarray(z['test'],dtype=np.int64)
    graph,_=load_graphs(str(s5.SRC));raw=graph[0].ndata['label']
    sel=torch.from_numpy(test_ix.copy())
    y=(raw[sel].argmax(dim=1) if raw.shape==(39357,2) else raw[sel].view(-1)).long().numpy().copy()
    require(len(y)==15743 and 0<int(y.sum())<len(y),'TEST_LABELS')
    del graph,raw
    for ratio in RATIOS:
        group=[]
        for seed in SEEDS:
            rec=jread(S4/ratio/f'seed_{seed}'/'frozen_training.json')
            # This function uses only SAVED scores; never calls original model.
            group.append(s5.verify_finished(ratio,seed,rec,y))
        row=next(x for x in summary['metrics'] if x['ratio']==ratio)
        for key in ('auprc','auroc','macro_f1','fraud_f1','fraud_precision','fraud_recall','g_mean',
                    'accuracy','balanced_accuracy','mcc','brier_score'):
            z=np.asarray([r[key] for r in group],dtype=np.float64)
            require(abs(row[key+'_mean']-float(z.mean()))<1e-10,'SUMMARY_MEAN_'+key)
            require(abs(row[key+'_sample_sd']-float(z.std(ddof=1)))<1e-10,'SUMMARY_STD_'+key)
    files={}
    def add(rel,source):
        name=PREFIX+rel
        pos=PurePosixPath(name)
        require(not pos.is_absolute() and '..' not in pos.parts,'UNSAFE_PUBLIC_NAME')
        require(pos.suffix in ALLOWED,'FORBIDDEN_EXT_'+name)
        require(name not in files,'DUP_PUBLIC')
        p=Path(source)
        require(p.is_file() and not p.is_symlink() and p.stat().st_size<(24<<20),'BAD_PUBLIC_FILE_'+str(p))
        files[name]=p.read_bytes()
    def put(rel,contents):
        name=PREFIX+rel
        require(name not in files,'DUPLICATE_'+name)
        files[name]=contents if isinstance(contents,bytes) else contents.encode()
    # Real source scripts and local forensic audit; model weights excluded.
    for pub,path in (
        ('stage3_tfinance.py',W/'EOGFD_TFINANCE_STAGE3_I1_20261023/stage3_tfinance.py'),
        ('stage3_launcher_import_fixed.sh',W/'EOGFD_TFINANCE_STAGE3_I1_20261023/vast_i1_one_command.sh'),
        ('stage4_tfinance.py',W/'EOGFD_TFINANCE_STAGE4_I1_20261023/stage4_tfinance.py'),
        ('stage4_launcher.sh',W/'EOGFD_TFINANCE_STAGE4_I1_20261023/vast_i1_stage4.sh'),
        ('i1_pretest_audit_backup_v2.py',W/'EOGFD_TFINANCE_PRETEST_AUDIT_I1_V2_20261023/i1_pretest_audit_backup_v2.py'),
        ('vast_i1_audit_backup_v2.sh',W/'EOGFD_TFINANCE_PRETEST_AUDIT_I1_V2_20261023/vast_i1_audit_backup_v2.sh'),
        ('i1_stage5_once.py',KIT/'i1_stage5_once.py'),
        ('i1_stage6_public_export.py',KIT/'i1_stage6_public_export.py'),
        ('vast_i1_finalize.sh',KIT/'vast_i1_finalize.sh'),
        ('enable_eogfd_cuda11.sh',R/'bootstrap/enable_eogfd_cuda11.sh'),
    ):add('scripts/'+pub,path)
    for i in range(1,13):
        root=S3/f'trial_{i:02d}'
        for name in ('trial_result.json','validation_history.json','epoch_times.csv','best_validation_scores.npy'):
            add(f'results/tuning/trial_{i:02d}/'+name,root/name)
    for name in ('stage3_run_manifest.json','EOGFD_TFinance_TR40_FrozenWinner.json'):
        add('results/tuning/'+name,S3/name)
    for ratio in RATIOS:
        for seed in SEEDS:
            f=S4/ratio/f'seed_{seed}';t=S5/ratio/f'seed_{seed}'
            for name in ('frozen_training.json','validation_history.json','epoch_times.csv','best_validation_scores.npy'):
                add(f'results/finals/{ratio}/seed_{seed}/'+name,f/name)
            for name in ('test_once_claim.json','test_metrics.json','test_scores.npy'):
                add(f'results/finals/{ratio}/seed_{seed}/'+name,t/name)
    for name in ('stage4_plan.json','EOGFD_TFinance_Stage4_12FrozenFinalsSummary.json'):
        add('results/finals/'+name,S4/name)
    add('results/finals/EOGFD_TFinance_Stage5_AllRatios_Summary.json',S5/'EOGFD_TFinance_Stage5_AllRatios_Summary.json')
    add('evidence/EOGFD_TFinance_PreTest_Audit.json',R/'diagnostics/EOGFD_57212396_tfinance/pretest_audit/EOGFD_TFinance_PreTest_Audit.json')
    add('evidence/train_only_one_update.json',R/'diagnostics/EOGFD_57212396_tfinance/train_only_one_update.json')
    for logname in ('EOGFD_57212396_tfinance_PREPILOT.log','EOGFD_57212396_TFinance_STAGE3_TR40.log','EOGFD_57212396_TFinance_STAGE4.log','EOGFD_57212396_TFinance_STAGE5.log'):
        p=R/'logs'/logname
        if p.exists():add('evidence/logs/'+logname,p)
    provenance={'dataset':'T-Finance','instance_id':'57212396','hostname':'60d39211b1fd',
        'upstream_model':'EOGFDHeteroModel','upstream_commit':s5.PIN,
        'original_model_repository_unchanged':True,'source_sha256':s5.SOURCE_SHA,'split_sha256':s5.SPLIT_SHA,
        'frozen_split_seed':2,'training_seeds':list(SEEDS),'ratios':list(RATIOS),
        'fixed_eval_rng_seed':100921,'source_nodes':39357,'source_stored_directed_edges':42445086,
        'graph_relation_count':1,'model_implementation_mode':'UNIFIED_INPUT_ADAPTER_NOT_AUTHOR_IDENTICAL',
        'venus_methodological_approval_recorded':False,'once_only_inference_per_final':True,
        'fresh_training_in_publication_export':False,'extra_test_inference_in_publication_export':False,
        'private_model_checkpoints_exported_to_public_git':False,
        'external_private_backup':'EOGFD_TFINANCE_I1_PRETEST_BACKUP_20261023.tar.gz',
        'pretest_backup_sha256':'35852eff778177305e357577d88a2a65007e2b9dc269db2bb18538da402f1266'}
    put('evidence/environment_and_provenance.json',jbytes(provenance))
    def num(n):return f'{n:.6f}'
    rows=['| Training ratio | Test AUPRC | Test AUROC | Test macro-F1 |','|---|---:|---:|---:|']
    for r in summary['metrics']:
        rows.append('| '+r['ratio']+' | '+' | '.join(num(r[m+'_mean'])+' ± '+num(r[m+'_sample_sd']) for m in ('auprc','auroc','macro_f1'))+' |')
    lines=['| Ratio | Seed | Test AUPRC | Test AUROC | Test macro-F1 |','|---|---:|---:|---:|---:|']
    for ratio in RATIOS:
        for seed in SEEDS:
            r=jread(S5/ratio/f'seed_{seed}'/'test_metrics.json')
            lines.append(f'| {ratio} | {seed} | {r["auprc"]:.6f} | {r["auroc"]:.6f} | {r["macro_f1"]:.6f} |')
    readme=f'''# EOGFD × T-Finance — completed frozen-split benchmark

**Status: COMPLETE.** Validation-only TR40 tuning **12/12**, fresh frozen final training **12/12**, once-only held-out test evaluation **12/12**. These are the completed Assessment 2 frozen-split benchmark outputs, not an author-identical workflow reproduction.

## Source, data and protocol

- Original model: [BERA-wx/EOGFD](https://github.com/BERA-wx/EOGFD/tree/{s5.PIN}) at pinned commit `{s5.PIN}`; original `EOGFDHeteroModel` source remains untouched.
- Canonical T-Finance original graph: 39,357 nodes, 42,445,086 stored directed edges, 10 features, one original relation, two-column author labels converted with `argmax`.
- Canonical graph SHA-256: `{s5.SOURCE_SHA}`. Frozen split SHA-256: `{s5.SPLIT_SHA}`.
- Frozen split seed 2; training seeds 2, 42, 72; TR40/TR30/TR20/TR10 nested training subsets, fixed validation/test partitions.
- TR40 seed-2 tuning: 12 planned trials, selected by validation AUPRC. Winner: trial 8 (validation AUPRC {frozen['winner_val_auprc']:.6f}); configuration `{json.dumps(frozen['winner_config'],sort_keys=True)}`.
- Fresh final trainings: up to 100 epochs, patience 20. Best checkpoint from validation AUPRC. Classification threshold from validation macro-F1, tie-break fraud recall then nearest 0.5.
- Test protocol: one exclusive test-evaluation claim per final checkpoint, claims persisted **before** inference; exactly one test inference per checkpoint, no test-based checkpoint or threshold choice. Publication recomputes saved-score metrics offline only.
- Training class weights from each ratio's training labels; Adam beta1 0.9, beta2 0.999, eps 1e-8; evaluation RNG seed 100921.
- Scientific limitation: canonical frozen-split input adapter differs from upstream's original split/test workflow. This is **not** an exact author-protocol reproduction. Supervisor methodological approval is **not recorded**.

## Held-out test results

Three-seed mean ± **sample** standard deviation (`ddof=1`):

'''+ '\n'.join(rows)+'\n\n## Individual final held-out results\n\n'+'\n'.join(lines)+'''\n
## Audit and reproducibility

- `results/tuning/`: 12 trial histories, per-epoch timings, stored validation scores, frozen winner and grid manifest.
- `results/finals/`: 12 final training records with validation histories, per-epoch timings, validation scores, test-once claims, saved test scores, test metrics and both frozen summaries.
- `scripts/`: actual Stage 3/4/5/6 runner scripts including the exact, forensically attested import-path repair. The original author model is linked via its immutable upstream commit.
- `evidence/`: provenance, source/split hashes, independent pre-test audit, pilot evidence and available execution logs.
- Excluded from Git: raw source graph, frozen split data, all checkpoint `.pt` weights, recovery states, CUDA/Python environments and private archives. Independently verified Mac private backup holds frozen checkpoints and validation evidence. Canonical graph and split retain separate verified Mac originals.
- Runtime measurements reflect RTX A6000 instance 57212396; cross-instance performance is not a controlled comparison.
'''
    put('README.md',readme)
    status={'model':'EOGFD','dataset':'T-Finance','status':'COMPLETED_CONDITIONAL_UNIFIED_BENCHMARK',
       'mode':'UNIFIED_INPUT_ADAPTER_NOT_AUTHOR_IDENTICAL','instance_id':'57212396',
       'machine_hostname':'60d39211b1fd','upstream_commit':s5.PIN,
       'source_sha256':s5.SOURCE_SHA,'split_sha256':s5.SPLIT_SHA,
       'split_seed':2,'training_seeds':list(SEEDS),'ratios':list(RATIOS),
       'completed_tuning_trials':12,'completed_frozen_trainings':12,
       'completed_once_only_heldout_test_evaluations':12,'one_time_test_inference_per_run':True,
       'winning_trial':8,'winning_validation_auprc':frozen['winner_val_auprc'],
       'frozen_config':frozen['winner_config'],'test_metrics_summary':summary['metrics'],
       'venus_approval_recorded':False,'author_identical_reproduction':False,
       'original_upstream_model_implementation':True,'publication_performed_no_model_inference':True}
    put('status.json',jbytes(status))
    manifest={'branch':'Pathik-ass3','destination':PREFIX,'dataset':'T-Finance',
       'source_sha256':s5.SOURCE_SHA,'split_sha256':s5.SPLIT_SHA,
       'completed_trials':12,'completed_finals':12,'once_only_tests':12,
       'test_inference_per_checkpoint':1,'publication_performed_model_inference':False,
       'files':{name:{'sha256':sha_bytes(data),'bytes':len(data)} for name,data in sorted(files.items())}}
    man_name=PREFIX+'evidence/EOGFD_TFinance_PublicationFileManifest.json'
    files[man_name]=jbytes(manifest)
    require(len(files)>120,'TOO_FEW_PUBLIC_FILES')
    require(all(not k.endswith(('.pt','.npz','.dgl','.zip','.tar.gz')) for k in files),'PRIVATE_ARTIFACT_LEAK')
    tmp=Path(str(OUT)+'.writing')
    require(not tmp.exists(),'PARTIAL_EXPORT_EXISTS')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for path,data in sorted(files.items()):z.writestr(path,data)
    with zipfile.ZipFile(tmp) as z:
        require(z.testzip() is None and set(z.namelist())==set(files),'PUBLIC_ARCHIVE')
        for name,b in files.items():require(sha_bytes(z.read(name))==sha_bytes(b),'PUBLIC_ARCHIVE_HASH')
    os.replace(tmp,OUT)
    with CHECKSUM.open('x') as f:f.write(sha(OUT)+'  '+OUT.name+'\n')
    log='I1_TFINANCE_STAGE6_PUBLICATION_ZIP_AUDITED=PASS files='+str(len(files))+' sha256='+sha(OUT)
    print(log,flush=True)
    print('I1_STAGE6_GITHUB_PUSH_PENDING_MAC_VERIFICATION=YES',flush=True)
    print('PUBLICATION_ZIP='+str(OUT),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        print('I1_TFINANCE_STAGE6_STOP='+repr(e),flush=True);traceback.print_exc();sys.exit(2)
