#!/usr/bin/env python3
from pathlib import Path
from types import SimpleNamespace
import os, sys, json, time, random, hashlib, shutil, subprocess, csv, gc
import numpy as np
import torch
import yaml

WORK = Path('/workspace/pmp_vast')
REPO = WORK / 'repo/PMP'
CANONICAL = Path('/workspace/dataset_audit/raw/yelpchi/YelpChi.mat')
SPLIT = WORK / 'shared/yelp_seed2_nested_splits.npz'
PACK_DIR = Path(__file__).resolve().parent
TRIAL_MANIFEST = PACK_DIR / 'pmp_yelp_tuning_trials.json'
ROOT = WORK / 'unified/yelp/tuning'
TRIAL_DIR = ROOT / 'trials'
CHECKPOINT_DIR = ROOT / 'checkpoints'
for p in [ROOT, TRIAL_DIR, CHECKPOINT_DIR]: p.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = ROOT / 'PMP_YELP_TR40_tuning_summary.csv'
SUMMARY_JSON = ROOT / 'PMP_YELP_TR40_tuning_summary.json'
WINNER_JSON = ROOT / 'PMP_YELP_FROZEN_WINNER.json'
META_JSON = ROOT / 'PMP_YELP_TUNING_META.json'

EXPECTED_COMMIT = '3f7629f6c180891a0bc1bba3c66d94d288a1ddae'
EXPECTED_DATA_SHA = 'fedb35a8fa539b27866244d3515a47a76b20080cdacb33112da3458fd2487b42'
EXPECTED_SPLIT_SHA = '0ea0af36dfc5a3a1f381ea2e3168377e45826498b6063190185c673e9ec8c22b'
MAX_EPOCHS, PATIENCE, TRAIN_SEED = 100, 20, 2

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def git(*args):
    return subprocess.check_output(['git','-C',str(REPO),*args], text=True).strip()

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def write_csv(path, rows):
    if not rows: return
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def save_state_cpu(model, path, meta):
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    torch.save({'state_dict':state,'meta':meta}, path)

print('===== PMP VAST — YELPCHI TR40 TUNING =====', flush=True)
print('External COMP8851 adapter: YES', flush=True)
print('Official PMP source edited: NO', flush=True)
print('Test evaluation during tuning: NO', flush=True)
print('Trials: 12 pre-registered', flush=True)
print('Selection: validation AUPRC', flush=True)
print('Max epochs/trial: 100 | patience: 20 | train seed: 2', flush=True)

print('\n===== 0. IDENTITY / PROTOCOL GATES =====', flush=True)
assert git('rev-parse','HEAD') == EXPECTED_COMMIT
status0=git('status','--porcelain')
assert status0 == '', f'Official PMP repository is dirty before tuning:\n{status0}'
assert sha256_file(CANONICAL) == EXPECTED_DATA_SHA
assert sha256_file(SPLIT) == EXPECTED_SPLIT_SHA
trials=json.loads(TRIAL_MANIFEST.read_text())
assert len(trials)==12 and [t['trial'] for t in trials]==list(range(1,13))
assert sorted(set(float(t['lr']) for t in trials)) == [1e-4,1e-3,1e-2]
assert sorted(set(float(t['weight_decay']) for t in trials)) == [0.0,1e-5,1e-4,1e-3]
s=np.load(SPLIT, allow_pickle=False)
assert len(s['TR40'])==18381 and len(s['val'])==9191 and len(s['test'])==18382
assert int(s['seed'][0])==2 and int(s['source_nodes'][0])==45954
print('SOURCE_DATA_SPLIT_GATE=PASS', flush=True)
print('TUNING_MANIFEST_GATE=PASS', flush=True)

sys.path.insert(0,str(REPO)); os.chdir(REPO)
from DataHelper.datasetHelper import DatasetHelper
from training_procedure import Trainer
from utils.logger import Logger
from utils.random_seeder import set_random_seed
from dgl.dataloading import MultiLayerFullNeighborSampler
from dgl.data.utils import _get_dgl_url

assert torch.cuda.is_available() and torch.cuda.device_count()==1
assert 'A6000' in torch.cuda.get_device_name(0).upper()
torch.cuda.set_device(0); torch.set_num_threads(8)

raw_cfg=yaml.safe_load((REPO/'config/yelp.yml').read_text())
base=dict(raw_cfg['LA-SAGE-S'])
fixed={'batch_size':512,'dataset_seed':717,'dropout':0.0,'full_neighbors':True,'hid_dim':48,'homo':False,'n_layer':1,'relation_agg':'cat','resi':0.2}
for k,v in fixed.items(): assert base.get(k)==v,(k,base.get(k),v)
print('\n===== 1. FIXED PMP YELP ARCHITECTURE =====', flush=True)
print('LA-SAGE-S | hidden=48 | dropout=0.0 | layers=1 | relation_agg=cat | resi=0.2', flush=True)
print('batch=512 | full_neighbors=True', flush=True)
print('Tuned parameters only: learning rate + weight decay', flush=True)
print('AUTHOR_ARCHITECTURE_FREEZE=PASS', flush=True)

print('\n===== 2. CANONICAL AUTHOR DATA LOAD =====', flush=True)
raw_root=Path('/dev/shm/pmp_yelp_tuning_raw'); shutil.rmtree(raw_root,ignore_errors=True); raw_root.mkdir(parents=True)
dgl_url=_get_dgl_url('dataset/FraudYelp.zip'); suffix=hashlib.sha1(dgl_url.encode()).hexdigest()[:8]
for sub in [f'yelp_{suffix}','yelp']:
    d=raw_root/sub; d.mkdir(parents=True,exist_ok=True); link=d/'YelpChi.mat'
    if link.exists() or link.is_symlink(): link.unlink()
    link.symlink_to(CANONICAL)
loader_cfg=dict(base)
loader_cfg.update({'dataset':'yelp','model':'LA-SAGE-S','model_name':'LA-SAGE-S','gpu_id':0,'seed':2,'train_size':0.4,'val_size':0.2,'multirun':1,'run_best':False,'no_dev':False,'num_workers':8,'epochs':100,'patience':20,'eval_interval':1,'test_each_epoch':False,'monitor':'ap_gnn','best_model_path':str(ROOT/'unused_author_checkpoint.pth')})
helper=DatasetHelper(config=loader_cfg,dName='yelp',dDescription='COMP8851 unified YelpChi')
helper.dataset_source_folder_path=str(raw_root); helper.load()
assert helper.num_nodes==45954 and helper.feat_dim==32 and helper.data.number_of_edges()==8051348
assert int((helper.labels==1).sum().item())==6677
assert set(helper.relations)=={'net_rsr','net_rtr','net_rur'}
N=helper.num_nodes
train_ids=torch.as_tensor(s['TR40'],dtype=torch.long); val_ids=torch.as_tensor(s['val'],dtype=torch.long); test_ids=torch.as_tensor(s['test'],dtype=torch.long)
train_mask=torch.zeros(N,dtype=torch.bool); train_mask[train_ids]=True
val_mask=torch.zeros(N,dtype=torch.bool); val_mask[val_ids]=True
test_mask=torch.zeros(N,dtype=torch.bool); test_mask[test_ids]=True
helper.data.ndata['train_mask']=train_mask; helper.data.ndata['val_mask']=val_mask; helper.data.ndata['test_mask']=test_mask
helper.train_mask=train_mask; helper.val_mask=val_mask; helper.test_mask=test_mask
helper.train_nid=train_ids; helper.val_nid=val_ids; helper.test_nid=test_ids
label_unk=torch.full((N,),2,dtype=torch.long); label_unk[train_ids]=helper.labels[train_ids]; helper.data.ndata['label_unk']=label_unk
assert torch.all(label_unk[val_ids]==2) and torch.all(label_unk[test_ids]==2)
print('TR40 / val / test:',len(train_ids),len(val_ids),len(test_ids),flush=True)
print('TR40 fraud:',int(helper.labels[train_ids].sum()),'| val fraud:',int(helper.labels[val_ids].sum()),flush=True)
print('Partition labels: TRAIN ONLY',flush=True)
print('FROZEN_SPLIT_ADAPTER_GATE=PASS',flush=True)

args=SimpleNamespace(dataset='yelp',num_workers=8,seed=2,data_dir=str(WORK/'canonical_raw'),hyper_file=str(REPO/'config'),log_dir=str(WORK/'logs'),best_model_path=str(WORK/'checkpoints'),train_size=0.4,val_size=0.2,no_dev=False,gpu_id=0,multirun=1,model='LA-SAGE-S',run_best=False)
logger=Logger(mode=[print])
summaries=[]

for spec in trials:
    tid=int(spec['trial']); lr=float(spec['lr']); wd=float(spec['weight_decay']); name=f'trial_{tid:02d}'
    tjson=TRIAL_DIR/f'{name}_summary.json'; ecsv=TRIAL_DIR/f'{name}_epochs.csv'; ckpt=CHECKPOINT_DIR/f'{name}_best_val_auprc.pth'
    if tjson.exists() and ckpt.exists():
        old=json.loads(tjson.read_text())
        if old.get('status')=='PASS' and old.get('repository_commit')==EXPECTED_COMMIT and old.get('split_sha256')==EXPECTED_SPLIT_SHA and float(old.get('lr'))==lr and float(old.get('weight_decay'))==wd:
            print(f'\n===== {name} RESUME: already PASS =====',flush=True); summaries.append(old); continue
    print(f'\n===== TRIAL {tid:02d}/12 | lr={lr:g} | wd={wd:g} =====',flush=True)
    seed_all(TRAIN_SEED); set_random_seed(TRAIN_SEED)
    config=dict(loader_cfg); config.update({'lr':lr,'weight_decay':wd,'seed':2,'epochs':100,'patience':20,'eval_interval':1,'test_each_epoch':False,'monitor':'ap_gnn'})
    sampler=MultiLayerFullNeighborSampler(num_layers=config['n_layer'])
    train_loader,val_loader,_test_loader_unused=helper.get_DGLloader(helper.data,sampler)
    T=Trainer(args=args,config=config,logger=logger); model,optimizer,loss_func,scheduler=T.init(helper)
    for g in optimizer.param_groups:
        assert abs(float(g['lr'])-lr)<1e-15 and abs(float(g.get('weight_decay',0.0))-wd)<1e-15
        assert tuple(g.get('betas',(0.9,0.999)))==(0.9,0.999) and abs(float(g.get('eps',1e-8))-1e-8)<1e-20
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    rows=[]; best=-float('inf'); best_epoch=-1; patience=0; tstart=time.perf_counter()
    for epoch in range(1,MAX_EPOCHS+1):
        torch.cuda.synchronize(); t0=time.perf_counter(); model,loss=T.train(epoch-1,model,loss_func,optimizer,train_loader,helper); torch.cuda.synchronize(); train_s=time.perf_counter()-t0
        avg_loss=float(loss.detach().item())/max(1,len(train_loader))
        torch.cuda.synchronize(); v0=time.perf_counter(); labels,probs,preds=T.evaluation(helper,val_loader,model,threshold_moving=config['threshold_moving'],thres=config['thres']); torch.cuda.synchronize(); val_s=time.perf_counter()-v0
        dev=T.eval_model(labels,probs,preds); auprc=float(dev.ap_gnn); improved=auprc>best+1e-12
        if improved:
            best=auprc; best_epoch=epoch; patience=0
            save_state_cpu(model,ckpt,{'trial':tid,'epoch':epoch,'val_auprc':auprc,'lr':lr,'weight_decay':wd,'train_seed':2,'split_sha256':EXPECTED_SPLIT_SHA,'repository_commit':EXPECTED_COMMIT})
        else: patience+=1
        rows.append({'trial':tid,'epoch':epoch,'lr':lr,'weight_decay':wd,'avg_train_loss':avg_loss,'train_seconds':float(train_s),'validation_seconds':float(val_s),'val_auprc':auprc,'val_auroc':float(dev.auc_gnn),'best_val_auprc_so_far':float(best),'best_epoch_so_far':int(best_epoch),'patience_count':int(patience),'peak_gpu_memory_mb_so_far':float(torch.cuda.max_memory_allocated()/1024**2)})
        write_csv(ecsv,rows)
        if epoch==1 or epoch%10==0 or improved or patience>=PATIENCE:
            print(f'trial={tid:02d} epoch={epoch:03d} loss={avg_loss:.6f} val_AUPRC={auprc:.6f} best={best:.6f}@{best_epoch} patience={patience}/{PATIENCE} train={train_s:.3f}s val={val_s:.3f}s',flush=True)
        if patience>=PATIENCE:
            print(f'EARLY_STOP trial={tid:02d} epoch={epoch} best_epoch={best_epoch}',flush=True); break
    total=time.perf_counter()-tstart
    summary={'status':'PASS','trial':tid,'lr':lr,'weight_decay':wd,'fixed_hid_dim':48,'fixed_dropout':0.0,'fixed_n_layer':1,'fixed_relation_agg':'cat','fixed_resi':0.2,'train_seed':2,'split':'TR40','repository_commit':EXPECTED_COMMIT,'dataset_sha256':EXPECTED_DATA_SHA,'split_sha256':EXPECTED_SPLIT_SHA,'best_val_auprc':float(best),'best_epoch':int(best_epoch),'epochs_completed':len(rows),'stopped_early':len(rows)<100,'total_trial_seconds':float(total),'mean_train_seconds':float(np.mean([r['train_seconds'] for r in rows])),'sd_train_seconds':float(np.std([r['train_seconds'] for r in rows],ddof=1)) if len(rows)>1 else 0.0,'mean_validation_seconds':float(np.mean([r['validation_seconds'] for r in rows])),'peak_gpu_memory_mb':float(torch.cuda.max_memory_allocated()/1024**2),'checkpoint':str(ckpt),'epoch_csv':str(ecsv),'test_evaluated':False,'test_loader_iterated':False}
    tjson.write_text(json.dumps(summary,indent=2)); summaries.append(summary)
    print(f'TRIAL {tid:02d} DONE | best val AUPRC={best:.6f} at epoch {best_epoch} | epochs={len(rows)}',flush=True)
    del model,optimizer,loss_func,scheduler,train_loader,val_loader,_test_loader_unused,T; gc.collect(); torch.cuda.empty_cache()

assert len(summaries)==12
summaries=sorted(summaries,key=lambda x:int(x['trial'])); write_csv(SUMMARY_CSV,summaries)
ranked=sorted(summaries,key=lambda x:(-float(x['best_val_auprc']),int(x['trial']))); winner=ranked[0]
SUMMARY_JSON.write_text(json.dumps({'dataset':'YelpChi','model':'PMP / LA-SAGE-S','tuning_track':'unified TR40','train_seed':2,'max_trials':12,'max_epochs':100,'patience':20,'selection_metric':'validation AUPRC','test_evaluated':False,'fixed_author_architecture':fixed,'tuned_parameters':['learning_rate','weight_decay'],'trial_manifest':trials,'ranked_results':ranked,'winner':winner},indent=2))
frozen={'status':'FROZEN','model':'PMP / LA-SAGE-S','dataset':'YelpChi','repository_commit':EXPECTED_COMMIT,'dataset_sha256':EXPECTED_DATA_SHA,'split_sha256':EXPECTED_SPLIT_SHA,'selection':'TR40 train_seed=2, best validation AUPRC, no test access','winner_trial':int(winner['trial']),'lr':float(winner['lr']),'weight_decay':float(winner['weight_decay']),'hid_dim':48,'dropout':0.0,'n_layer':1,'relation_agg':'cat','resi':0.2,'batch_size':512,'full_neighbors':True,'best_tuning_epoch':int(winner['best_epoch']),'best_tuning_val_auprc':float(winner['best_val_auprc']),'tuning_checkpoint':winner['checkpoint'],'test_evaluated_during_tuning':False}
WINNER_JSON.write_text(json.dumps(frozen,indent=2))
META_JSON.write_text(json.dumps({'repository_commit':EXPECTED_COMMIT,'dataset_sha256':EXPECTED_DATA_SHA,'split_sha256':EXPECTED_SPLIT_SHA,'trial_manifest_sha256':sha256_file(TRIAL_MANIFEST),'summary_csv_sha256':sha256_file(SUMMARY_CSV),'summary_json_sha256':sha256_file(SUMMARY_JSON),'winner_json_sha256':sha256_file(WINNER_JSON),'gpu':torch.cuda.get_device_name(0)},indent=2))
status_after=git('status','--porcelain'); non_cache=[line for line in status_after.splitlines() if '__pycache__' not in line and not line.strip().endswith('.pyc')]
assert not non_cache,'Unexpected official repository changes:\n'+'\n'.join(non_cache)
print('\n===== PMP YELP TR40 TUNING RESULTS =====',flush=True)
for r in ranked:
    print(f"trial={int(r['trial']):02d} | lr={float(r['lr']):g} | wd={float(r['weight_decay']):g} | epoch={int(r['best_epoch']):3d} | val AUPRC={float(r['best_val_auprc']):.6f}",flush=True)
print('\n===== FROZEN WINNER =====',flush=True)
print('winner trial   :',winner['trial'],flush=True); print('learning rate  :',winner['lr'],flush=True); print('weight decay   :',winner['weight_decay'],flush=True)
print('hidden/dropout : 48 / 0.0 (author Yelp settings)',flush=True); print('best epoch     :',winner['best_epoch'],flush=True); print('best val AUPRC :',f"{winner['best_val_auprc']:.6f}",flush=True)
print('TEST_SET_ACCESSED_DURING_TUNING=NO',flush=True); print('PMP_YELP_TR40_TUNING=PASS',flush=True); print('FROZEN_CONFIG=',WINNER_JSON,flush=True); print('SUMMARY_CSV=',SUMMARY_CSV,flush=True)
