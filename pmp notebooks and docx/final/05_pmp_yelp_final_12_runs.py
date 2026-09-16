#!/usr/bin/env python3
from pathlib import Path
from types import SimpleNamespace
import csv, gc, hashlib, json, math, os, random, shutil, subprocess, sys, time
import numpy as np
import torch, yaml
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

WORK=Path('/workspace/pmp_vast'); REPO=WORK/'repo/PMP'
CANONICAL=Path('/workspace/dataset_audit/raw/yelpchi/YelpChi.mat')
SPLIT=WORK/'shared/yelp_seed2_nested_splits.npz'
WINNER=WORK/'unified/yelp/tuning/PMP_YELP_FROZEN_WINNER.json'
ROOT=WORK/'unified/yelp/final'; RUN_DIR=ROOT/'runs'; CKPT_DIR=ROOT/'checkpoints'
for p in (ROOT,RUN_DIR,CKPT_DIR): p.mkdir(parents=True,exist_ok=True)
FINAL_RUNS_CSV=ROOT/'PMP_YELP_FINAL_12_RUNS.csv'; RATIO_SUMMARY_CSV=ROOT/'PMP_YELP_FINAL_RATIO_SUMMARY.csv'; FINAL_SUMMARY_JSON=ROOT/'PMP_YELP_FINAL_SUMMARY.json'; STATUS_TXT=ROOT/'PMP_YELP_FINAL_STATUS.txt'
EXPECTED_COMMIT='3f7629f6c180891a0bc1bba3c66d94d288a1ddae'
EXPECTED_DATA_SHA='fedb35a8fa539b27866244d3515a47a76b20080cdacb33112da3458fd2487b42'
EXPECTED_SPLIT_SHA='0ea0af36dfc5a3a1f381ea2e3168377e45826498b6063190185c673e9ec8c22b'
RATIOS=['TR40','TR30','TR20','TR10']; SEEDS=[2,42,72]
EXPECTED_SIZES={'TR40':18381,'TR30':13785,'TR20':9190,'TR10':4595,'val':9191,'test':18382}
MAX_EPOCHS=100; PATIENCE=20; THRESHOLDS=np.round(np.arange(0.01,1.00,0.01),2)

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def git(*a): return subprocess.check_output(['git','-C',str(REPO),*a],text=True).strip()
def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def write_csv(path,rows):
    if not rows: return
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
def save_state_cpu(model,path,meta):
    torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'meta':meta},path)
def load_state(model,path):
    o=torch.load(path,map_location='cpu'); model.load_state_dict(o['state_dict']); return o.get('meta',{})
def mean_sd(vals):
    a=np.asarray(vals,dtype=float); return float(a.mean()), float(a.std(ddof=1)) if len(a)>1 else 0.0

def choose_threshold(y,p):
    y=np.asarray(y).astype(int); p=np.asarray(p,dtype=float); rows=[]
    for t in THRESHOLDS:
        pred=(p>=t).astype(int)
        rows.append({'threshold':float(t),'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),'fraud_recall':float(recall_score(y,pred,pos_label=1,zero_division=0)),'distance_to_0_5':float(abs(float(t)-0.5))})
    rows.sort(key=lambda r:(-r['macro_f1'],-r['fraud_recall'],r['distance_to_0_5'],r['threshold']))
    return rows[0],rows

def metrics(y,p,t):
    y=np.asarray(y).astype(int); p=np.asarray(p,dtype=float); pred=(p>=float(t)).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel(); tpr=tp/(tp+fn) if tp+fn else 0.0; tnr=tn/(tn+fp) if tn+fp else 0.0
    return {'auprc':float(average_precision_score(y,p)),'auroc':float(roc_auc_score(y,p)),'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),'fraud_precision':float(precision_score(y,pred,pos_label=1,zero_division=0)),'fraud_recall':float(recall_score(y,pred,pos_label=1,zero_division=0)),'fraud_f1':float(f1_score(y,pred,pos_label=1,zero_division=0)),'gmean':float(math.sqrt(tpr*tnr)),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

print('===== PMP × YELPCHI — FINAL 12-RUN UNIFIED GRID =====',flush=True)
print('Ratios: TR40/TR30/TR20/TR10 | seeds: 2/42/72',flush=True)
print('Fresh final TR40 included: YES | tuning checkpoint reused as final: NO',flush=True)
print('Test used for selection: NO',flush=True)

# hard gates
assert git('rev-parse','HEAD')==EXPECTED_COMMIT
assert git('status','--porcelain')=='','Official PMP repository must be clean.'
assert sha256_file(CANONICAL)==EXPECTED_DATA_SHA
assert sha256_file(SPLIT)==EXPECTED_SPLIT_SHA
assert WINNER.exists()
wsha=sha256_file(WINNER); winner=json.loads(WINNER.read_text())
assert winner['status']=='FROZEN' and winner['repository_commit']==EXPECTED_COMMIT and winner['dataset_sha256']==EXPECTED_DATA_SHA and winner['split_sha256']==EXPECTED_SPLIT_SHA
assert int(winner['winner_trial'])==2 and abs(float(winner['lr'])-0.01)<1e-15 and abs(float(winner['weight_decay'])-1e-5)<1e-20
assert int(winner['hid_dim'])==48 and float(winner['dropout'])==0.0 and int(winner['n_layer'])==1
s=np.load(SPLIT,allow_pickle=False)
for k,n in EXPECTED_SIZES.items(): assert len(s[k])==n,(k,len(s[k]),n)
assert int(s['seed'][0])==2 and int(s['source_nodes'][0])==45954
assert set(map(int,s['TR10']))<=set(map(int,s['TR20']))<=set(map(int,s['TR30']))<=set(map(int,s['TR40']))
print('HARD_GATES=PASS | winner trial=2 lr=0.01 wd=1e-05',flush=True)

sys.path.insert(0,str(REPO)); os.chdir(REPO)
from DataHelper.datasetHelper import DatasetHelper
from training_procedure import Trainer
from utils.logger import Logger
from utils.random_seeder import set_random_seed
from dgl.dataloading import MultiLayerFullNeighborSampler
from dgl.data.utils import _get_dgl_url
assert torch.cuda.is_available() and torch.cuda.device_count()==1 and 'A6000' in torch.cuda.get_device_name(0).upper()
torch.cuda.set_device(0); torch.set_num_threads(8)
raw_cfg=yaml.safe_load((REPO/'config/yelp.yml').read_text()); base=dict(raw_cfg['LA-SAGE-S'])
fixed={'batch_size':512,'dataset_seed':717,'dropout':0.0,'full_neighbors':True,'hid_dim':48,'homo':False,'n_layer':1,'relation_agg':'cat','resi':0.2}
for k,v in fixed.items(): assert base.get(k)==v,(k,base.get(k),v)
base.update({'dataset':'yelp','model':'LA-SAGE-S','model_name':'LA-SAGE-S','gpu_id':0,'multirun':1,'run_best':False,'no_dev':False,'num_workers':8,'epochs':MAX_EPOCHS,'patience':PATIENCE,'eval_interval':1,'test_each_epoch':False,'monitor':'ap_gnn','lr':float(winner['lr']),'weight_decay':float(winner['weight_decay'])})

raw_root=Path('/dev/shm/pmp_yelp_final_raw'); shutil.rmtree(raw_root,ignore_errors=True); raw_root.mkdir(parents=True,exist_ok=True)
u=_get_dgl_url('dataset/FraudYelp.zip'); suffix=hashlib.sha1(u.encode()).hexdigest()[:8]
for sub in [f'yelp_{suffix}','yelp']:
    d=raw_root/sub; d.mkdir(parents=True,exist_ok=True); link=d/'YelpChi.mat'
    if link.exists() or link.is_symlink(): link.unlink()
    link.symlink_to(CANONICAL)
helper=DatasetHelper(config=base,dName='yelp',dDescription='COMP8851 final unified YelpChi'); helper.dataset_source_folder_path=str(raw_root); helper.load()
assert helper.num_nodes==45954 and helper.feat_dim==32 and helper.data.number_of_edges()==8051348 and int((helper.labels==1).sum())==6677 and set(helper.relations)=={'net_rsr','net_rtr','net_rur'}
print('DATA_MODEL_GATE=PASS | graph loaded once',flush=True)
logger=Logger(mode=[print])

def apply_split(ratio):
    N=helper.num_nodes; tr=torch.as_tensor(s[ratio],dtype=torch.long); va=torch.as_tensor(s['val'],dtype=torch.long); te=torch.as_tensor(s['test'],dtype=torch.long)
    tm=torch.zeros(N,dtype=torch.bool); vm=torch.zeros(N,dtype=torch.bool); sm=torch.zeros(N,dtype=torch.bool); tm[tr]=1; vm[va]=1; sm[te]=1
    helper.data.ndata['train_mask']=tm; helper.data.ndata['val_mask']=vm; helper.data.ndata['test_mask']=sm
    helper.train_mask=tm; helper.val_mask=vm; helper.test_mask=sm; helper.train_nid=tr; helper.val_nid=va; helper.test_nid=te
    lu=torch.full((N,),2,dtype=torch.long); lu[tr]=helper.labels[tr]; helper.data.ndata['label_unk']=lu
    assert int((lu!=2).sum())==len(tr) and torch.all(lu[va]==2) and torch.all(lu[te]==2)
    return tr,va,te

def args_for(seed,ratio):
    ratio_frac={'TR40':0.4,'TR30':0.3,'TR20':0.2,'TR10':0.1}[ratio]
    return SimpleNamespace(dataset='yelp',num_workers=8,seed=int(seed),data_dir=str(WORK/'canonical_raw'),hyper_file=str(REPO/'config'),log_dir=str(WORK/'logs'),best_model_path=str(WORK/'checkpoints'),train_size=ratio_frac,val_size=0.2,no_dev=False,gpu_id=0,multirun=1,model='LA-SAGE-S',run_best=False)

summaries=[]
for ratio in RATIOS:
  for seed in SEEDS:
    rid=f'yelp_{ratio.lower()}_seed{seed}'; sp=RUN_DIR/f'{rid}_summary.json'; epcsv=RUN_DIR/f'{rid}_epochs.csv'; thcsv=RUN_DIR/f'{rid}_val_threshold_grid.csv'; ckpt=CKPT_DIR/f'{rid}_best_val_auprc.pth'
    if sp.exists():
        old=json.loads(sp.read_text())
        if old.get('status')=='PASS' and old.get('run_id')==rid and old.get('ratio')==ratio and int(old.get('train_seed',-1))==seed and old.get('repository_commit')==EXPECTED_COMMIT and old.get('dataset_sha256')==EXPECTED_DATA_SHA and old.get('split_sha256')==EXPECTED_SPLIT_SHA and old.get('winner_sha256')==wsha and old.get('test_evaluated_once') is True and ckpt.exists():
            print(f'===== {rid}: RESUME PASS =====',flush=True); summaries.append(old); continue
    for p in [sp,epcsv,thcsv,ckpt]:
        if p.exists(): p.unlink()
    print(f'\n===== FINAL RUN {rid} =====',flush=True)
    tr,va,te=apply_split(ratio); assert len(tr)==EXPECTED_SIZES[ratio]
    cfg=dict(base); cfg.update({'seed':seed,'train_size':{'TR40':0.4,'TR30':0.3,'TR20':0.2,'TR10':0.1}[ratio],'val_size':0.2,'lr':float(winner['lr']),'weight_decay':float(winner['weight_decay']),'epochs':MAX_EPOCHS,'patience':PATIENCE,'eval_interval':1,'test_each_epoch':False,'monitor':'ap_gnn','best_model_path':str(ckpt)})
    seed_all(seed); set_random_seed(seed)
    sampler=MultiLayerFullNeighborSampler(num_layers=cfg['n_layer']); train_loader,val_loader,test_loader=helper.get_DGLloader(helper.data,sampler)
    T=Trainer(args=args_for(seed,ratio),config=cfg,logger=logger); model,opt,loss_func,scheduler=T.init(helper)
    for g in opt.param_groups:
        assert abs(float(g['lr'])-float(winner['lr']))<1e-15 and abs(float(g.get('weight_decay',0.0))-float(winner['weight_decay']))<1e-20 and tuple(g.get('betas',(0.9,0.999)))==(0.9,0.999) and abs(float(g.get('eps',1e-8))-1e-8)<1e-20
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); rows=[]; best=-1.0; best_epoch=-1; pc=0; wall=time.perf_counter()
    for epoch in range(1,MAX_EPOCHS+1):
        torch.cuda.synchronize(); t0=time.perf_counter(); model,loss=T.train(epoch-1,model,loss_func,opt,train_loader,helper); torch.cuda.synchronize(); train_s=time.perf_counter()-t0
        avg_loss=float(loss.detach().item())/max(1,len(train_loader))
        torch.cuda.synchronize(); v0=time.perf_counter(); vy,vp,_=T.evaluation(helper,val_loader,model,threshold_moving=False,thres=0.5); torch.cuda.synchronize(); val_s=time.perf_counter()-v0
        ap=float(average_precision_score(vy,vp)); auc=float(roc_auc_score(vy,vp)); improved=ap>best+1e-12
        if improved:
            best=ap; best_epoch=epoch; pc=0; save_state_cpu(model,ckpt,{'run_id':rid,'ratio':ratio,'train_seed':seed,'epoch':epoch,'val_auprc':ap,'lr':float(winner['lr']),'weight_decay':float(winner['weight_decay']),'winner_sha256':wsha,'repository_commit':EXPECTED_COMMIT,'dataset_sha256':EXPECTED_DATA_SHA,'split_sha256':EXPECTED_SPLIT_SHA})
        else: pc+=1
        rows.append({'run_id':rid,'ratio':ratio,'train_seed':seed,'epoch':epoch,'avg_train_loss':avg_loss,'train_seconds':float(train_s),'validation_seconds':float(val_s),'val_auprc':ap,'val_auroc':auc,'best_val_auprc_so_far':best,'best_epoch_so_far':best_epoch,'patience_count':pc,'peak_gpu_memory_mb_so_far':float(torch.cuda.max_memory_allocated()/1024**2)})
        write_csv(epcsv,rows)
        if epoch==1 or epoch%10==0 or improved or pc>=PATIENCE: print(f'{rid} epoch={epoch:03d} loss={avg_loss:.6f} val_AUPRC={ap:.6f} best={best:.6f}@{best_epoch} patience={pc}/{PATIENCE} train={train_s:.3f}s val={val_s:.3f}s',flush=True)
        if pc>=PATIENCE: print(f'EARLY_STOP {rid} epoch={epoch} best_epoch={best_epoch}',flush=True); break
    wall_s=time.perf_counter()-wall; peak_train=float(torch.cuda.max_memory_allocated()/1024**2); meta=load_state(model,ckpt); assert int(meta['epoch'])==best_epoch
    # threshold from validation only
    torch.cuda.synchronize(); q0=time.perf_counter(); vy,vp,_=T.evaluation(helper,val_loader,model,threshold_moving=False,thres=0.5); torch.cuda.synchronize(); val_inf=time.perf_counter()-q0
    th_choice,th_rows=choose_threshold(vy,vp); write_csv(thcsv,th_rows); th=float(th_choice['threshold']); vm=metrics(vy,vp,th)
    print(f'{rid} FROZEN checkpoint epoch={best_epoch} val_AUPRC={best:.6f} threshold={th:.2f}',flush=True)
    # exactly one test evaluation after config/checkpoint/threshold frozen
    torch.cuda.synchronize(); q0=time.perf_counter(); ty,tp,_=T.evaluation(helper,test_loader,model,threshold_moving=False,thres=0.5); torch.cuda.synchronize(); test_inf=time.perf_counter()-q0
    tm=metrics(ty,tp,th); peak_all=float(torch.cuda.max_memory_allocated()/1024**2); tmean,tsd=mean_sd([r['train_seconds'] for r in rows]); vmean,vsd=mean_sd([r['validation_seconds'] for r in rows])
    rs={'status':'PASS','run_id':rid,'model':'PMP / LA-SAGE-S','dataset':'YelpChi','ratio':ratio,'train_seed':seed,'train_nodes':len(tr),'validation_nodes':len(va),'test_nodes':len(te),'train_fraud':int(helper.labels[tr].sum()),'validation_fraud':int(helper.labels[va].sum()),'test_fraud':int(helper.labels[te].sum()),'repository_commit':EXPECTED_COMMIT,'dataset_sha256':EXPECTED_DATA_SHA,'split_sha256':EXPECTED_SPLIT_SHA,'winner_sha256':wsha,'winner_trial':int(winner['winner_trial']),'lr':float(winner['lr']),'weight_decay':float(winner['weight_decay']),'hid_dim':48,'dropout':0.0,'n_layer':1,'relation_agg':'cat','resi':0.2,'batch_size':512,'full_neighbors':True,'max_epochs':MAX_EPOCHS,'patience':PATIENCE,'epochs_completed':len(rows),'best_epoch':best_epoch,'best_validation_auprc':best,'selected_threshold':th,'threshold_rule':'validation Macro-F1 max; tie fraud recall; then closeness to 0.5','validation_metrics_at_selected_threshold':vm,'test_metrics':tm,'test_evaluated_once':True,'test_used_for_selection':False,'train_seconds_total':float(sum(r['train_seconds'] for r in rows)),'train_seconds_mean':tmean,'train_seconds_sd':tsd,'validation_seconds_total_during_training':float(sum(r['validation_seconds'] for r in rows)),'validation_seconds_mean':vmean,'validation_seconds_sd':vsd,'frozen_validation_inference_seconds':val_inf,'test_inference_seconds':test_inf,'training_wall_seconds':wall_s,'peak_trainval_gpu_memory_mb':peak_train,'peak_overall_gpu_memory_mb':peak_all,'checkpoint_path':str(ckpt),'checkpoint_sha256':sha256_file(ckpt),'epoch_csv':str(epcsv),'epoch_csv_sha256':sha256_file(epcsv),'threshold_csv':str(thcsv),'threshold_csv_sha256':sha256_file(thcsv)}
    sp.write_text(json.dumps(rs,indent=2)); summaries.append(rs)
    print(f"FINAL {rid} PASS | AUPRC={tm['auprc']:.6f} AUROC={tm['auroc']:.6f} MacroF1={tm['macro_f1']:.6f} Recall={tm['fraud_recall']:.6f}",flush=True)
    del model,opt,loss_func,scheduler,train_loader,val_loader,test_loader,T; gc.collect(); torch.cuda.empty_cache()

assert len(summaries)==12
by={(r['ratio'],int(r['train_seed'])):r for r in summaries}; assert len(by)==12
flat=[]
for ratio in RATIOS:
  for seed in SEEDS:
    r=by[(ratio,seed)]; assert r['status']=='PASS' and r['test_evaluated_once'] and not r['test_used_for_selection']; m=r['test_metrics']
    flat.append({'dataset':'YelpChi','model':'PMP','ratio':ratio,'seed':seed,'train_nodes':r['train_nodes'],'epochs_completed':r['epochs_completed'],'best_epoch':r['best_epoch'],'best_validation_auprc':r['best_validation_auprc'],'threshold':r['selected_threshold'],'auprc':m['auprc'],'auroc':m['auroc'],'macro_f1':m['macro_f1'],'fraud_precision':m['fraud_precision'],'fraud_recall':m['fraud_recall'],'fraud_f1':m['fraud_f1'],'gmean':m['gmean'],'train_seconds_total':r['train_seconds_total'],'train_seconds_mean':r['train_seconds_mean'],'train_seconds_sd':r['train_seconds_sd'],'test_inference_seconds':r['test_inference_seconds'],'peak_gpu_memory_mb':r['peak_overall_gpu_memory_mb']})
write_csv(FINAL_RUNS_CSV,flat)
metric_names=['auprc','auroc','macro_f1','fraud_precision','fraud_recall','fraud_f1','gmean']; ratio_rows=[]; ratio_json={}
for ratio in RATIOS:
    rr=[x for x in flat if x['ratio']==ratio]; row={'ratio':ratio,'n_seeds':3}; rj={}
    for m in metric_names:
        a,b=mean_sd([x[m] for x in rr]); row[m+'_mean']=a; row[m+'_sd']=b; rj[m]={'mean':a,'sd':b}
    a,b=mean_sd([x['train_seconds_total'] for x in rr]); row['total_train_seconds_mean']=a; row['total_train_seconds_sd']=b
    c,d=mean_sd([x['test_inference_seconds'] for x in rr]); row['test_inference_seconds_mean']=c; row['test_inference_seconds_sd']=d
    e,f=mean_sd([x['peak_gpu_memory_mb'] for x in rr]); row['peak_gpu_memory_mb_mean']=e; row['peak_gpu_memory_mb_sd']=f
    ratio_rows.append(row); ratio_json[ratio]={'metrics':rj,'total_train_seconds':{'mean':a,'sd':b},'test_inference_seconds':{'mean':c,'sd':d},'peak_gpu_memory_mb':{'mean':e,'sd':f}}
write_csv(RATIO_SUMMARY_CSV,ratio_rows)
final={'status':'PASS','model':'PMP / LA-SAGE-S','dataset':'YelpChi','final_runs_complete':12,'expected_final_runs':12,'ratios':RATIOS,'seeds':SEEDS,'repository_commit':EXPECTED_COMMIT,'dataset_sha256':EXPECTED_DATA_SHA,'split_sha256':EXPECTED_SPLIT_SHA,'winner_json':str(WINNER),'winner_sha256':wsha,'frozen_config':{'winner_trial':int(winner['winner_trial']),'lr':float(winner['lr']),'weight_decay':float(winner['weight_decay']),'hid_dim':48,'dropout':0.0,'n_layer':1,'relation_agg':'cat','resi':0.2,'batch_size':512,'full_neighbors':True},'checkpoint_selection':'best validation AUPRC','threshold_selection':'validation Macro-F1 over 0.01..0.99; tie fraud recall, then closeness to 0.5','test_isolation':'test evaluated once per final run after config, checkpoint and threshold frozen','ratio_summary':ratio_json,'final_runs_csv':str(FINAL_RUNS_CSV),'ratio_summary_csv':str(RATIO_SUMMARY_CSV),'gpu':torch.cuda.get_device_name(0)}
FINAL_SUMMARY_JSON.write_text(json.dumps(final,indent=2))
STATUS_TXT.write_text('PMP_YELP_FINAL=PASS\nPMP_DATASETS_COMPLETE=1/6\nPMP_YELP_FINAL_RUNS=12/12\nTEST_ISOLATION=PASS\n'+f"WINNER_TRIAL={winner['winner_trial']}\nLR={winner['lr']}\nWEIGHT_DECAY={winner['weight_decay']}\nSUMMARY_JSON={FINAL_SUMMARY_JSON}\n")
assert git('status','--porcelain')=='','Official PMP repo became dirty.'
print('\n===== PMP YELPCHI FINAL RESULTS — MEAN ± SD =====',flush=True)
for r in ratio_rows:
    print(f"{r['ratio']} | AUPRC={r['auprc_mean']:.6f}±{r['auprc_sd']:.6f} | AUROC={r['auroc_mean']:.6f}±{r['auroc_sd']:.6f} | MacroF1={r['macro_f1_mean']:.6f}±{r['macro_f1_sd']:.6f} | Prec={r['fraud_precision_mean']:.6f}±{r['fraud_precision_sd']:.6f} | Recall={r['fraud_recall_mean']:.6f}±{r['fraud_recall_sd']:.6f} | FraudF1={r['fraud_f1_mean']:.6f}±{r['fraud_f1_sd']:.6f} | GMean={r['gmean_mean']:.6f}±{r['gmean_sd']:.6f}",flush=True)
print('\n===== FINAL GATE =====',flush=True)
print('PMP_YELP_FINAL=PASS'); print('PMP_YELP_FINAL_RUNS=12/12'); print('PMP_DATASETS_COMPLETE=1/6'); print('TEST_ISOLATION=PASS'); print('PMP_AUTHOR_REPO_CLEAN=PASS')
print('FINAL_RUNS_CSV=',FINAL_RUNS_CSV); print('RATIO_SUMMARY_CSV=',RATIO_SUMMARY_CSV); print('FINAL_SUMMARY_JSON=',FINAL_SUMMARY_JSON)
