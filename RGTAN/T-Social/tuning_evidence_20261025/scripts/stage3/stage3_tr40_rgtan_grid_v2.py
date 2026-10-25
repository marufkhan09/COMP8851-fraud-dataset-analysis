#!/usr/bin/env python3
"""RGTAN x T-Social (I1): pinned Stage-2 verification, cost probe, and ONE validation-only tuning trial.

No test index/label access or scoring. Frozen TR40 train IDs + stage2 training labels,
5fold OOF risk/context for training, full TR40 view for validation. Author network
unchanged; bounded [10,10] neighbor sampler is an explicit methodological adaptation.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, random, socket, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT=Path('/workspace/comp8851')
AUTHOR=ROOT/'repos/antifraud'
CACHE=ROOT/'assets/rgtan/tsocial/exact_risk_sparse_stage2'
SOURCE=ROOT/'assets/tsocial/tsocial'
SPLIT=ROOT/'assets/tsocial/tsocial_seed2_nested_splits.npz'
PILOT=ROOT/'diagnostics/rgtan/tsocial/stage3_realrisk_pilot/single_realrisk_capacity_pilot.json'
STAGE2_AUDITS=ROOT/'diagnostics/rgtan/tsocial'
OUT=ROOT/'results/rgtan/tsocial/stage3_tr40_tune_gate_v1'
PIN='65e583f03ac5c5983e49cc35179fe7a65908b9c1'
SOURCE_SHA='8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0'
SPLIT_SHA='b7e76835612e72a8fe5c69bbe87c8ea7b2e5790cbd8849e2a3ac776364bc6519'
HOST='60d39211b1fd'
N=5_781_065; E=146_211_016; TRN=2_312_426; VAL=1_156_213
BATCH=1024; SEED=2; FANOUTS=(10,10); MAX_EPOCHS=100; PATIENCE=20
NAMES=('degree','riskstat','1hop_degree','2hop_degree','1hop_riskstat','2hop_riskstat')
FILES=([f'risk_view_{i}.npy' for i in range(6)]+[f'label_context_{i}.npy' for i in range(6)]
       + ['train_ids.npy','train_labels.npy','val_ids.npy','fold_of_node.npy'])
GRID=[{'hidden_dim':h,'drop':[d,d/2], 'base_lr':lr}
      for h in (32,64) for d in (.2,.3) for lr in (.001,.003,.005)]
assert len(GRID)==12

def req(c,msg):
    if not c:raise RuntimeError('STOP_'+str(msg))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read_json(p):
    req(Path(p).is_file(),'MISSING_'+str(p))
    return json.loads(Path(p).read_text())

def atomic_save(path, writer, overwrite=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.partial')
    req(not temp.exists(),'STALE_PARTIAL_'+str(temp))
    req(overwrite or not path.exists(),'ALREADY_EXISTS_'+str(path))
    try:
        with temp.open('xb') as f:
            writer(f);f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise

def atomic_json(path,obj,overwrite=False):
    def writer(f):f.write((json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())
    atomic_save(path,writer,overwrite)

def npy_save(path,arr,overwrite=False):
    atomic_save(path,lambda f:np.save(f,arr,allow_pickle=False),overwrite)

def torch_save(path,obj,torch,overwrite=False):
    atomic_save(path,lambda f:torch.save(obj,f),overwrite)

def unit_selftest():
    tr=np.arange(1000,dtype=np.int64);y=np.zeros(1000,dtype=np.int8);y[::17]=1
    folds=np.arange(1000,dtype=np.int8)%5
    for k in range(5):
        chosen=tr[folds==k]
        ctx=np.full(1000,2,dtype=np.int8)
        ctx[tr[folds!=k]]=y[folds!=k]
        req(np.all(ctx[chosen]==2),'FOLD_OWN_SEEDS_NOT_MASKED')
        req(np.array_equal(ctx[tr[folds!=k]],y[folds!=k]),'OTHER_FOLD_LABELS_NOT_AVAILABLE')
    req(len(GRID)==12 and GRID[0]==dict(hidden_dim=32,drop=[.2,.1],base_lr=.001),'GRID_INVALID')
    print('I1_RGTAN_TSOCIAL_TR40_STAGE3_FOLD_GRID_UNIT_TEST=PASS',flush=True)

def check_readiness(deep):
    req(socket.gethostname()==HOST,'WRONG_I1_HOST')
    req(subprocess.check_output(['git','-C',str(AUTHOR),'rev-parse','HEAD'],text=True).strip()==PIN,'AUTHOR_PIN_CHANGED')
    req(not subprocess.check_output(['git','-C',str(AUTHOR),'status','--porcelain'],text=True).strip(),'AUTHOR_SOURCE_DIRTY')
    man=read_json(CACHE/'risk_crossfit_complete.json')
    req(man['status']=='RGTAN_TSOCIAL_I1_EXACT_TRAIN_ONLY_5FOLD_CROSSFIT_COMPLETE','CROSSFIT_NOT_COMPLETE')
    req(man['source_sha256']==SOURCE_SHA and man['split_sha256']==SPLIT_SHA and man['author_commit']==PIN,'DATA_HASH_PIN_MISMATCH')
    req(man['nodes']==N and man['original_graph_edges']==E and man['train_count']==TRN and man['train_fraud']==69712,'STAGE2_COUNTS_CHANGED')
    req(man['ratio']=='TR40' and man['features']==list(NAMES) and set(man['files'])==set(FILES),'STAGE2_SCHEMA_CHANGED')
    req(man['validation_labels_used'] is False and man['test_labels_used'] is False and man['test_scored'] is False,'STAGE2_LABEL_PROTOCOL')
    audit_sha=sha(CACHE/'risk_crossfit_complete.json')
    audits=[p for p in STAGE2_AUDITS.glob('stage2_independent_audit_*.json')
            if read_json(p).get('status')=='I1_RGTAN_TSOCIAL_STAGE2_INDEPENDENT_AUDIT_PASS'
            and read_json(p).get('stage2_complete_manifest_sha256')==audit_sha
            and read_json(p).get('test_scored') is False]
    req(audits,'NO_MATCHING_STAGE2_INDEPENDENT_AUDIT')
    pilot=read_json(PILOT)
    req(pilot['status']=='I1_RGTAN_TSOCIAL_REAL_RISK_PILOT_PASS','REAL_RISK_PILOT_NOT_PASS')
    req(pilot['risk_crossfit_manifest_sha256']==audit_sha and pilot['test_scored'] is False
        and pilot['validation_labels_accessed'] is False,'PILOT_FINGERPRINT_MISMATCH')
    req(SOURCE.stat().st_size==4_110_300_257 and SPLIT.stat().st_size==14_180_251,'SOURCE_FILE_SIZE_CHANGED')
    if deep:
        req(sha(SOURCE)==SOURCE_SHA and sha(SPLIT)==SPLIT_SHA,'SOURCE_HASH_CHANGED')
    for name in FILES:
        p=CACHE/name
        req(p.is_file() and p.stat().st_size==man['files'][name]['bytes'],'FEATURE_SIZE_CHANGED_'+name)
        if deep:req(sha(p)==man['files'][name]['sha256'],'FEATURE_HASH_CHANGED_'+name)
    print('I1_RGTAN_TSOCIAL_STAGE3_PINNED_AUTHOR_FROZEN_SPLIT_AND_AUDITED_16_FILES=PASS',flush=True)
    return man,audit_sha

def load_arrays():
    a={key:np.load(CACHE/(key+'.npy'),mmap_mode='r',allow_pickle=False)
        for key in ('train_ids','train_labels','val_ids','fold_of_node')}
    for k in range(6):
        a[f'risk_{k}']=np.load(CACHE/f'risk_view_{k}.npy',mmap_mode='r',allow_pickle=False)
        a[f'context_{k}']=np.load(CACHE/f'label_context_{k}.npy',mmap_mode='r',allow_pickle=False)
    req(a['train_ids'].shape==(TRN,) and a['val_ids'].shape==(VAL,) and a['train_labels'].shape==(TRN,),'SAVED_ID_SHAPE')
    req(int(np.sum(a['train_labels'],dtype=np.int64))==69712,'TRAIN_LABELS_CHANGED')
    ff=np.asarray(a['fold_of_node'][a['train_ids']],dtype=np.int8)
    req(set(np.unique(ff).tolist())==set(range(5)),'FOLD_SET_CHANGED')
    for k in range(5):
        req(np.all(a[f'context_{k}'][a['train_ids'][ff==k]]==2),'FOLD_OWN_LABEL_LEAK')
    req(np.all(a['context_5'][a['val_ids']]==2),'VALIDATION_LABEL_CONTEXT_LEAK')
    return a,ff

def initialize(seed):
    import torch,dgl,psutil
    req(torch.__version__=='1.12.1+cu113' and dgl.__version__=='0.8.1' and torch.cuda.is_available(),'CUDA_DGL_ENV_CHANGED')
    req(psutil.virtual_memory().available > 24*1024**3,'LOW_AVAILABLE_RAM')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.set_num_threads(8)
    dgl.seed(seed)
    return torch,dgl,psutil

def graph_load(a,torch,dgl,include_val_y):
    gs,_=dgl.load_graphs(str(SOURCE));req(len(gs)==1,'GRAPH_COUNT')
    g=gs[0]
    req(g.num_nodes()==N and g.num_edges()==E and list(g.canonical_etypes)==[('_U','_V','_U')],'CANONICAL_GRAPH_LAYOUT_CHANGED')
    req('feature' in g.ndata and 'label' in g.ndata,'SOURCE_ATTRIBUTES_MISSING')
    x=g.ndata.pop('feature').float().numpy()
    req(x.shape==(N,10) and np.isfinite(x).all(),'SOURCE_X_INVALID')
    labels=g.ndata.pop('label')
    # The sole permitted validation-label index operation is inside full tuning trials.
    val_y=(labels[torch.from_numpy(a['val_ids'].copy())].cpu().numpy().reshape(-1).astype(np.int8,copy=True)
           if include_val_y else None)
    del labels
    for key in list(g.ndata):g.ndata.pop(key)
    for key in list(g.edata):g.edata.pop(key)
    req(not list(g.ndata) and not list(g.edata),'GRAPH_LABEL_METADATA_NOT_REMOVED')
    print(f'I1_RGTAN_TSOCIAL_TR40_ORIGINAL_GRAPH_READY=PASS nodes={N} edges={E} validation_labels_indexed={include_val_y} test_labels_indexed=NO',flush=True)
    return g,x,val_y

class Engine:
    def __init__(self,g,x,a,folds,config,torch,dgl):
        import pandas as pd
        from dgl.dataloading import MultiLayerNeighborSampler,NodeDataLoader
        from methods.rgtan.rgtan_model import RGTAN
        self.g,self.x,self.a,self.torch,self.dgl=g,x,a,torch,dgl
        self.seed_nodes=[];self.fold_seed_labels=[]
        train=np.asarray(a['train_ids']);yt=np.asarray(a['train_labels'])
        for k in range(5):
            mask=(folds==k)
            self.seed_nodes.append(np.asarray(train[mask],dtype=np.int64).copy())
            self.fold_seed_labels.append(np.asarray(yt[mask],dtype=np.int64).copy())
        req(sum(map(len,self.seed_nodes))==TRN,'TR40_FOLD_COUNT_MISMATCH')
        placeholder={name:torch.empty(0,device='cuda:0') for name in NAMES}
        self.model=RGTAN(in_feats=10,hidden_dim=config['hidden_dim'],n_layers=2,n_classes=2,
                         heads=[4,4],activation=torch.nn.PReLU(),drop=config['drop'],device='cuda:0',
                         gated=True,ref_df=pd.DataFrame(x),cat_features={},neigh_features=placeholder,nei_att_head=2).to('cuda:0')
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=config['base_lr']*(BATCH/1024)**0.5,weight_decay=1e-4)
        self.sampler=MultiLayerNeighborSampler(list(FANOUTS))
        self.NodeDataLoader=NodeDataLoader
        self.optimizer_steps=0

    def inputs(self,input_ids,output_ids,blocks,fold,stage):
        torch=self.torch;ins=input_ids.cpu().numpy();outs=output_ids.cpu().numpy()
        ctx=np.array(self.a[f'context_{fold}'][ins],dtype=np.int64,copy=True)
        req(np.isin(ctx,(0,1,2)).all(),'INVALID_PROPAGATION_LABEL')
        # All target node labels must be masked for label propagation.
        mask=np.isin(ins,outs,assume_unique=False)
        req(np.all(ctx[mask]==2),'SEED_TARGET_LABEL_CONTEXT_EXPOSED')
        risk=np.array(self.a[f'risk_{fold}'][ins],dtype=np.float32,copy=True)
        xx=np.array(self.x[ins],dtype=np.float32,copy=True)
        req(np.isfinite(xx).all() and np.isfinite(risk).all(),'NONFINITE_FEATURE')
        risks={name:torch.from_numpy(risk[:,i].copy()).to('cuda:0') for i,name in enumerate(NAMES)}
        gpu_blocks=[b.to('cuda:0') for b in blocks]
        logits=self.model(gpu_blocks,torch.from_numpy(xx).to('cuda:0'),
                          torch.from_numpy(ctx).to('cuda:0'),{},risks)
        req(logits.shape==(len(outs),2) and bool(torch.isfinite(logits).all().item()),'BAD_LOGITS')
        return logits

    def epoch(self,epoch,limit_per_fold=None):
        torch=self.torch;dgl=self.dgl
        import torch.nn.functional as F
        self.model.train();loss_sum=0.0;count=0;steps=0
        for k in np.random.RandomState(SEED+epoch).permutation(5):
            k=int(k)
            # Shuffle fold seed indices with NumPy rather than relying on library dataloader RNG on resume.
            idx=np.random.RandomState(SEED+epoch*37+k).permutation(len(self.seed_nodes[k]))
            if limit_per_fold is not None:idx=idx[:BATCH*limit_per_fold]
            ids=self.seed_nodes[k][idx]; targets=self.fold_seed_labels[k][idx]
            ymap=np.full(N,-1,dtype=np.int8);ymap[ids]=targets
            dgl.seed(SEED+epoch*17+k)
            loader=self.NodeDataLoader(self.g,torch.from_numpy(ids.copy()),self.sampler,
                batch_size=BATCH,shuffle=False,drop_last=False,num_workers=0)
            for input_ids,output_ids,blocks in loader:
                out=output_ids.cpu().numpy()
                yy=np.asarray(ymap[out],dtype=np.int64)
                req(np.all(yy>=0),'FOLD_TRAIN_TARGET_NOT_FOUND')
                self.optimizer.zero_grad(set_to_none=True)
                logits=self.inputs(input_ids,output_ids,blocks,k,'train')
                loss=F.cross_entropy(logits,torch.from_numpy(yy.copy()).to('cuda:0'))
                req(bool(torch.isfinite(loss).item()),'NONFINITE_TRAIN_LOSS')
                loss.backward();torch.nn.utils.clip_grad_norm_(self.model.parameters(),5.0)
                self.optimizer.step();steps+=1;self.optimizer_steps+=1
                n=len(out);count+=n;loss_sum+=float(loss.item())*n
                del logits,loss,blocks,input_ids,output_ids
            del ymap,loader
        if limit_per_fold is None:req(count==TRN,'FULL_TR40_EPOCH_COUNT_MISMATCH')
        else:req(count==min(TRN,5*BATCH*limit_per_fold),'BOUNDED_PROBE_TRAIN_COUNT')
        torch.cuda.synchronize()
        return {'train_loss':loss_sum/count,'train_count':count,'train_steps':steps}

    def validation(self,val_y=None,limit_batches=None):
        from sklearn.metrics import average_precision_score
        torch=self.torch;dgl=self.dgl
        req(val_y is not None or limit_batches is not None,'VALIDATION_LABELS_REQUIRED_FOR_COMPLETE_VALIDATION')
        dgl.seed(90210)
        self.model.eval()
        val_ids=np.asarray(self.a['val_ids'],dtype=np.int64)
        if limit_batches is not None:val_ids=val_ids[:BATCH*limit_batches]
        loader=self.NodeDataLoader(self.g,torch.from_numpy(val_ids.copy()),self.sampler,
            batch_size=BATCH,shuffle=False,drop_last=False,num_workers=0)
        scores=np.empty(len(val_ids),dtype=np.float32);at=0
        with torch.no_grad():
            for input_ids,out_ids,blocks in loader:
                expected=val_ids[at:at+len(out_ids)]
                req(np.array_equal(out_ids.numpy(),expected),'VALIDATION_SEED_ORDER_CHANGED')
                logits=self.inputs(input_ids,out_ids,blocks,5,'val')
                p=torch.softmax(logits,dim=1)[:,1].detach().cpu().numpy()
                scores[at:at+len(p)]=p;at+=len(p)
                del logits,input_ids,out_ids,blocks
        req(at==len(val_ids) and np.isfinite(scores).all(),'INCOMPLETE_VALIDATION')
        torch.cuda.synchronize()
        ap=float(average_precision_score(val_y,scores)) if val_y is not None else None
        return ap,scores

def probe(man,artifact_sha,a,folds):
    report=OUT/'realrisk_tr40_multi_batch_probe.json'
    req(not report.exists(),'PROBE_ALREADY_DONE_INSPECT_REPORT')
    torch,dgl,psutil=initialize(SEED)
    g,x,_=graph_load(a,torch,dgl,False)
    eng=Engine(g,x,a,folds,GRID[0],torch,dgl)
    torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
    tr=eng.epoch(epoch=1,limit_per_fold=32);tt=time.perf_counter()-t
    vstart=time.perf_counter();ap,scores=eng.validation(val_y=None,limit_batches=64);vt=time.perf_counter()-vstart
    req(ap is None and len(scores)==BATCH*64,'PROBE_VAL_LABEL_PROTOCOL')
    peak=torch.cuda.max_memory_allocated()/1024**3
    rss=psutil.Process(os.getpid()).memory_info().rss/1024**3
    req(peak<38.0,'PROBE_GPU_OOM_DANGER')
    full_train_steps=sum(math.ceil(len(eng.seed_nodes[k])/BATCH) for k in range(5))
    val_steps=math.ceil(VAL/BATCH)
    estimate=tt/tr['train_steps']*full_train_steps+vt/64*val_steps
    rec={'status':'I1_RGTAN_TSOCIAL_STAGE3_COST_PROBE_PASS','model':'RGTAN','dataset':'T-Social',
         'instance':'57212396','source_sha256':SOURCE_SHA,'split_sha256':SPLIT_SHA,
         'risk_crossfit_manifest_sha256':artifact_sha,'config':GRID[0],'batch_size':BATCH,'fanouts':list(FANOUTS),
         'train_steps_sampled':tr['train_steps'],'train_nodes_sampled':tr['train_count'],
         'train_seconds':tt,'validation_unscored_batches':64,'validation_unscored_seconds':vt,
         'projected_full_epoch_seconds':estimate,'projection_warning':'Measured multi-batch extrapolation, not a full epoch; I/O and later trials may vary.',
         'expected_train_steps_per_epoch':full_train_steps,'expected_validation_steps_per_epoch':val_steps,
         'max_gpu_allocated_gib':peak,'process_rss_gib':rss,'validation_labels_indexed':False,
         'test_labels_indexed':False,'test_scored':False,'tuning_trials_completed':0,
         'optimizer_steps_diagnostic_only':eng.optimizer_steps,'checkpoint_written':False,
         'author_commit':PIN,'runner_sha256':sha(__file__)}
    atomic_json(report,rec)
    print('I1_RGTAN_TSOCIAL_TR40_MULTI_BATCH_COST_PROBE='+json.dumps(rec,sort_keys=True),flush=True)
    print('I1_RGTAN_TSOCIAL_STAGE3_COST_PROBE=PASS TEST_SCORED=NO',flush=True)

# Checkpoints at complete epoch boundaries: if a job is interrupted inside an epoch,
# --resume explicitly reloads last saved epoch. DGL's RNG state cannot be perfectly
# replayed across CUDA/DGL versions; keep exact hardware/runtime pinned.
def one_trial(man,artifact_sha,a,folds,trial,resume):
    req(2<=trial<=12,'ONLY_TRIALS_02_THROUGH_12_AUTHORIZED_IN_V2')
    probe_rec=read_json(OUT/'realrisk_tr40_multi_batch_probe.json')
    req(probe_rec.get('status')=='I1_RGTAN_TSOCIAL_STAGE3_COST_PROBE_PASS','COST_PROBE_NOT_PASS')
    folder=OUT/f'trial_{trial:02d}'
    folder.mkdir(parents=True,exist_ok=True)
    done=folder/'trial_complete.json'
    req(not done.exists(),'TRIAL_ALREADY_COMPLETE_DO_NOT_RERUN')
    latest=folder/'latest_epoch.pt';best_p=folder/'best_validation_model.pt'
    best_scores=folder/'best_validation_scores.npy';history_p=folder/'history.json'
    req(not any(folder.glob('*.partial')),'PARTIAL_ARTIFACT_REQUIRES_INSPECTION')
    if not resume:req(not latest.exists() and not best_p.exists() and not history_p.exists(),'EXISTING_TRIAL_USE_EXPLICIT_RESUME')
    else:req(latest.is_file() and history_p.is_file(),'NO_EPOCH_CHECKPOINT_TO_RESUME')
    cfg=GRID[trial-1]
    torch,dgl,psutil=initialize(SEED)
    g,x,val_y=graph_load(a,torch,dgl,True)
    req(val_y.shape==(VAL,) and set(np.unique(val_y).tolist())=={0,1},'BAD_VALIDATION_LABELS')
    eng=Engine(g,x,a,folds,cfg,torch,dgl)
    best=-1.;best_epoch=0;stale=0;start_epoch=1;hist=[]
    if resume:
        c=torch.load(latest,map_location='cpu')
        req(c['author_commit']==PIN and c['risk_crossfit_manifest_sha256']==artifact_sha
            and c['runner_sha256']==sha(__file__) and c['config']==cfg,'RECOVERY_FINGERPRINT_MISMATCH')
        hist=read_json(history_p)
        req(len(hist)==c['epoch'],'RECOVERY_HISTORY_EPOCH_MISMATCH')
        eng.model.load_state_dict(c['model']);eng.optimizer.load_state_dict(c['optimizer'])
        # CPU-loaded optimizer state must be moved to CUDA explicitly.
        for state in eng.optimizer.state.values():
            for key,value in state.items():
                if torch.is_tensor(value):state[key]=value.cuda()
        torch.set_rng_state(c['rng_torch_cpu']);torch.cuda.set_rng_state_all(c['rng_torch_cuda'])
        np.random.set_state(c['rng_numpy']);random.setstate(c['rng_python'])
        best=c['best'];best_epoch=c['best_epoch'];stale=c['stale'];start_epoch=c['epoch']+1
        eng.optimizer_steps=c['optimizer_steps']
        req(best_p.is_file() and best_scores.is_file(),'RECOVERY_MISSING_BEST_ARTIFACT')
        saved_best=torch.load(best_p,map_location='cpu')
        req(saved_best['epoch']==best_epoch and abs(saved_best['best_val_auprc']-best)<1e-10,
            'RECOVERY_BEST_CHECKPOINT_AHEAD_OR_BEHIND_LATEST_STOP_INSPECT')
        print(f'I1_RGTAN_TSOCIAL_TR40_TRIAL_{trial:02d}_RESUME=PASS starting_epoch={start_epoch} best={best:.6f}',flush=True)
    wall_start=time.perf_counter()
    for epoch in range(start_epoch,MAX_EPOCHS+1):
        st=time.perf_counter();tr=eng.epoch(epoch);train_s=time.perf_counter()-st
        st=time.perf_counter();ap,scores=eng.validation(val_y);val_s=time.perf_counter()-st
        if ap>best+1e-9:
            best=ap;best_epoch=epoch;stale=0
            torch_save(best_p,{'model':eng.model.state_dict(),'config':cfg,'epoch':epoch,
                  'best_val_auprc':ap,'author_commit':PIN,'source_sha256':SOURCE_SHA,
                  'split_sha256':SPLIT_SHA,'risk_crossfit_manifest_sha256':artifact_sha,
                  'test_scored':False,'sampler_fanouts':list(FANOUTS)},torch,overwrite=True)
            npy_save(best_scores,scores,overwrite=True)
        else:stale+=1
        h=dict(epoch=epoch,train_loss=tr['train_loss'],train_count=tr['train_count'],
            train_steps=tr['train_steps'],train_seconds=train_s,val_AUPRC=ap,val_seconds=val_s,
            best_val_AUPRC=best,stale=stale)
        hist.append(h)
        atomic_json(history_p,hist,overwrite=True)
        torch_save(latest,dict(epoch=epoch,model=eng.model.state_dict(),optimizer=eng.optimizer.state_dict(),
            config=cfg,author_commit=PIN,risk_crossfit_manifest_sha256=artifact_sha,
            runner_sha256=sha(__file__),best=best,best_epoch=best_epoch,stale=stale,
            optimizer_steps=eng.optimizer_steps,rng_torch_cpu=torch.get_rng_state(),
            rng_torch_cuda=torch.cuda.get_rng_state_all(),rng_numpy=np.random.get_state(),
            rng_python=random.getstate(),test_scored=False),torch,overwrite=True)
        print(f'I1_RGTAN_TSOCIAL_TR40_TRIAL_{trial:02d} epoch={epoch} loss={tr["train_loss"]:.5f} '
              f'val_AUPRC={ap:.6f} best={best:.6f} best_epoch={best_epoch} '
              f'train_s={train_s:.1f} val_s={val_s:.1f} peak_gib={torch.cuda.max_memory_allocated()/1024**3:.3f} '
              f'stale={stale}',flush=True)
        if stale>=PATIENCE:break
    req(best_p.is_file() and best_scores.is_file() and latest.is_file(),'MISSING_TRIAL_ARTIFACTS')
    rec=dict(status='I1_RGTAN_TSOCIAL_TR40_VALIDATION_ONLY_TRIAL_COMPLETE',
        trial=trial,model='RGTAN',dataset='T-Social',instance='57212396',config=cfg,
        max_epochs=MAX_EPOCHS,patience=PATIENCE,seed=SEED,batch=BATCH,
        epochs_completed=hist[-1]['epoch'],best_epoch=best_epoch,best_val_auprc=best,
        optimizer_steps=eng.optimizer_steps,trial_wall_seconds=time.perf_counter()-wall_start,
        author_commit=PIN,source_sha256=SOURCE_SHA,split_sha256=SPLIT_SHA,
        risk_crossfit_manifest_sha256=artifact_sha,runner_sha256=sha(__file__),
        best_model_sha256=sha(best_p),best_val_scores_sha256=sha(best_scores),
        history_sha256=sha(history_p),latest_epoch_sha256=sha(latest),
        original_full_neighbour_sampler_replaced=True,bounded_fanouts=list(FANOUTS),
        supervisor_approval_recorded=False,validation_labels_used_for_selection=True,
        test_labels_indexed=False,test_scored=False)
    atomic_json(done,rec)
    print(f'I1_RGTAN_TSOCIAL_TR40_TRIAL_{trial:02d}_VALIDATION_ONLY_COMPLETE=PASS '
          f'best_val_AUPRC={best:.6f} best_epoch={best_epoch} epochs={hist[-1]["epoch"]}',flush=True)
    print('I1_RGTAN_TSOCIAL_STAGE3_HELDOUT_TEST_EVALUATIONS=0',flush=True)
    print('I1_RGTAN_TSOCIAL_STAGE3_NEXT_GATE=REVIEW_ALL_12_TRIALS_FOR_FROZEN_WINNER',flush=True)

def check_trial1_integrity():
    folder=OUT/'trial_01';rec=read_json(folder/'trial_complete.json')
    req(rec['status']=='I1_RGTAN_TSOCIAL_TR40_VALIDATION_ONLY_TRIAL_COMPLETE','TRIAL1_BAD_STATUS')
    req(rec['trial']==1 and rec['best_epoch']==23 and rec['epochs_completed']==43,'TRIAL1_BAD_ID_EPOCH')
    req(abs(rec['best_val_auprc']-0.584989)<1e-6,'TRIAL1_METRIC_CHANGED')
    req(rec['config']==GRID[0] and rec['test_scored'] is False and rec['test_labels_indexed'] is False,'TRIAL1_PROTOCOL_CHANGED')
    mapping=[('best_validation_model.pt','best_model_sha256'),('best_validation_scores.npy','best_val_scores_sha256'),('history.json','history_sha256'),('latest_epoch.pt','latest_epoch_sha256')]
    for fn,key in mapping:req(sha(folder/fn)==rec[key],'TRIAL1_CHECKPOINT_HASH_CHANGED_'+fn)
    hist=read_json(folder/'history.json')
    req(len(hist)==43 and abs(max(x['val_AUPRC'] for x in hist)-rec['best_val_auprc'])<1e-10,'TRIAL1_HISTORY_MISMATCH')
    print('I1_RGTAN_TSOCIAL_STAGE3_TRIAL01_VERIFIED_REUSE=PASS',flush=True)

def check_done(trial,artifact_sha):
    folder=OUT/f'trial_{trial:02d}'
    done=folder/'trial_complete.json'
    if not done.exists():return False
    rec=read_json(done)
    req(rec['status']=='I1_RGTAN_TSOCIAL_TR40_VALIDATION_ONLY_TRIAL_COMPLETE' and rec['trial']==trial,'BAD_COMPLETE_RECORD')
    req(rec['config']==GRID[trial-1] and rec['seed']==2 and rec['batch']==1024,'COMPLETE_CONFIG_CHANGED')
    req(rec['risk_crossfit_manifest_sha256']==artifact_sha and rec['runner_sha256']==sha(__file__),'COMPLETE_RUNNER_FINGERPRINT_CHANGED')
    req(rec['test_scored'] is False and rec['test_labels_indexed'] is False,'COMPLETE_LABEL_PROTOCOL_CHANGED')
    for fn,key in [('best_validation_model.pt','best_model_sha256'),('best_validation_scores.npy','best_val_scores_sha256'),('history.json','history_sha256'),('latest_epoch.pt','latest_epoch_sha256')]:
        req(sha(folder/fn)==rec[key],'COMPLETE_HASH_CHANGED_'+str(trial)+'_'+fn)
    hist=read_json(folder/'history.json')
    req(len(hist)==rec['epochs_completed'],'COMPLETE_HISTORY_LENGTH')
    req(abs(max(x['val_AUPRC'] for x in hist)-rec['best_val_auprc'])<1e-10,'COMPLETE_HISTORY_METRIC')
    print(f'I1_RGTAN_TSOCIAL_TR40_TRIAL_{trial:02d}_VERIFIED_REUSE=PASS val_AUPRC={rec["best_val_auprc"]:.6f}',flush=True)
    return True

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('selftest','verify','trial','resume-trial','all'),required=True)
    p.add_argument('--trial',type=int,default=None)
    args=p.parse_args()
    if args.mode=='selftest':unit_selftest();return
    req(args.trial is None if args.mode in ('verify','all') else 2<=args.trial<=12,'TRIAL_ARGUMENT_INVALID')
    man,fingerprint=check_readiness(deep=args.mode=='verify')
    check_trial1_integrity()
    if args.mode=='verify':
        print('I1_RGTAN_TSOCIAL_STAGE3_GRID_V2_VERIFY=PASS TEST_SCORED=NO',flush=True);return
    a,folds=load_arrays()
    if args.mode=='all':
        for t in range(2,13):
            if check_done(t,fingerprint):continue
            folder=OUT/f'trial_{t:02d}'
            resume=(folder/'latest_epoch.pt').exists()
            one_trial(man,fingerprint,a,folds,t,resume)
            req(check_done(t,fingerprint),'TRIAL_NOT_COMPLETE_AFTER_RUN')
        scores={}
        for t in range(1,13):
            r=read_json(OUT/f'trial_{t:02d}'/'trial_complete.json')
            scores[t]=r['best_val_auprc']
        w=max(scores,key=lambda t:(scores[t],-t))
        summary=dict(status='I1_RGTAN_TSOCIAL_TR40_12_VALIDATION_TUNING_TRIALS_FROZEN',
           winner_trial=w,winner_val_auprc=scores[w],winner_config=GRID[w-1],
           trials=12,validation_only=True,test_scored=False,test_evaluations=0,
           author_commit=PIN,source_sha256=SOURCE_SHA,split_sha256=SPLIT_SHA,
           risk_crossfit_manifest_sha256=fingerprint,runner_v2_sha256=sha(__file__),
           trials_sha256={str(t):sha(OUT/f'trial_{t:02d}'/'trial_complete.json') for t in range(1,13)},
           original_full_neighbour_sampler_replaced=True,supervisor_approval_recorded=False)
        out=OUT/'I1_RGTAN_TSocial_Stage3_12ValidationTrialsFrozen.json'
        if out.exists():req(read_json(out)==summary,'FROZEN_SUMMARY_CHANGED')
        else:atomic_json(out,summary)
        print(f'I1_RGTAN_TSOCIAL_STAGE3_12_TRIALS_FROZEN=PASS WINNER={w} AUPRC={scores[w]:.6f} TEST_SCORED=NO',flush=True)
    else:
        t=args.trial
        req(not check_done(t,fingerprint),'TRIAL_ALREADY_COMPLETE_USE_ALL_FOR_VERIFIED_REUSE')
        one_trial(man,fingerprint,a,folds,t,args.mode=='resume-trial')

if __name__=='__main__':
    try:main()
    except BaseException as e:
        print('I1_RGTAN_TSOCIAL_TR40_GRID_V2_STOP='+repr(e),flush=True)
        raise
