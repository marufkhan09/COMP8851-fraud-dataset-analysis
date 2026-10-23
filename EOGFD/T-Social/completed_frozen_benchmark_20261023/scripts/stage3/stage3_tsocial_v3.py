#!/usr/bin/env python3
"""EOGFDHeteroModel x canonical T-Social, process-scoped elite-only selection and streamed ES fitness in both phases.
Stage 3 ONLY: TR40 seed2 validation-only tuning with streamed ES fitness in train+val, 12 configs, no test scoring.
Single writer + durable epoch recovery; tuning winner frozen only after 12.
This is a unified frozen-split adapter with documented selector memory optimization; NOT unmodified author execution.
"""
from __future__ import annotations
import argparse, csv, datetime, gc, hashlib, json, os, random, socket, statistics, subprocess, sys, time, traceback, uuid, types
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from dgl.data.utils import load_graphs
from sklearn.metrics import average_precision_score, roc_auc_score

BASE=Path('/workspace/comp8851')
KIT=Path('/workspace/EOGFD_DUAL_I1_I2_20261023')
SRC=BASE/'assets/tsocial/tsocial'
SPLIT=BASE/'assets/tsocial/tsocial_seed2_nested_splits.npz'
PILOT=BASE/'diagnostics/EOGFD_57231045_tsocial/scoped_elite_only_fullgraph_train_update.json'
AUTHOR=BASE/'repos/EOGFD'
RESULTS=BASE/'results/eogfd/tsocial/stage3_scoped_elite_plus_streamed_train_eval_tr40_tuning'
MEMDIAG=Path('/workspace/EOGFD_TSOCIAL_I2_MEMDIAG_20261023/tsocial_select_diagnostic.py')
MEMDIAG_SHA='8f324b22a3ff5bbf3df53fc15552cd2f00c518d742c8693bfafd5dcc56db18af'
MODE='UNIFIED_FROZEN_SPLITS_SCOPED_ELITE_PLUS_STREAMED_TRAIN_EVAL_IMPL_OPT_NOT_AUTHOR_UNMODIFIED'
SOURCE_SHA='8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0'
SPLIT_SHA='b7e76835612e72a8fe5c69bbe87c8ea7b2e5790cbd8849e2a3ac776364bc6519'
PIN='a4726f173bc8c923109ddcb2786f4d9c5206ae7f'
EVAL_SEED=100921
MAX_EPOCHS=100
PATIENCE=20
RATIOS=('TR40','TR30','TR20','TR10')
SEEDS=(2,42,72)
GRID=[dict(lr=lr,population_size=pop,mutation_rate=mutation,
           hidden_dim=10,order=8,beta_p=1,weight_decay=0.,lamb=0.)
      for lr in (.001,.003,.01)
      for pop in (10,20)
      for mutation in (.05,.10)]
assert len(GRID)==12

def log(*args):print(*args,flush=True)
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''): h.update(block)
    return h.hexdigest()

def write_json_atomic(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.writing.'+uuid.uuid4().hex)
    with tmp.open('x') as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def create_or_identical(path,obj):
    if path.exists():
        assert json.loads(path.read_text())==obj,('CONFLICT_EXISTING_RECORD',str(path))
    else:write_json_atomic(path,obj)

def pt_atomic(path,obj):
    tmp=path.with_name(path.name+'.writing.'+uuid.uuid4().hex)
    with tmp.open('xb') as f:
        torch.save(obj,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def npy_atomic(path,arr):
    tmp=path.with_name(path.name+'.writing.'+uuid.uuid4().hex)
    with tmp.open('xb') as f:
        np.save(f,arr,allow_pickle=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def set_seeds(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)

def model_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k!='theta_last'}

def restore_state(model,state):
    r=model.load_state_dict(state,strict=False)
    assert r.missing_keys==['theta_last'] and not r.unexpected_keys,r

def decode_only(raw,ids):
    idx=torch.from_numpy(np.asarray(ids,dtype=np.int64).copy())
    if raw.shape==(5781065,2):raise RuntimeError('STOP_UNEXPECTED_SOCIAL_LABEL_ENCODING')
    elif raw.shape in ((5781065,),(5781065,1)):y=raw[idx].long().view(-1)
    else:raise RuntimeError(('STOP_UNEXPECTED_LABEL_ENCODING',tuple(raw.shape)))
    assert set(torch.unique(y).tolist())=={0,1},'SPLIT_LABELS_NOT_BOTH_CLASSES'
    return y

def validate_assets():
    assert socket.gethostname()=='a92ee6fa0813',('STOP_WRONG_I2_HOST',socket.gethostname())
    assert KIT.is_dir() and (KIT/'preflight_pilot.py').is_file()
    assert MEMDIAG.is_file()
    assert torch.__version__=='1.12.1+cu113' and dgl.__version__=='0.8.1'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    free,total=torch.cuda.mem_get_info()
    assert free>44*2**30,('STOP_GPU_NOT_IDLE_ENOUGH_FOR_36_GIB_TRAINING',free/2**30)
    assert subprocess.check_output(['git','-C',str(AUTHOR),'rev-parse','HEAD'],text=True).strip()==PIN
    assert not subprocess.check_output(['git','-C',str(AUTHOR),'status','--porcelain'],text=True).strip()
    assert digest(SRC)==SOURCE_SHA, 'STOP SOURCE HASH CHANGED'
    assert digest(SPLIT)==SPLIT_SHA, 'STOP FROZEN SPLIT HASH CHANGED'
    pilot=json.loads(PILOT.read_text())
    assert pilot['status']=='SCOPED_ELITE_ONLY_OPTIMIZATION_TRAIN_ONLY_FULLGRAPH_ONE_UPDATE_PASS'
    assert pilot['source_sha256']==SOURCE_SHA and pilot['split_sha256']==SPLIT_SHA
    assert pilot['upstream_commit']==PIN and pilot['test_scored'] is False
    assert pilot['runtime_optimizer_select_implementation_modified'] is True and pilot['original_model_repository_unchanged'] is True
    assert digest(MEMDIAG)==MEMDIAG_SHA,'STOP_MEMORY_DIAGNOSTIC_SCRIPT_HASH_CHANGED'
    assert pilot['polynomial_order']==8 and pilot['hidden_dim']==10 and pilot['population_size']==20
    with np.load(SPLIT,allow_pickle=False) as z:
        ids={k:np.asarray(z[k],dtype=np.int64).copy() for k in ('TR10','TR20','TR30','TR40','val','test')}
        for key in ('seed','split_seed'):
            if key in z.files:assert int(z[key].item())==2
    expected={'TR10':578106,'TR20':1156212,'TR30':1734319,'TR40':2312426,'val':1156213,'test':2312426}
    for key,num in expected.items():
        a=ids[key];assert len(a)==num and (a>=0).all() and (a<5781065).all()
        assert len(np.unique(a))==num
    for small,large in [('TR10','TR20'),('TR20','TR30'),('TR30','TR40')]:assert np.isin(ids[small],ids[large]).all()
    for a,b in [('TR40','val'),('TR40','test'),('val','test')]:assert not np.intersect1d(ids[a],ids[b],assume_unique=True).size
    log('I2_TSOCIAL_TUNING_CANONICAL_INPUTS=PASS TEST_SCORED=NO')
    return ids



def install_scoped_optimizer():
    # Import the exact script that passed I2's recorded one-update fullgraph pilot.
    # This performs small-tensor CPU/CUDA equivalence and RNG-state checks,
    # then patches only the current Python process; the git repo is unchanged.
    import importlib.util
    assert digest(MEMDIAG)==MEMDIAG_SHA
    spec=importlib.util.spec_from_file_location('eogfd_tsocial_pinned_memory_diagnostic',MEMDIAG)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.install_select_override()
    from models.eogfd import EvolutionStrategyThetaOptimizer
    assert EvolutionStrategyThetaOptimizer.select.__name__=='select_only_used_elite'
    log('I2_TSOCIAL_STAGE3_PROCESS_LOCAL_SELECTOR_EQUIVALENCE=PASS')

def dataset_gpu(ids):
    grs,_=load_graphs(str(SRC));assert len(grs)==1
    graph=grs[0]
    assert graph.num_nodes()==5781065 and graph.num_edges()==146211016
    assert graph.canonical_etypes==[('_U','_V','_U')], ('STOP_GRAPH_ETYPES_CHANGED',graph.canonical_etypes)
    assert len(graph.ntypes)==1 and len(graph.canonical_etypes)==1
    assert 'feature' in graph.ndata and 'label' in graph.ndata
    features=graph.ndata.pop('feature').float()
    raw=graph.ndata.pop('label')
    ytrain=decode_only(raw,ids['TR40']);yval=decode_only(raw,ids['val'])
    assert int(ytrain.sum())==69712 and int(yval.sum())==34856
    del raw
    for key in list(graph.ndata.keys()):graph.ndata.pop(key)
    for key in list(graph.edata.keys()):graph.edata.pop(key)
    assert not list(graph.ndata) and not list(graph.edata)
    assert features.shape==(5781065,10) and bool(torch.isfinite(features).all())
    graph=graph.to('cuda:0');features=features.to('cuda:0')
    train_ix=torch.from_numpy(ids['TR40']).long().to('cuda:0')
    val_ix=torch.from_numpy(ids['val']).long().to('cuda:0')
    ytrain=ytrain.to('cuda:0');yval=yval.cpu().numpy().astype(np.int64)
    weight=torch.tensor([1.0,(2312426-69712)/69712],device='cuda:0')
    log('I2_TSOCIAL_TUNING_GRAPH_READY=PASS nodes=5781065 original_edges=146211016 relation_count=1 train=2312426 val=1156213 test_labels_indexed=NO')
    return dict(graph=graph,features=features,train_ix=train_ix,val_ix=val_ix,ytrain=ytrain,yval=yval,weight=weight)

def metrics(y,scores):
    y=np.asarray(y,dtype=np.int64).reshape(-1)
    scores=np.asarray(scores,dtype=np.float64).reshape(-1)
    assert len(y)==len(scores) and np.isfinite(scores).all()
    assert np.all((scores>=0)&(scores<=1)) and set(np.unique(y).tolist())=={0,1}
    ts=np.arange(1,100,dtype=np.float64)/100
    positive=y.astype(bool)[:,None];pred=scores[:,None]>ts[None,:]
    tp=np.sum(pred&positive,axis=0);tn=np.sum((~pred)&(~positive),axis=0)
    fp=np.sum(pred&(~positive),axis=0);fn=np.sum((~pred)&positive,axis=0)
    fpos=2*tp/np.maximum(2*tp+fp+fn,1)
    fneg=2*tn/np.maximum(2*tn+fp+fn,1)
    macro=(fpos+fneg)/2;recall=tp/np.maximum(tp+fn,1)
    ix=max(range(99),key=lambda i:(round(float(macro[i]),12),round(float(recall[i]),12),-abs(float(ts[i])-.5),-float(ts[i])))
    return dict(val_auprc=float(average_precision_score(y,scores)),val_auroc=float(roc_auc_score(y,scores)),
                val_macro_f1=float(macro[ix]),val_fraud_recall=float(recall[ix]),threshold=float(ts[ix]))

def stream_fitness(self, bases, population, target):
    """Equivalent per-candidate fitness formula, bounded intermediate memory.

    Only called inside no_grad evaluation; candidates and all random operations
    still use original upstream implementation. p-by-N-by-F tensors for the
    *intermediate projections* are not allocated simultaneously.
    Floating-point kernel scheduling may differ at the last few bits.
    """
    assert not torch.is_grad_enabled(), 'STOP_STREAM_FITNESS_GRAD_ENABLED'
    assert population.ndim==3 and bases.ndim==3
    assert bases.shape[0]==population.shape[1] and bases.shape[1]==population.shape[2]
    assert target.shape==(population.shape[1],bases.shape[2])
    values=[]
    target3=target.unsqueeze(0)
    for j in range(population.shape[0]):
        poly=torch.einsum('nkf,pnk->pnf',bases,population[j:j+1])
        feats=self.conv.linear(poly)
        fitness=F.mse_loss(feats,target3,reduction='none').mean(dim=(1,2))
        values.append(fitness)
    return torch.cat(values,dim=0)


def streamed_train_eval_evolutionary_step(self, graph, h, base_theta):
    """Upstream evolutionary_step equation, with P-wise fitness evaluation.

    This is a process-scoped documented training/validation memory-engineering adaptation.
    Upstream model source remains untouched, node/edge/order/pop unchanged.
    """
    assert not torch.is_grad_enabled(), 'STOP_STREAM_EVO_GRAD_ENABLED'
    bases=self.conv.compute_bases(graph,h,self.d)
    population=self.es_optimizer.initialize_population(base_theta)
    # Original call order and pseudorandom-number operations preserved.
    fitness=stream_fitness(self,bases,population,h)
    donors=self.es_optimizer.mutate(population)
    perm=torch.randperm(population.size(0),device=population.device)
    trials=self.es_optimizer.crossover(population,donors[perm])
    fitness_trials=stream_fitness(self,bases,trials,h)
    combined_pop=torch.cat([trials,base_theta.unsqueeze(0)],dim=0)
    combined_fit=torch.cat([fitness_trials,fitness.mean().unsqueeze(0)],dim=0)
    best_theta=self.es_optimizer.select(combined_pop,combined_fit,elite_num=1)[0]
    return best_theta


def assert_streamed_equivalence_on_small_graphs():
    """Verify original-vs-streamed theta selection + RNG on CPU and CUDA."""
    from models.eogfd import EOGFDHeteroModel
    for device in ('cpu','cuda:0'):
        # Checks use same pinned upstream DGL kernels on the actual installed env.
        n=37
        edges_src=torch.arange(n,dtype=torch.int64)
        edges_dst=(edges_src*7+3)%n
        graph=dgl.heterograph({('_U','_V','_U'):(edges_src,edges_dst)},num_nodes_dict={'_U':n}).to(device)
        for seed in (2,42,72):
            set_seeds(seed)
            model=EOGFDHeteroModel(in_feats=10,hid_feats=10,num_classes=2,d=8,beta_p=1,population_size=20,mutation_rate=.1).to(device)
            h=torch.randn(n,10,device=device)
            theta=F.softmax(model.theta_generator(h),dim=1)
            sub=graph[graph.canonical_etypes[0]]
            cpu_before=torch.get_rng_state().clone()
            cuda_before=torch.cuda.get_rng_state(0).clone()
            with torch.no_grad():
                original=model.evolutionary_step(sub,h,theta)
            cpu_original_after=torch.get_rng_state().clone()
            cuda_original_after=torch.cuda.get_rng_state(0).clone()
            torch.set_rng_state(cpu_before);torch.cuda.set_rng_state(cuda_before,device=0)
            with torch.no_grad():
                streamed=streamed_train_eval_evolutionary_step(model,sub,h,theta)
            assert torch.allclose(original,streamed,rtol=1e-5,atol=1e-6),('STOP_STREAMED_THETA_DIFF',device,seed,float((original-streamed).abs().max().item()))
            assert torch.equal(torch.get_rng_state(),cpu_original_after),('STOP_STREAMED_CPU_RNG_CHANGED',device,seed)
            assert torch.equal(torch.cuda.get_rng_state(0),cuda_original_after),('STOP_STREAMED_CUDA_RNG_CHANGED',device,seed)
    log('I2_TSOCIAL_STREAMED_EVAL_EVOLUTION_SMALL_GRAPH_EQUIVALENCE=PASS devices=CPU_CUDA seeds=2_42_72')



def assert_streamed_forward_backward_equivalence():
    """Small-graph equivalence including author forward, loss, backward and RNG.

    Original comparison already uses the independently verified elite-only
    selector in both paths. This tests the additional candidate-streaming change.
    """
    import copy
    from models.eogfd import EOGFDHeteroModel
    for device in ('cpu','cuda:0'):
        n=31
        s=torch.arange(n,dtype=torch.int64)
        d=(s*13+4)%n
        g=dgl.heterograph({('_U','_V','_U'):(s,d)},num_nodes_dict={'_U':n}).to(device)
        for seed in (2,42,72):
            set_seeds(seed)
            model0=EOGFDHeteroModel(in_feats=10,hid_feats=10,num_classes=2,d=8,beta_p=1,
                    population_size=20,mutation_rate=.1).to(device)
            model1=copy.deepcopy(model0)
            model1.evolutionary_step=types.MethodType(streamed_train_eval_evolutionary_step,model1)
            feat=torch.randn(n,10,device=device)
            y=torch.randint(0,2,(n,),device=device)
            cpu_before=torch.get_rng_state().clone()
            cuda_before=torch.cuda.get_rng_state(0).clone()
            out0,_=model0(g,feat)
            loss0=F.cross_entropy(out0,y)
            loss0.backward()
            cpu_after=torch.get_rng_state().clone()
            cuda_after=torch.cuda.get_rng_state(0).clone()
            torch.set_rng_state(cpu_before);torch.cuda.set_rng_state(cuda_before,device=0)
            out1,_=model1(g,feat)
            loss1=F.cross_entropy(out1,y)
            loss1.backward()
            assert torch.allclose(out0,out1,atol=1e-5,rtol=1e-5),('STOP_FWD_EQ',device,seed,
                        float((out0-out1).abs().max().item()))
            assert torch.allclose(loss0,loss1,atol=1e-5,rtol=1e-5),('STOP_LOSS_EQ',device,seed)
            for (key0,p0),(key1,p1) in zip(model0.named_parameters(),model1.named_parameters()):
                assert key0==key1,('STOP_PARAM_NAMES',key0,key1)
                assert p0.grad is not None and p1.grad is not None,('STOP_PARAM_GRAD_NONE',key0)
                assert torch.allclose(p0.grad,p1.grad,atol=1e-4,rtol=1e-4),('STOP_GRAD_EQ',device,seed,key0)
            assert torch.equal(torch.get_rng_state(),cpu_after),('STOP_FWD_CPU_RNG',device,seed)
            assert torch.equal(torch.cuda.get_rng_state(0),cuda_after),('STOP_FWD_CUDA_RNG',device,seed)
    log('I2_TSOCIAL_SMALL_GRAPH_FORWARD_BACKWARD_EQUIVALENCE=PASS CPU_CUDA_SEEDS=2_42_72')

def eval_val(model,b):
    model.eval()
    assert model.__dict__.get('evolutionary_step') is not None, 'STOP_STREAMED_MODEL_SCOPE_MISSING'
    assert model.evolutionary_step.__func__ is streamed_train_eval_evolutionary_step, 'STOP_WRONG_MEMORY_OPT'
    # The same process-local no-grad streamed ES fitness operates during both
    # training and validation; validation RNG is isolated from the training stream.
    torch.cuda.synchronize()
    with torch.random.fork_rng(devices=[0],enabled=True):
        torch.manual_seed(EVAL_SEED);torch.cuda.manual_seed_all(EVAL_SEED)
        with torch.no_grad():
            logits,_=model(b['graph'],b['features'])
            scores=torch.softmax(logits[b['val_ix']],dim=1)[:,1].detach().cpu().numpy().astype(np.float64)
            del logits
    assert len(scores)==1156213
    return metrics(b['yval'],scores),scores

def build_model(cfg):
    from models.eogfd import EOGFDHeteroModel
    model=EOGFDHeteroModel(in_feats=10,hid_feats=cfg['hidden_dim'],num_classes=2,
        d=cfg['order'],beta_p=cfg['beta_p'],population_size=cfg['population_size'],
        mutation_rate=cfg['mutation_rate']).to('cuda:0')
    # Bind only on this in-memory instance. Does not touch pinned author .py.
    model.evolutionary_step=types.MethodType(streamed_train_eval_evolutionary_step,model)
    assert model.evolutionary_step.__func__ is streamed_train_eval_evolutionary_step
    return model

def epoch_csv(path,history):
    tmp=path.with_name(path.name+'.writing.'+uuid.uuid4().hex)
    with tmp.open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['epoch','training_seconds','validation_seconds','cumulative_training_seconds'])
        cumulative=0.
        for h in history:
            cumulative+=h['training_seconds'];w.writerow([h['epoch'],f'{h["training_seconds"]:.8f}',f'{h["validation_seconds"]:.8f}',f'{cumulative:.8f}'])
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def trial_run(i,b):
    cfg=GRID[i-1];name=f'trial_{i:02d}'
    dest=RESULTS/name;dest.mkdir(parents=True,exist_ok=True)
    final=dest/'trial_result.json';recfile=dest/'epoch_recovery.pt'
    ckpt=dest/'best_validation.pt';scofile=dest/'best_validation_scores.npy'
    historyfile=dest/'validation_history.json';csvfile=dest/'epoch_times.csv'
    if final.exists():
        obj=json.loads(final.read_text())
        assert obj['trial']==i and obj['config']==cfg and obj['source_sha256']==SOURCE_SHA and obj['split_sha256']==SPLIT_SHA
        assert obj['test_scored'] is False and obj['test_labels_indexed'] is False
        for p,k in ((ckpt,'checkpoint_sha256'),(scofile,'validation_scores_sha256'),(historyfile,'history_sha256'),(csvfile,'epoch_times_sha256')):
            assert p.is_file() and digest(p)==obj[k],('STOP_COMPLETE_TRIAL_BAD_ARTIFACT',str(p))
        if recfile.exists():
            old=torch.load(recfile,map_location='cpu')
            assert old['config']==cfg and old['source_sha256']==SOURCE_SHA and old['split_sha256']==SPLIT_SHA
            assert old['test_scored'] is False and len(old['history'])==obj['epochs_completed']
            recfile.unlink()
            log(f'I2_TSOCIAL_{name}_RECOVERY_AFTER_COMPLETE_CLEANUP=PASS')
        log(f'I2_TSOCIAL_STAGE3_{name}_VERIFIED_REUSE=PASS AUPRC={obj["best_val_auprc"]:.6f}')
        return obj
    assert not any(p.exists() for p in (ckpt,scofile,historyfile,csvfile)), ('STOP_PARTIAL_FINAL_WITHOUT_TRIAL_JSON',name)
    set_seeds(2)
    model=build_model(cfg)
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],betas=(.9,.999),eps=1e-8,weight_decay=cfg['weight_decay'])
    start=1;stale=0;best=-1.;best_epoch=None;best_metrics=None;best_state=None;best_scores=None;history=[]
    if recfile.exists():
        r=torch.load(recfile,map_location='cpu')
        assert r['config']==cfg and r['source_sha256']==SOURCE_SHA and r['split_sha256']==SPLIT_SHA
        assert r['test_scored'] is False and r['test_labels_indexed'] is False
        restore_state(model,r['model_state']);opt.load_state_dict(r['optimizer_state'])
        for state in opt.state.values():
            for k,v in state.items():
                if isinstance(v,torch.Tensor):state[k]=v.to('cuda:0')
        start=r['epoch']+1;stale=r['stale'];best=r['best'];best_epoch=r['best_epoch']
        best_metrics=r['best_metrics'];best_state=r['best_state'];best_scores=r['best_scores'];history=r['history']
        torch.set_rng_state(r['cpu_torch_rng']);torch.cuda.set_rng_state(r['cuda_torch_rng'],device='cuda:0')
        np.random.set_state(r['numpy_rng']);random.setstate(r['python_rng'])
        assert len(history)==start-1 and (best_state is not None)==(best_epoch is not None)
        log(f'I2_TSOCIAL_{name}_RECOVERY_EPOCH_START={start}')
    torch.cuda.reset_peak_memory_stats()
    tic=time.perf_counter()
    for epoch in range(start,MAX_EPOCHS+1):
        if stale>=PATIENCE:break
        model.train();opt.zero_grad(set_to_none=True);ts=time.perf_counter()
        logits,_=model(b['graph'],b['features'])
        loss=F.cross_entropy(logits[b['train_ix']],b['ytrain'],weight=b['weight'])
        assert bool(torch.isfinite(loss).item()),('STOP_NONFINITE_TRAIN_LOSS',i,epoch)
        loss.backward();opt.step();torch.cuda.synchronize()
        train_s=time.perf_counter()-ts;vloss=float(loss.detach().cpu().item())
        del logits,loss
        # Release entirely dead transient GPU blocks before the next ES pass;
        # does not change graph, gradients, parameter values, or RNG state.
        torch.cuda.empty_cache()
        tv=time.perf_counter();val,scores=eval_val(model,b);torch.cuda.synchronize();val_s=time.perf_counter()-tv
        improved=val['val_auprc']>best+1e-12
        if improved:
            best=val['val_auprc'];best_epoch=epoch;best_metrics=val;best_state=model_state(model);best_scores=scores.copy();stale=0
        else:stale+=1
        history.append(dict(epoch=epoch,train_loss=vloss,training_seconds=train_s,validation_seconds=val_s,
                            improved=bool(improved),best_so_far=float(best),stale=stale,**val))
        prog=dict(epoch=epoch,config=cfg,source_sha256=SOURCE_SHA,split_sha256=SPLIT_SHA,
          model_state=model_state(model),optimizer_state=opt.state_dict(),history=history,
          best=best,best_epoch=best_epoch,best_metrics=best_metrics,best_state=best_state,best_scores=best_scores,
          stale=stale,cpu_torch_rng=torch.get_rng_state(),cuda_torch_rng=torch.cuda.get_rng_state(device='cuda:0'),
          numpy_rng=np.random.get_state(),python_rng=random.getstate(),test_scored=False,test_labels_indexed=False)
        pt_atomic(recfile,prog)
        del prog,scores
        torch.cuda.empty_cache()
        peak=torch.cuda.max_memory_allocated()/2**30
        if epoch in (1,2) or epoch%10==0 or stale>=PATIENCE:
            log(f'I2_TSOCIAL_TRIAL_{i:02d} epoch={epoch} loss={vloss:.5f} val_AUPRC={val["val_auprc"]:.6f} best={best:.6f} train_s={train_s:.3f} val_s={val_s:.3f} peak_gib={peak:.3f} stale={stale}')
        if peak>43:raise RuntimeError(('STOP_UNSAFE_GPU_MEMORY_ALLOCATED_GIB',peak))
    assert best_state is not None and best_scores is not None and best_epoch is not None
    # The snapshot used for the winner is the exact validation-selected best-epoch model.
    pt_atomic(ckpt,dict(model_state=best_state,epoch=best_epoch,config=cfg,
             dataset='T-Social',ratio='TR40',seed=2,source_sha256=SOURCE_SHA,split_sha256=SPLIT_SHA,
             upstream_commit=PIN,validation=best_metrics,theta_last_buffer_excluded=True,
             test_scored=False,test_labels_indexed=False))
    npy_atomic(scofile,best_scores)
    write_json_atomic(historyfile,history)
    epoch_csv(csvfile,history)
    obj=dict(dataset='T-Social',model='EOGFDHeteroModel',trial=i,ratio='TR40',seed=2,
       config=cfg,upstream_commit=PIN,source_sha256=SOURCE_SHA,split_sha256=SPLIT_SHA,
       mode=MODE,
       venus_methodological_approval_recorded=False,
       best_val_auprc=best,best_epoch=best_epoch,epochs_completed=len(history),
       best_validation_metrics=best_metrics,
       checkpoint_sha256=digest(ckpt),validation_scores_sha256=digest(scofile),
       history_sha256=digest(historyfile),epoch_times_sha256=digest(csvfile),
       runtime_select_override=True,scoped_memory_optimizer=True,streamed_fitness_validation_only=True,
       streamed_fitness_training_and_validation=True,
       original_repository_unchanged=True,per_epoch_timings_preserved=True,max_epochs=MAX_EPOCHS,patience=PATIENCE,
       elapsed_seconds=sum(h['training_seconds']+h['validation_seconds'] for h in history),
       peak_gpu_allocated_gib=float(torch.cuda.max_memory_allocated()/2**30),
       fixed_eval_rng_seed=EVAL_SEED,threshold_selection='validation_macroF1_then_fraud_recall_then_nearest_0.5',
       selection_metric='validation_AUPRC',test_labels_indexed=False,test_scored=False)
    write_json_atomic(final,obj)
    if recfile.exists():recfile.unlink()
    log(f'I2_TSOCIAL_STAGE3_TRIAL_{i:02d}_VALIDATION_ONLY_COMPLETE=PASS best_val_AUPRC={best:.6f} best_epoch={best_epoch} epochs={len(history)}')
    log(f'I2_TSOCIAL_STAGE3_TRIAL_{i:02d}_MEASURED_EPOCH_MEDIAN_TRAIN_SECONDS={statistics.median(h["training_seconds"] for h in history):.3f}')
    log(f'I2_TSOCIAL_STAGE3_TRIAL_{i:02d}_MEASURED_EPOCH_MEDIAN_VALIDATION_SECONDS={statistics.median(h["validation_seconds"] for h in history):.3f}')
    del model,opt;gc.collect();torch.cuda.empty_cache()
    return obj

def main():
    p=argparse.ArgumentParser(description='I2 EOGFD T-Social Stage3 validation-only with documented scoped selector optimization; no test access')
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--trial',type=int,choices=range(1,13),metavar='1..12')
    g.add_argument('--all',action='store_true')
    args=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0',('STOP_ASSIGN_ONE_GPU_EXPLICITLY',os.environ.get('CUDA_VISIBLE_DEVICES'))
    torch.set_num_threads(min(8,os.cpu_count() or 2))
    ids=validate_assets()
    install_scoped_optimizer()
    assert_streamed_equivalence_on_small_graphs()
    assert_streamed_forward_backward_equivalence()
    gc.collect();torch.cuda.empty_cache()
    RESULTS.mkdir(parents=True,exist_ok=True)
    m=dict(dataset='T-Social',mode=MODE,
        original_eogfd_author_commit=PIN,source_sha256=SOURCE_SHA,split_sha256=SPLIT_SHA,
        split_seed=2,tuning_seed=2,ratios=list(RATIOS),final_seeds=list(SEEDS),grid=GRID,
        max_epochs=MAX_EPOCHS,patience=PATIENCE,eval_seed=EVAL_SEED,
        selection='TR40_seed2_validation_AUPRC',test_labels_indexed=False,test_scored=False,
        original_native_relation_count=1,original_repository_unchanged=True,runtime_select_override=True,
        memory_diagnostic_sha256=MEMDIAG_SHA,streamed_fitness_validation_only=True,
        streamed_fitness_chunk_population_size=1,
        streamed_fitness_training_and_validation=True,
        note='Input graph and nested splits verified. Process-local elite-only selection, plus train-and-validation per-candidate streamed population fitness. Small-graph theta and RNG-state tested. Full-graph bitwise equivalence not claimed. Author repository unchanged; not unmodified author-run.')
    create_or_identical(RESULTS/'stage3_run_manifest.json',m)
    if args.all:
        gate=RESULTS/'trial_08'/'trial_result.json'
        assert gate.is_file(), 'STOP: run --trial8 complete validation gate first'
        gr=json.loads(gate.read_text())
        assert gr['trial']==8 and gr['test_scored'] is False
        assert gr['runtime_select_override'] is True and gr['peak_gpu_allocated_gib']<43
        log('I2_TSOCIAL_STAGE3_ONE_COMPLETE_TRIAL_GATE_ALREADY_SATISFIED=PASS')
    if (RESULTS/'EOGFD_TSocial_TR40_FrozenWinner.json').exists() and args.trial:
        raise RuntimeError('STOP WINNER ALREADY FROZEN; DO NOT CREATE NEW TRIAL')
    attempt_name=(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex)
    attempt_path=RESULTS/'attempts'/(attempt_name+'.json')
    attempt_record={'attempt_id':attempt_name,'scope':'all' if args.all else f'trial_{args.trial:02d}',
      'dataset':'T-Social','status':'STARTED','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'source_sha256':SOURCE_SHA,'split_sha256':SPLIT_SHA,'test_scored':False}
    write_json_atomic(attempt_path,attempt_record)
    try:
        b=dataset_gpu(ids)
        which=range(1,13) if args.all else [args.trial]
        for i in which:trial_run(i,b)
        attempt_record['status']='COMPLETED'
    except BaseException as err:
        attempt_record['status']='FAILED_OR_INTERRUPTED'
        attempt_record['error']=repr(err)
        write_json_atomic(attempt_path,attempt_record)
        raise
    write_json_atomic(attempt_path,attempt_record)
    if args.all:
        results=[json.loads((RESULTS/f'trial_{i:02d}'/'trial_result.json').read_text()) for i in range(1,13)]
        assert all(r['test_scored'] is False for r in results)
        winner=max(results,key=lambda r:(r['best_val_auprc'],-r['trial']))
        frozen=dict(status='TR40_12_VALIDATION_ONLY_TRIALS_COMPLETE',dataset='T-Social',
           upstream_commit=PIN,source_sha256=SOURCE_SHA,split_sha256=SPLIT_SHA,
           winner_trial=winner['trial'],winner_config=winner['config'],
           winner_val_auprc=winner['best_val_auprc'],winner_checkpoint_sha256=winner['checkpoint_sha256'],
           selection='validation_AUPRC',ratio='TR40',training_seed=2,
           final_ratios=list(RATIOS),final_seeds=list(SEEDS),
           tuning_trials_completed=12,test_scored=False,
           author_identical_reproduction=False,scoped_memory_optimizer=True,venus_methodological_approval_recorded=False)
        create_or_identical(RESULTS/'EOGFD_TSocial_TR40_FrozenWinner.json',frozen)
        log(f'I2_TSOCIAL_STAGE3_12_TRIALS_FROZEN=PASS WINNER_TRIAL={winner["trial"]} VAL_AUPRC={winner["best_val_auprc"]:.6f} TEST_SCORED=NO')
    else:
        log('I2_TSOCIAL_STAGE3_ONE_COMPLETE_TRIAL_GATE=PASS NEXT_REVIEW_REAL_EPOCH_TIMINGS_BEFORE_ALL_12')
    log('I2_TSOCIAL_STAGE3_HELDOUT_TEST_EVALUATIONS=0')

if __name__=='__main__':
    try:main()
    except BaseException as e:
        log('I2_TSOCIAL_STAGE3_STOP='+repr(e));traceback.print_exc();sys.exit(2)
