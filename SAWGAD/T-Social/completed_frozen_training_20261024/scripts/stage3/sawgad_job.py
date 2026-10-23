#!/usr/bin/env python3
"""SAWGAD on canonical T-Social TR40 seed 2: sustained adapted training.

16 planned training rounds, 1024 sampled minibatch updates per round, 32 warmup
updates with synthetic-validation checkpoint selection, and full frozen validation after every completed round. NO TEST ACCESS.
This is an exploratory single-config ADAPTED run; not an author-identical
reproduction, 12-trial search or held-out benchmark. No tuning approval presumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import socket
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path('/workspace/comp8851')
REPO = ROOT / 'repos/sawgad_tsocial_author'
SOURCE = ROOT / 'assets/tsocial/tsocial'
SPLIT = ROOT / 'assets/tsocial/tsocial_seed2_nested_splits.npz'
PIN = '64d6d531a2f03549f0bfbe461092beedd6f70bd2'
SOURCE_SHA = '8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0'
SPLIT_SHA = 'b7e76835612e72a8fe5c69bbe87c8ea7b2e5790cbd8849e2a3ac776364bc6519'
OUT = ROOT / 'results/sawgad/tsocial/stage3_tune12_final3x4_adapted_v1' / 'UNINITIALIZED' 
LOG = ROOT / 'logs/SAWGAD_57231045_TSOCIAL_TR40_FULLVAL_ADAPTED_V1.log'
NODES = 5_781_065
EDGES = 146_211_016
EXPECTED = {'TR10':578106,'TR20':1156212,'TR30':1734319,'TR40':2312426,'val':1156213,'test':2312426}
GRID=[{'hidden_dim':h,'drop_rate':d,'full_lr':lr}
      for h in (32,64) for d in (0.0,0.25) for lr in (0.001,0.003,0.005)]
SEED = 2
RATIO = 'TR40'
STAGE = 'tune'
CONFIG = {'hidden_dim':64,'drop_rate':0.25,'full_lr':0.003}
RUN_ID = 'UNINITIALIZED'
HEADS = 5


def need(ok, message):
    if not ok: raise RuntimeError('STOP: '+message)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def emit(*args):
    print(*args,flush=True)


def atomic_json(path, record):
    tmp=path.with_name(path.name+'.writing')
    with open(tmp,'w') as f:
        json.dump(record,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n')
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def protocol(a):
    return dict(
        model='SAWGAD', dataset='T-Social', phase='TR40_12TRIAL_VAL_ONLY_TUNE' if STAGE=='tune' else 'FROZEN_3SEED_4RATIO_TRAIN_ONLY',
        run_id=RUN_ID, stage=STAGE,ratio=RATIO,seed=SEED,
        warmup_steps=a.warmup_steps,warmup_synthetic_val_period=4,
        warmup_validation_negatives=160,warmup_selection='best_mean_synthetic_validation_AUROC',
        supervised_steps_per_epoch=a.supervised_steps_per_epoch, supervised_epochs=a.supervised_epochs,
        validation_nodes=EXPECTED['val'],validation_batch_size=a.validation_batch_size,
        fanout=a.fanout,hidden_dim=CONFIG['hidden_dim'],encoder_layers=1,encoder_heads=4,
        hidden_multiplier=1.25,drop_rate=CONFIG['drop_rate'],weight_decay=0.002,
        warmup_lr=0.0003,full_lr=CONFIG['full_lr'],synthetic_anomalies_per_type=4,
        joint_regularization_weight=4.44,real_pos_per_step=8,real_neg_per_step=248,
        synthetic_normals_per_step=64,source_sha256=SOURCE_SHA,split_sha256=SPLIT_SHA,
        author_commit=PIN,feature_adapter='author_global_transductive_minmax',
        sampling_mode='one_hop_inbound_bounded_neighbour_context_not_fullgraph',
        checkpoints_selected_on='full_frozen_validation_AUPRC',
        complete_validation=True,heldout_test_indexed=False,heldout_test_scored=False,
        final_canonical_benchmark=False,methodological_approval_required=True,
        configuration_frozen_from_trial=getattr(a,'frozen_from_trial',None),
        configuration_frozen_sha256=getattr(a,'frozen_sha256',None),
        training_budget_not_equivalent_to_full_dataset_epoch=True)


def selftest():
    assert sum([EXPECTED['TR40'],EXPECTED['val'],EXPECTED['test']])==NODES
    assert set(EXPECTED)=={'TR10','TR20','TR30','TR40','val','test'}
    assert HEADS==5
    emit('I2_SAWGAD_TUNE12_FINAL3X4_STATIC_SELFTEST=PASS')
    emit('I2_ADAPTED_RESEARCH_ONLY_NOT_AUTHOR_EQUIVALENT=YES')
    emit('I2_HELDOUT_TEST_SCORING_ENABLED=NO')


def read_split():
    import numpy as np
    with np.load(SPLIT,allow_pickle=False) as z:
        need(set(EXPECTED).issubset(z.files),'split archive missing required keys')
        ids={k:np.asarray(z[k],dtype=np.int64).copy() for k in EXPECTED}
        for k in ('seed','split_seed'):
            if k in z.files:need(int(z[k].item())==2,'immutable split seed changed')
    # Sorting + uniqueness safeguards; avoid 6x huge repeated np.isin calls after earlier audit.
    for k,n in EXPECTED.items():
        x=ids[k];need(x.ndim==1 and len(x)==n and x.min()>=0 and x.max()<NODES,f'split {k} shape/bounds')
        need(len(np.unique(x))==n,'split duplicate '+k)
    need(np.isin(ids['TR10'],ids['TR20']).all(),'TR10 nesting')
    need(np.isin(ids['TR20'],ids['TR30']).all(),'TR20 nesting')
    need(np.isin(ids['TR30'],ids['TR40']).all(),'TR30 nesting')
    for a,b in [('TR40','val'),('TR40','test'),('val','test')]:
        need(len(np.intersect1d(ids[a],ids[b]))==0,'split overlap '+a+'/'+b)
    emit('I2_FROZEN_SPLIT_AUDIT=PASS')
    return ids


def run(a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    import dgl
    from dgl.data.utils import load_graphs
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sawgad.model import SAWGADModel
    from sawgad.anomaly import NUM_ANOMALY_TYPES, inject_synthetic_anomalies

    need(socket.gethostname()=='a92ee6fa0813','WRONG HOST I2')
    need(subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==PIN,'upstream code not pinned')
    need(not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip(),'upstream code modified')
    need(sha(SOURCE)==SOURCE_SHA and sha(SPLIT)==SPLIT_SHA,'frozen assets changed')
    need(torch.__version__.split('+')[0]=='2.4.0' and dgl.__version__.split('+')[0]=='2.4.0','wrong isolated author deps')
    need(torch.cuda.is_available(),'CUDA not available')
    need(torch.cuda.mem_get_info()[0] > 30*1024**3,'insufficient free VRAM')
    need(NUM_ANOMALY_TYPES==5,'author anomaly family changed')
    for x in (a.warmup_steps,a.supervised_steps_per_epoch,a.supervised_epochs,a.fanout):need(x>0,'invalid positive arg')
    need(128 <= a.validation_batch_size <= 4096,'invalid validation batch size')
    need(a.supervised_steps_per_epoch<=2048 and a.supervised_epochs<=32 and a.warmup_steps<=128,'training budget exceeds bounded protocol')
    conf=protocol(a)
    OUT.mkdir(parents=True,exist_ok=True)
    need(not (OUT/'pilot_completed.json').exists(),'run completed already; do not rerun')
    manifest=OUT/'pilot_protocol.json'
    if a.resume:
        need(manifest.exists(),'cannot resume without previous run manifest')
        need(json.loads(manifest.read_text())==conf,'resume config changed')
    else:
        need(not manifest.exists() and not (OUT/'resume_state.pt').exists(),'existing run: use --resume, not overwrite')
        atomic_json(manifest,conf)
    emit('I2_AUTHOR_PIN_SOURCE_SPLIT_ENV_AUDIT=PASS')
    ids=read_split()
    torch.manual_seed(SEED); np.random.seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(min(torch.get_num_threads(),8))
    t0=time.perf_counter()
    g_list,_=load_graphs(str(SOURCE))
    need(len(g_list)==1,'expected one canonical graph')
    graph=g_list[0]
    need(graph.device.type=='cpu' and graph.num_nodes()==NODES and graph.num_edges()==EDGES,'canonical graph mismatch')
    need('feature' in graph.ndata and 'label' in graph.ndata,'canonical graph missing data')
    train_ix=torch.from_numpy(ids[RATIO])
    val_ix=torch.from_numpy(ids['val'])
    labels=graph.ndata['label']
    train_y=labels[train_ix].cpu().numpy().astype(np.int8)
    # All frozen validation labels, never test labels. No subset shortcut.
    val_roots=ids['val'].copy().astype(np.int64)
    need(len(val_roots)==EXPECTED['val'],'incomplete validation IDs')
    val_y=labels[torch.from_numpy(val_roots)].cpu().numpy().astype(np.int8)
    need(0<int(train_y.sum())<len(train_y),'train ratio lacks one class')
    if RATIO=='TR40':need(int(train_y.sum())==69712,'TR40 training fraud count mismatch')
    need(np.isin(train_y,[0,1]).all() and np.isin(val_y,[0,1]).all(),'label encoding changed')
    need(int(val_y.sum())>=2 and int(val_y.sum())<len(val_y)-2,'validation diagnostic lacks class support')
    pos_pool=ids[RATIO][train_y==1]
    neg_pool=ids[RATIO][train_y==0]
    emit('I2_TRAIN_POOL_AUDIT=PASS', 'ratio=',RATIO,'seed=',SEED,'nodes=',len(pos_pool)+len(neg_pool),'fraud=',len(pos_pool))
    val_neg_pool=val_roots[val_y==0]
    need(len(val_neg_pool)>=160,'insufficient clean validation negatives')
    warmup_val_roots=np.random.default_rng(20261023).choice(val_neg_pool,size=160,replace=False).astype(np.int64)
    ratio_weight=float(len(neg_pool)/len(pos_pool))
    del labels,train_y,train_ix,val_ix
    del graph.ndata['label']
    need('label' not in graph.ndata,'graph still carries labels')
    feature=graph.ndata['feature'].float()
    need(tuple(feature.shape)==(NODES,10),'feature shape differs')
    fmin=feature.amin(dim=0);fmax=feature.amax(dim=0)
    need(bool(torch.isfinite(fmin).all() and torch.isfinite(fmax).all()),'nonfinite source features')
    feature=(feature-fmin)/(fmax-fmin).clamp_min(1e-12)
    need(bool(torch.isfinite(feature).all()),'nonfinite minmax normalized features')
    # graph attributes remain label-free and shared canonical topology is not modified.
    graph.ndata['feature']=feature
    emit('I2_AUTHOR_MINMAX_FEATURE_ADAPTER=PASS', 'range=',round(float(feature.min()),5),round(float(feature.max()),5))
    emit('I2_TRAIN_LABELS_ONLY_VALIDATION_SELECTED_TEST_LABELS_INDEXED=NO',f'full_validation={len(val_y)}',f'val_fraud={int(val_y.sum())}')
    del ids
    emit('I2_CANONICAL_GRAPH_PREP_SECONDS=',round(time.perf_counter()-t0,3))

    model=SAWGADModel(in_features=10,hidden_dim=CONFIG['hidden_dim'],num_synthetic_heads=5,drop_rate=CONFIG['drop_rate'],
         encoder_num_layers=1,encoder_num_heads=4,encoder_hidden_multiplier=1.25).to('cuda')
    warm_opt=torch.optim.Adam(model.parameters(),lr=.0003,weight_decay=.002)
    main_opt=torch.optim.Adam(model.parameters(),lr=CONFIG['full_lr'],weight_decay=.002)
    phase='warmup'; done_warm=0;done_epoch=0;done_step=0;round_losses=[];best_ap=-1.;history=[];best_warmup_auc=-1.

    if a.resume and (OUT/'resume_state.pt').exists():
        resume=torch.load(OUT/'resume_state.pt',map_location='cpu',weights_only=False)
        need(resume['protocol']==conf,'resume protocol mismatch')
        model.load_state_dict(resume['model'])
        warm_opt.load_state_dict(resume['warm_opt'])
        main_opt.load_state_dict(resume['main_opt'])
        for optimizer in (warm_opt,main_opt):
            for values in optimizer.state.values():
                for k,v in values.items():
                    if isinstance(v,torch.Tensor): values[k]=v.to('cuda')
        phase=resume['phase'];done_warm=int(resume['done_warm']);done_epoch=int(resume['done_epoch'])
        best_ap=float(resume['best_ap']);history=resume['history'];best_warmup_auc=float(resume.get('best_warmup_auc',-1.))
        done_step=int(resume.get('done_step',0));round_losses=list(resume.get('round_losses',[]))
        need(len(round_losses)==done_step,'midround loss count mismatch')
        torch.set_rng_state(resume['torch_rng'])
        torch.cuda.set_rng_state_all(resume['cuda_rng'])
        emit('I2_RESUMED_AT_PHASE=',phase,'done_epochs=',done_epoch)

    def checkpoint(current_phase):
        record=dict(protocol=conf,model=model.state_dict(),warm_opt=warm_opt.state_dict(),
            main_opt=main_opt.state_dict(),phase=current_phase,done_warm=done_warm,
            done_epoch=done_epoch,done_step=done_step,round_losses=round_losses,best_ap=best_ap,
            best_warmup_auc=best_warmup_auc,history=history,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            test_labels_indexed=False,test_metrics_computed=False)
        dest=OUT/'resume_state.pt';tmp=OUT/'resume_state.pt.writing'
        torch.save(record,tmp);os.replace(tmp,dest)
        atomic_json(OUT/'pilot_history.json',dict(history=history,done_warm=done_warm,
            done_epoch=done_epoch,done_step=done_step,best_full_validation_AUPRC=best_ap,
            best_warmup_synthetic_val_AUROC=best_warmup_auc,phase=current_phase,
            full_validation=True,test_labels_indexed=False,test_scored=False))

    def compact_graph(root_ids, sample_seed):
        # Real-data one-hop inbound bounded-neighbour *induced message context*.
        # One GAT layer, matching the original CLI architecture depth.
        dgl.seed(int(sample_seed))
        roots=torch.as_tensor(root_ids,dtype=torch.int64,device='cpu')
        sampled=dgl.sampling.sample_neighbors(graph,roots,fanout=a.fanout,edge_dir='in',replace=False)
        src,dst=sampled.edges()
        src=src.cpu().long();dst=dst.cpu().long()
        need(len(src)<=a.fanout*len(roots),'sampled edge cap exceeded')
        nodes=torch.unique(torch.cat((roots,src,dst)))
        si=torch.searchsorted(nodes,src); di=torch.searchsorted(nodes,dst)
        need(torch.equal(nodes[si],src) and torch.equal(nodes[di],dst),'bad subgraph relabel')
        local=torch.searchsorted(nodes,roots)
        sub=dgl.graph((si,di),num_nodes=len(nodes))
        sub=dgl.to_simple(dgl.remove_self_loop(sub))
        sub=dgl.add_self_loop(sub)
        sub.ndata['feature']=graph.ndata['feature'][nodes].contiguous()
        return sub.to('cuda'),local.to('cuda')

    def anomalous_loss(synth_graph, local_roots, sample_seed):
        torch.manual_seed(int(sample_seed))
        torch.cuda.manual_seed(int(sample_seed))
        synth, labels_per_head=inject_synthetic_anomalies(synth_graph,local_roots,anomalies_per_type=4)
        locs=[];ys=[]
        for lab in labels_per_head:
            lab=lab.reshape(-1).cpu()
            positives=torch.nonzero(lab==1).reshape(-1)
            negatives=torch.nonzero(lab==0).reshape(-1)
            need(len(positives)==4 and len(negatives)>=4,'bad synthetic split')
            # Deterministic, balanced synthetic supervision from training normals only.
            choice=torch.cat((positives,negatives[:4]))
            locs.append(local_roots[choice.to('cuda')]);ys.append(lab[choice].float().to('cuda').unsqueeze(1))
        logits=model.score_synthetic(synth,locs)
        need(len(logits)==5,'synthetic heads mismatch')
        return torch.stack([F.binary_cross_entropy_with_logits(p,y) for p,y in zip(logits,ys)]).mean()

    def train_step(stage,step_idx):
        seed=SEED*100_000+step_idx+(0 if stage=='warmup' else 5_000_000)
        rng=np.random.default_rng(seed)
        syn_roots=rng.choice(neg_pool,size=(128 if stage=='warmup' else 64),replace=False)
        if stage=='warmup':
            roots=syn_roots.astype(np.int64)
            sub,loc=compact_graph(roots,seed)
            loss=anomalous_loss(sub,loc,seed+1)
            opt=warm_opt
        else:
            real_pos=rng.choice(pos_pool,size=8,replace=False)
            real_neg=rng.choice(neg_pool,size=248,replace=False)
            # normal synthetic candidates distinct from labelled real batch
            while np.intersect1d(syn_roots,real_neg).size:
                syn_roots=rng.choice(neg_pool,size=64,replace=False)
            roots=np.concatenate((real_pos,real_neg,syn_roots)).astype(np.int64)
            sub,loc=compact_graph(roots,seed)
            y=torch.cat((torch.ones(8),torch.zeros(248))).to('cuda').unsqueeze(1)
            pw=torch.tensor(ratio_weight,device='cuda')
            real=model.score_real(sub,loc[:256])
            real_loss=F.binary_cross_entropy_with_logits(real,y,pos_weight=pw)
            synthetic=anomalous_loss(sub,loc[256:],seed+1)
            loss=real_loss+4.44*synthetic
            opt=main_opt
        opt.zero_grad(set_to_none=True)
        need(bool(torch.isfinite(loss)),'nonfinite '+stage+' loss')
        loss.backward()
        opt.step()
        value=float(loss.detach().cpu())
        del sub,loc
        return value

    def validate(epoch):
        model.eval()
        scores=[]
        with torch.inference_mode():
            for start in range(0,len(val_roots),a.validation_batch_size):
                batch=val_roots[start:start+a.validation_batch_size]
                sub,loc=compact_graph(batch,10_000_000+start)
                prob=model.score_real(sub,loc).view(-1).detach().cpu().numpy()
                need(np.isfinite(prob).all(),'nonfinite validation scores')
                scores.append(prob)
                del sub,loc
                if start>0 and (start//a.validation_batch_size)%100==0:
                    emit('I2_FULL_VALIDATION_PROGRESS=',min(start+a.validation_batch_size,len(val_roots)),'/',len(val_roots))
        pred=np.concatenate(scores)
        need(len(pred)==len(val_y),'validation score count')
        return float(average_precision_score(val_y,pred))

    def synthetic_validation_auc():
        # Replicate the author's warm-up selection principle (mean synthetic AUROC),
        # but on a bounded, fixed validation-negative sample (not author-identical).
        model.eval()
        with torch.no_grad():
            local_graph, local_roots=compact_graph(warmup_val_roots,90000011)
            torch.manual_seed(90000012)
            torch.cuda.manual_seed(90000012)
            synthetic_val, synthetic_labels=inject_synthetic_anomalies(
                local_graph,local_roots,anomalies_per_type=4)
            logits=model.score_synthetic(synthetic_val,[local_roots]*HEADS)
            aucs=[]
            for y,scores in zip(synthetic_labels,logits):
                y_np=y.numpy().astype('int8')
                score_np=scores.detach().cpu().numpy().reshape(-1)
                need(len(y_np)==160 and y_np.sum()==4,'bad synthetic warm-up val labels')
                aucs.append(float(roc_auc_score(y_np,score_np)))
        auc=float(statistics.mean(aucs))
        need(math.isfinite(auc) and 0<=auc<=1,'nonfinite synthetic val AUROC')
        return auc

    if phase=='warmup':
        model.train(); losses=[];start=time.perf_counter()
        emit('I2_SAWGAD_REAL_TRAINING_STARTED=YES','run_id=',RUN_ID,'stage=synthetic_warmup')
        for i in range(done_warm,a.warmup_steps):
            l=train_step('warmup',i);losses.append(l)
            done_warm=i+1
            if done_warm % 4 == 0 or done_warm == a.warmup_steps:
                auc=synthetic_validation_auc()
                if auc>best_warmup_auc:
                    best_warmup_auc=auc
                    tmp=OUT/'best_warmup_synthetic_val.pt.writing'
                    torch.save(dict(model=model.state_dict(),warmup_steps=done_warm,
                        synthetic_val_AUROC=auc,validation_only=True,
                        author_equivalent=False,protocol=conf),tmp)
                    os.replace(tmp,OUT/'best_warmup_synthetic_val.pt')
                emit('I2_WARMUP_SYNTHETIC_VALIDATION_AUROC=',round(auc,6),
                    'best=',round(best_warmup_auc,6),'step=',done_warm)
                model.train()
            checkpoint('warmup')
            emit(f'I2_SAWGAD_WARMUP_STEP={i+1}/{a.warmup_steps} loss={l:.6f} elapsed_s={time.perf_counter()-start:.1f}')
        need(best_warmup_auc>=0 and (OUT/'best_warmup_synthetic_val.pt').exists(),
             'no warmup synthetic-validation winner')
        best=torch.load(OUT/'best_warmup_synthetic_val.pt',map_location='cpu',weights_only=False)
        need(best['protocol']==conf and best['synthetic_val_AUROC']==best_warmup_auc,
             'warmup best protocol invalid')
        model.load_state_dict(best['model'])
        model.to('cuda')
        phase='supervised';checkpoint(phase)
        emit('I2_SAWGAD_WARMUP_BEST_STATE_RESTORED=PASS',
             'best_synthetic_val_AUROC=',round(best_warmup_auc,6),
             'winning_step=',best['warmup_steps'])
        emit('I2_SAWGAD_WARMUP_CHECKPOINT=PASS completed_steps=',done_warm)

    for epoch in range(done_epoch,a.supervised_epochs):
        model.train(); start=time.perf_counter()
        for step in range(done_step,a.supervised_steps_per_epoch):
            n=epoch*a.supervised_steps_per_epoch+step
            l=train_step('supervised',n);round_losses.append(l);done_step=step+1
            if done_step%64==0 or done_step==a.supervised_steps_per_epoch:
                checkpoint('supervised')
                emit(f'I2_SAWGAD_{RUN_ID}_TRAIN round={epoch+1}/{a.supervised_epochs} batch={done_step}/{a.supervised_steps_per_epoch} loss={l:.6f} elapsed_s={time.perf_counter()-start:.1f}')
        validation_start=time.perf_counter()
        ap=validate(epoch)
        validation_seconds=time.perf_counter()-validation_start
        need(math.isfinite(ap) and 0<=ap<=1,'bad full-validation AUPRC')
        duration=time.perf_counter()-start
        row=dict(epoch=epoch+1,train_batch_count=a.supervised_steps_per_epoch,
            mean_training_loss=float(statistics.mean(round_losses)),full_validation_AUPRC=ap,
            full_validation_count=len(val_y),full_validation_fraud=int(val_y.sum()),
            validation_seconds=round(validation_seconds,3),
            elapsed_seconds=round(duration,3),gpu_peak_allocated_gib=round(torch.cuda.max_memory_allocated()/1024**3,4))
        history.append(row)
        if ap>best_ap:
            best_ap=ap
            tmp=OUT/'best_full_validation.pt.writing'
            torch.save(dict(model=model.state_dict(),epoch=epoch+1,
                selection='TR40_FULL_FROZEN_VALIDATION_12TRIAL_TUNING' if STAGE=='tune' else 'FROZEN_CONFIG_FULL_VALIDATION_FINAL_TRAINING',
                full_validation_AUPRC=ap,protocol=conf),tmp)
            os.replace(tmp,OUT/'best_full_validation.pt')
        done_epoch=epoch+1;done_step=0;round_losses=[]
        checkpoint('supervised' if done_epoch<a.supervised_epochs else 'done')
        emit('I2_SAWGAD_EPOCH_COMPLETE='+json.dumps(row,sort_keys=True))

    summary=dict(status='ADAPTED_TUNING_VALIDATION_ONLY_COMPLETED' if STAGE=='tune' else 'ADAPTED_FINAL_TRAINING_COMPLETE_PRETEST_AUDIT_PENDING',protocol=conf,
        total_warmup_optimizer_steps=done_warm,
        best_warmup_synthetic_val_AUROC=best_warmup_auc,
        total_supervised_optimizer_steps=done_epoch*a.supervised_steps_per_epoch,
        full_validation_AUPRC_best=best_ap,
        full_validation_fraud=int(val_y.sum()),
        full_validation_examples=len(val_y),
        max_cuda_allocated_gib=round(torch.cuda.max_memory_allocated()/1024**3,4),
        max_process_rss_gib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,4),
        all_validation_nodes_evaluated=True,
        heldout_test_labels_indexed=False,
        heldout_test_metrics_computed=False,
        fullgraph_author_equivalent=False,
        final_benchmark_result=False,
        methodological_approval_recorded=False)
    atomic_json(OUT/'pilot_completed.json',summary)
    emit('I2_SAWGAD_WORKER_COMPLETED=PASS','run_id=',RUN_ID)
    emit('I2_SAWGAD_NO_TEST_LABEL_ACCESS=PASS')
    emit('I2_SAWGAD_HOLDOUT_TEST_AUTHORIZED=NO')
    emit('I2_SAWGAD_HELDOUT_TEST_AUTHORIZED=NO')
    emit('I2_SAWGAD_RUN_SUMMARY='+json.dumps(summary,sort_keys=True))


def main():
    import argparse
    from types import SimpleNamespace
    global SEED, RATIO, STAGE, CONFIG, OUT, RUN_ID
    p=argparse.ArgumentParser(description='SAWGAD T-Social adaptive benchmark job runner, NO TEST')
    p.add_argument('--spec', type=Path)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--selftest',action='store_true')
    p.add_argument('--check-spec',action='store_true')
    a=p.parse_args()
    if a.selftest:
        selftest();return
    need(a.spec is not None and a.spec.is_file(),'missing run spec')
    s=json.loads(a.spec.read_text())
    need(s.get('schema')=='sawgad-tsocial-job-v1','unknown run schema')
    need(s.get('source_sha256')==SOURCE_SHA and s.get('split_sha256')==SPLIT_SHA,'source/split mismatch')
    STAGE=s['stage'];RATIO=s['ratio'];SEED=int(s['seed']);RUN_ID=s['run_id']
    need(STAGE in ('tune','final'),'invalid stage')
    need(RATIO in EXPECTED and RATIO not in ('val','test'),'invalid train ratio')
    need(SEED in (2,42,72),'invalid seed')
    need(isinstance(s.get('trial'),int) and 1<=s['trial']<=12 if STAGE=='tune' else s.get('trial') is None,'trial index invalid')
    need(RUN_ID==('tune_%02d'%s['trial'] if STAGE=='tune' else 'final_%s_seed%d'%(RATIO,SEED)),'run id mismatch')
    need((STAGE=='tune' and RATIO=='TR40' and SEED==2) or STAGE=='final','trial controls')
    CONFIG=dict(s['config'])
    need(set(CONFIG)=={'hidden_dim','drop_rate','full_lr'},'unexpected hyperparameter key')
    need(CONFIG['hidden_dim'] in (32,64) and CONFIG['drop_rate'] in (0.0,0.25) and CONFIG['full_lr'] in (0.001,0.003,0.005),'unapproved grid config')
    if STAGE=='tune':
        need(s['config']==GRID[int(s['trial'])-1], 'tuning config not at its frozen grid position')
    if STAGE=='final':
        frozen=Path(s['frozen_path'])
        need(frozen.is_file(),'winner freeze file missing')
        lock=json.loads(frozen.read_text())
        need(lock.get('schema')=='sawgad-tsocial-winner-v1','freeze schema mismatch')
        need(lock['winner_config']==CONFIG,'final config differs from frozen winner')
        need(lock['winner_trial']==s['frozen_from_trial'],'winner trial mismatch')
        need(sha(frozen)==s['frozen_sha256'],'winner freeze hash mismatch')
    else:
        need(s.get('frozen_sha256') is None,'tune must not use lock')
    OUT=ROOT/'results/sawgad/tsocial/stage3_tune12_final3x4_adapted_v1'/RUN_ID
    need(str(OUT.resolve()).startswith(str(ROOT.resolve())),'output escapes root')
    kwargs={k:s[k] for k in ('warmup_steps','supervised_steps_per_epoch','supervised_epochs','validation_batch_size','fanout')}
    kwargs.update(resume=a.resume,frozen_from_trial=s.get('frozen_from_trial'),frozen_sha256=s.get('frozen_sha256'))
    need(16<=kwargs['warmup_steps']<=64 and 512<=kwargs['supervised_steps_per_epoch']<=1024,'unexpected warmup or update budget')
    need(8<=kwargs['supervised_epochs']<=16 and kwargs['validation_batch_size']==2048 and kwargs['fanout']==10,'unexpected validation/sampler budget')
    expected_epochs=8 if STAGE=='tune' else 16
    need(kwargs['supervised_epochs']==expected_epochs and kwargs['supervised_steps_per_epoch']==1024,'stage budget changed')
    need(kwargs['warmup_steps']==32,'warmup budget changed')
    if a.check_spec:
        emit('I2_JOB_SPEC_STATIC_CHECK=PASS',RUN_ID,STAGE,RATIO,SEED);return
    run(SimpleNamespace(**kwargs))


if __name__=='__main__':
    main()
