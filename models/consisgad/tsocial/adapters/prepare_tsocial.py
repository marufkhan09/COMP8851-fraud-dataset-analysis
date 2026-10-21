import hashlib, json, time
from pathlib import Path
import numpy as np, dgl
from dgl.data.utils import load_graphs
from sklearn.model_selection import train_test_split
RAW=Path('/workspace/tsocial_raw/raw/tsocial'); OUT=Path('/workspace/tsocial_work')
EXPECTED='8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0'
h=hashlib.sha256()
with open(RAW,'rb') as f:
    for b in iter(lambda:f.read(1<<24),b''): h.update(b)
assert h.hexdigest()==EXPECTED, h.hexdigest()
print('raw sha OK',flush=True)
g=load_graphs(str(RAW))[0][0]
n=g.num_nodes(); y=g.ndata['label'].numpy().astype(np.int64)
src,dst=g.edges()
print('nodes',n,'edges',g.num_edges(),'fraud',int((y==1).sum()),'labels',np.unique(y),flush=True)
assert n==5781065 and g.num_edges()==146211016 and int((y==1).sum())==174280
np.save(OUT/'data/features.npy',g.ndata['feature'].float().numpy())
np.save(OUT/'data/labels.npy',y)
np.save(OUT/'data/src.npy',src.numpy().astype(np.int32)); np.save(OUT/'data/dst.npy',dst.numpy().astype(np.int32))
idx=np.arange(n)
tr40,rest=train_test_split(idx,train_size=int(0.4*n),stratify=y,random_state=2)
val,test=train_test_split(rest,train_size=int(0.2*n),stratify=y[rest],random_state=2)
trains={'TR40':np.sort(tr40)}; cur=tr40
for name,frac in (('TR30',0.75),('TR20',2/3),('TR10',0.5)):
    cur,_=train_test_split(cur,train_size=int(len(cur)*frac),stratify=y[cur],random_state=2)
    trains[name]=np.sort(cur)
man={'status':'PROVISIONAL - regenerated from documented convention, not Pathik frozen file','raw_sha256':EXPECTED,'splits':{}}
for name,tr in trains.items():
    un=np.setdiff1d(tr40,tr)
    p=OUT/f'splits/tsocial_{name}_seed2.npz'
    np.savez(p,train=tr,validation=np.sort(val),test=np.sort(test),unused=un)
    man['splits'][name]={'train':len(tr),'train_fraud':int(y[tr].sum()),'validation':len(val),'test':len(test),'unused':len(un),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    print(name,man['splits'][name],flush=True)
(OUT/'splits/manifest.json').write_text(json.dumps(man,indent=1))
