from pathlib import Path
import re

ROOT=Path("/workspace/gaap_vast")
SRC=ROOT/"scripts/02_gaap_yelp_unified_tr40_smoke.py"
DST=ROOT/"scripts/02_gaap_amazon_unified_tr40_smoke.py"

s=SRC.read_text()

def require(text,label):
    if text not in s:
        raise SystemExit(f"MISSING_ANCHOR:{label}:{text}")

require('ROOT / "shared/yelp_seed2_nested_splits.npz"',"split_path")
require('assert cfg["loader_type"] == "MiniF"',"loader_type")
require('g.ndata["train_mask"] = mask(tr)',"train_mask")
require("TEST_EVALUATED=NO","test_gate")
require("TEST_LABEL_ISOLATION=PASS","isolation_gate")

s=s.replace(
    'ROOT / "shared/yelp_seed2_nested_splits.npz"',
    'ROOT / "shared/amazon/amazon_seed2_nested_splits.npz"'
)

s=s.replace(
    "/workspace/dataset_audit/raw/yelpchi/YelpChi.mat",
    "/workspace/gaap_vast/shared/amazon/Amazon.mat"
)

s=s.replace(
    "fedb35a8fa539b27866244d3515a47a76b20080cdacb33112da3458fd2487b42",
    "4b7e3f9cccc62b736792707393ccd74332a1a0592dba128ac6b2989bf1ee9d63"
)

s=s.replace(
    "0ea0af36dfc5a3a1f381ea2e3168377e45826498b6063190185c673e9ec8c22b",
    "0fd96816c440b247b8cac768896bc454b1ecf4abebaf5001c1a3e1d900efa4fd"
)

for old,new in [
    ("18381","3455"),
    ("13785","2591"),
    ("9190","1727"),
    ("4595","863"),
    ("9191","1728"),
    ("18382","3456"),
    ("45954","11944"),
]:
    s=re.sub(rf'(?<!\d){old}(?!\d)',new,s)

s=s.replace("(11944, 32)","(11944, 25)")
s=s.replace("(11944,32)","(11944,25)")
s=s.replace("shape[1] == 32","shape[1] == 25")
s=s.replace("shape[1]==32","shape[1]==25")

anchor='assert cfg["loader_type"] == "MiniF"'
inject='\n'.join([
    anchor,
    'cfg["data_name"] = "amazon_comp8851"',
    'cfg["d_in"] = 25',
    'print("GAAP_AMAZON_CONFIG_BASE=yelp.yaml", flush=True)',
    'print("GAAP_AMAZON_DATA_NAME=amazon_comp8851", flush=True)',
    'print("GAAP_AMAZON_D_IN=25", flush=True)',
])
s=s.replace(anchor,inject,1)

mask_anchor='g.ndata["train_mask"] = mask(tr)'
override='\n'.join([
    'import dgl as _dgl_amazon',
    '_adapter_graphs, _ = _dgl_amazon.load_graphs("/workspace/gaap_vast/shared/amazon/amazon_gaap_input_adapter")',
    'assert len(_adapter_graphs) == 1',
    'g = _adapter_graphs[0]',
    'assert int(g.num_nodes()) == 11944',
    'assert int(g.num_edges()) == 8796784',
    'assert tuple(g.ndata["feature"].shape) == (11944, 25)',
    'assert int(g.ndata["label"].numel()) == 11944',
    'print("GAAP_AMAZON_CANONICAL_ADAPTER_LOADED=PASS", flush=True)',
    'print("GAAP_AMAZON_GRAPH_NODES=11944", flush=True)',
    'print("GAAP_AMAZON_GRAPH_EDGES=8796784", flush=True)',
    'print("GAAP_AMAZON_GRAPH_FEATURES=25", flush=True)',
    '',
    'g.ndata["train_mask"] = mask(tr)',
])
s=s.replace(mask_anchor,override,1)

for old,new in [
    ("GAAP_YELP_UNIFIED_2EPOCH_SMOKE","GAAP_AMAZON_UNIFIED_2EPOCH_SMOKE"),
    ("GAAP_YELP","GAAP_AMAZON"),
]:
    s=s.replace(old,new)

must_have=[
    'shared/amazon/amazon_seed2_nested_splits.npz',
    'amazon_gaap_input_adapter',
    'cfg["data_name"] = "amazon_comp8851"',
    'cfg["d_in"] = 25',
    'GAAP_AMAZON_CANONICAL_ADAPTER_LOADED=PASS',
    'TEST_EVALUATED=NO',
    'TEST_LABEL_ISOLATION=PASS',
    '3455','1728','3456','11944','8796784'
]
for x in must_have:
    if x not in s:
        raise SystemExit("MISSING_REQUIRED:"+x)

for x in [
    'shared/yelp_seed2_nested_splits.npz',
    '0ea0af36dfc5a3a1f381ea2e3168377e45826498b6063190185c673e9ec8c22b',
    'fedb35a8fa539b27866244d3515a47a76b20080cdacb33112da3458fd2487b42',
    '/workspace/dataset_audit/raw/yelpchi/YelpChi.mat'
]:
    if x in s:
        raise SystemExit("YELP_CONTROL_RESIDUE:"+x)

DST.write_text(s)
print("AMAZON_RUNNER_PATCH=PASS")
print("DST="+str(DST))
