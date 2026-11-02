# 18-pair compatibility matrix

`PASS_WITH_SHARED_ADAPTER`: the official code supports the dataset; only the frozen canonical data and splits are fed through a documented adapter. `ADAPTED (documented)`: the official code has no entry for the dataset; the decision is recorded in audit/DEVIATIONS.md section B. No pair is UNSUPPORTED.

Execution evidence column: local CPU smoke (2 epochs). Smoke runs are feasibility tests, not results.

| Model | Dataset | Official support | Status | Adapter | Smoke | Smoke detail |
|---|---|---|---|---|---|---|
| SAGAD | yelpchi | native (GADBench 'yelp') | PASS_WITH_SHARED_ADAPTER | canonical union view -> load_data graph ops; official semi_train.conf.yaml values (dataset block); exact vectorised MRQ | COMPLETE | reload check passed; 2 epochs; 23s on cpu (full graph) |
| SAGAD | amazon | native | PASS_WITH_SHARED_ADAPTER | canonical union view -> load_data graph ops; official semi_train.conf.yaml values (dataset block); exact vectorised MRQ | COMPLETE | reload check passed; 2 epochs; 26s on cpu (full graph) |
| SAGAD | tfinance | native | PASS_WITH_SHARED_ADAPTER | canonical union view -> load_data graph ops; official semi_train.conf.yaml values (dataset block); exact vectorised MRQ | COMPLETE | reload check passed; 2 epochs; 128s on cpu (full graph) |
| SAGAD | elliptic | native | PASS_WITH_SHARED_ADAPTER | canonical union view -> load_data graph ops; official semi_train.conf.yaml values (dataset block); exact vectorised MRQ | COMPLETE | reload check passed; 2 epochs; 11s on cpu (full graph) |
| SAGAD | fdcompcn | not in official config | ADAPTED (documented) | canonical union view -> load_data graph ops; official semi_train.conf.yaml values (default block); exact vectorised MRQ | COMPLETE | reload check passed; 2 epochs; 6s on cpu (full graph) |
| SAGAD | tsocial | native (official MRQ impractical) | PASS_WITH_SHARED_ADAPTER | canonical union view -> load_data graph ops; official semi_train.conf.yaml values (dataset block); exact vectorised MRQ | COMPLETE | reload check passed; 2 epochs; 54s on cpu (300,000-node induced subgraph (smoke only)) |
| SpaceGNN | yelpchi | native (GADBench 'yelp') | PASS_WITH_SHARED_ADAPTER | canonical union view + self-loops; feature_i propagation; PARAMETERS / Table 4 values | COMPLETE | reload check passed; 2 epochs; 126s on cpu (full graph) |
| SpaceGNN | amazon | native | PASS_WITH_SHARED_ADAPTER | canonical union view + self-loops; feature_i propagation; PARAMETERS / Table 4 values | COMPLETE | reload check passed; 2 epochs; 969s on cpu (full graph) |
| SpaceGNN | tfinance | native | PASS_WITH_SHARED_ADAPTER | canonical union view + self-loops; feature_i propagation; PARAMETERS / Table 4 values | COMPLETE | reload check passed; 2 epochs; 265s on cpu (8,000-node induced subgraph (smoke only)) |
| SpaceGNN | elliptic | unsupported (crash: stdneg undefined) | ADAPTED (documented) | canonical union view + self-loops; feature_i propagation; argparse defaults + std 0.015 (B3) | COMPLETE | reload check passed; 2 epochs; 313s on cpu (full graph) |
| SpaceGNN | fdcompcn | unsupported (crash: stdneg undefined) | ADAPTED (documented) | canonical union view + self-loops; feature_i propagation; argparse defaults + std 0.015 (B3) | COMPLETE | reload check passed; 2 epochs; 34s on cpu (full graph) |
| SpaceGNN | tsocial | native | PASS_WITH_SHARED_ADAPTER | canonical union view + self-loops; feature_i propagation; PARAMETERS / Table 4 values | COMPLETE | reload check passed; 2 epochs; 3150s on cpu (300,000-node induced subgraph (smoke only)) |
| CRoC | yelpchi | native | PASS_WITH_SHARED_ADAPTER | relational branch: one-hot relation types, min-max features, base model | COMPLETE | reload check passed; 2 epochs; 79s on cpu (full graph) |
| CRoC | amazon | native | PASS_WITH_SHARED_ADAPTER | relational branch: one-hot relation types, min-max features, base model | COMPLETE | reload check passed; 2 epochs; 281s on cpu (full graph) |
| CRoC | tfinance | native ('tfin') | PASS_WITH_SHARED_ADAPTER | single-relation branch: constant type, z-score features, _Large model | COMPLETE | reload check passed; 2 epochs; 2670s on cpu (full graph) |
| CRoC | elliptic | unsupported (NotImplementedError) | ADAPTED (documented) | single-relation branch: constant type, z-score features, _Large model (B5/B6) | COMPLETE | reload check passed; 2 epochs; 24s on cpu (full graph) |
| CRoC | fdcompcn | unsupported (NotImplementedError) | ADAPTED (documented) | relational branch: one-hot relation types, min-max features, base model (3 semantic relations; stored union 'homo' excluded, B8) | COMPLETE | reload check passed; 2 epochs; 7s on cpu (full graph) |
| CRoC | tsocial | native ('tsoc') | PASS_WITH_SHARED_ADAPTER | single-relation branch: constant type, z-score features, _Large model | COMPLETE | reload check passed; 2 epochs; 67s on cpu (300,000-node induced subgraph (smoke only)) |
