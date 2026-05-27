# Philemon — Temporal Subgraph Processing with Heterogeneous Memory Hierarchies

> *"Noch hab ich mich ins Freie nicht gekämpft."*
> — Philemon & Baucis, Faust II Act 5

Philemon and Baucis endure through time — temporal graph data persists
across heterogeneous memory tiers: hot data in H100 HBM, warm in A6000 GDDR,
cold in CPU DRAM.

## Upstream

| Directory | Origin | Role |
|-----------|--------|------|
| `upstream/temgraph` | [iykw/TEM-Graph](https://github.com/iykw/TEM-Graph) | Temporal subgraph interval index |
| `upstream/rapidstore` | [SJTU-Liquid/RapidStore](https://github.com/SJTU-Liquid/RapidStore) | Dynamic graph storage for concurrent queries |
