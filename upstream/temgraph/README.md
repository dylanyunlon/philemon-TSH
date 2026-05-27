# TEM-Graph
[VLDB 2026] Efficient Temporal Subgraph Management: A New Interval Index



Data file: ./data/toy.dat

Query file: ./data/query_toy.dat

Compile 

```sh
cd build
cmake ..
make
```



Run

```sh
./temgraph -q contains ../data/toy.dat ../data/query_toy.dat
./temgraph -q contained ../data/toy.dat ../data/query_toy.dat

```

