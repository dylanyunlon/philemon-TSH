# CI Run Summary — 20260608_085119

## Results

| Experiment | Scale | PASS | FAIL | Time(ms) | Status |
|-----------|-------|------|------|----------|--------|
| experiment | scale | pass | fail | time_ms | status |
| m145_m146_algo_reader_utils_experiment | 1000 | 47 | 0 |  |  |
| 0 | 47 | OK |  |  |  |
| m145_m146_algo_reader_utils_experiment | 10000 | 47 | 0 |  |  |
| 0 | 331 | OK |  |  |  |
| m145_m146_algo_reader_utils_experiment | 100000 | 47 | 0 |  |  |
| 0 | 3725 | OK |  |  |  |
| m145_m146_algo_reader_utils_experiment | 1000000 | 47 | 0 |  |  |
| 0 | 43491 | OK |  |  |  |

## Algorithm Performance (from logs)

### m145_m146_algo_reader_utils_experiment_scale1000000
```
  ┌─────────────┬───────────┬──────────────────────┐
  │ Algorithm   │ Time (ms) │ Key metric           │
  ├─────────────┼───────────┼──────────────────────┤
  │ BFS         │     24.06 │ 532259 reachable          │
  │ PageRank    │    585.41 │ 20 iters, L1=2.97e-04    │
  │ SSSP        │    193.47 │ 1184804 relaxations       │
  │ WCC         │     91.16 │ 370989 components        │
  └─────────────┴───────────┴──────────────────────┘

═══════════════════════════════════════════════════════
 Results: 47 PASS, 0 FAIL
```

### m145_m146_algo_reader_utils_experiment_scale100000
```
  ┌─────────────┬───────────┬──────────────────────┐
  │ Algorithm   │ Time (ms) │ Key metric           │
  ├─────────────┼───────────┼──────────────────────┤
  │ BFS         │      2.32 │ 63074 reachable          │
  │ PageRank    │     43.05 │ 20 iters, L1=3.20e-04    │
  │ SSSP        │     11.37 │ 140812 relaxations       │
  │ WCC         │      5.68 │ 27236 components        │
  └─────────────┴───────────┴──────────────────────┘

═══════════════════════════════════════════════════════
 Results: 47 PASS, 0 FAIL
```

### m145_m146_algo_reader_utils_experiment_scale10000
```
  ┌─────────────┬───────────┬──────────────────────┐
  │ Algorithm   │ Time (ms) │ Key metric           │
  ├─────────────┼───────────┼──────────────────────┤
  │ BFS         │      0.28 │ 7408 reachable          │
  │ PageRank    │      2.86 │ 20 iters, L1=3.32e-04    │
  │ SSSP        │      0.91 │ 15598 relaxations       │
  │ WCC         │      0.53 │ 1715 components        │
  └─────────────┴───────────┴──────────────────────┘

═══════════════════════════════════════════════════════
 Results: 47 PASS, 0 FAIL
```

### m145_m146_algo_reader_utils_experiment_scale1000
```
  ┌─────────────┬───────────┬──────────────────────┐
  │ Algorithm   │ Time (ms) │ Key metric           │
  ├─────────────┼───────────┼──────────────────────┤
  │ BFS         │      0.07 │ 786 reachable          │
  │ PageRank    │      0.34 │ 20 iters, L1=3.89e-04    │
  │ SSSP        │      0.13 │ 1424 relaxations       │
  │ WCC         │      0.09 │ 126 components        │
  └─────────────┴───────────┴──────────────────────┘

═══════════════════════════════════════════════════════
 Results: 47 PASS, 0 FAIL
```

## For Next Claude

读取以下文件获取详细日志:
```
-rw-rw-r-- 1 jiacheng jiacheng    0 Jun  8 08:51 experiment/logs/20260608_085119/compile_m145_m146_algo_reader_utils_experiment.log
-rw-rw-r-- 1 jiacheng jiacheng 9463 Jun  8 08:52 experiment/logs/20260608_085119/m145_m146_algo_reader_utils_experiment_scale1000000_debug.log
-rw-rw-r-- 1 jiacheng jiacheng 4100 Jun  8 08:52 experiment/logs/20260608_085119/m145_m146_algo_reader_utils_experiment_scale1000000.log
-rw-rw-r-- 1 jiacheng jiacheng 9190 Jun  8 08:51 experiment/logs/20260608_085119/m145_m146_algo_reader_utils_experiment_scale100000_debug.log
-rw-rw-r-- 1 jiacheng jiacheng 4073 Jun  8 08:51 experiment/logs/20260608_085119/m145_m146_algo_reader_utils_experiment_scale100000.log
-rw-rw-r-- 1 jiacheng jiacheng 8874 Jun  8 08:51 experiment/logs/20260608_085119/m145_m146_algo_reader_utils_experiment_scale10000_debug.log
-rw-rw-r-- 1 jiacheng jiacheng 4048 Jun  8 08:51 experiment/logs/20260608_085119/m145_m146_algo_reader_utils_experiment_scale10000.log
-rw-rw-r-- 1 jiacheng jiacheng 8345 Jun  8 08:51 experiment/logs/20260608_085119/m145_m146_algo_reader_utils_experiment_scale1000_debug.log
-rw-rw-r-- 1 jiacheng jiacheng 4025 Jun  8 08:51 experiment/logs/20260608_085119/m145_m146_algo_reader_utils_experiment_scale1000.log
```

下一步任务: M147-M148 (main.cpp + wrapper.h + driver.h)
