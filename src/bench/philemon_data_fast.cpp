/**
 * philemon_data_fast.cpp — Fast data generation for Philemon-TSH
 * Produces 4 JSON files with 2000-step X-axis like the data_demo.
 * Optimized: reuses bridge across steps, reduced iteration counts.
 *
 * Build: g++ -std=c++17 -O2 -pthread -I src -o philemon_fast src/bench/philemon_data_fast.cpp
 */

#include "../core/tiered_allocator.hpp"
#include "../bridge/temporal_bridge.hpp"
#include "../scheduler/migration_scheduler.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <random>
#include <cmath>
#include <thread>
#include <vector>
#include <atomic>
#include <sys/resource.h>

using namespace philemon;

static double get_peak_rss_mb() {
    struct rusage u; return getrusage(RUSAGE_SELF, &u) == 0 ? u.ru_maxrss / 1024.0 : -1.0;
}

static std::string vj(const std::vector<double>& v) {
    std::ostringstream o; o << "[";
    for (size_t i = 0; i < v.size(); ++i) { if (i) o << ","; o << std::setprecision(6) << v[i]; }
    o << "]"; return o.str();
}

static std::vector<TemporalEdge>
gen_edges(size_t n, int32_t ts_max, uint64_t mv, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<TemporalEdge> e; e.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        double u1 = std::uniform_real_distribution<>(0,1)(rng), u2 = std::uniform_real_distribution<>(0,1)(rng);
        uint64_t s = (uint64_t)(u1*u1*mv), d = (uint64_t)(u2*u2*mv);
        if (s==d) d=(d+1)%mv;
        int32_t t0 = std::uniform_int_distribution<int32_t>(0,ts_max)(rng);
        int32_t t1 = std::min(t0 + std::uniform_int_distribution<int32_t>(1,100)(rng), ts_max);
        e.emplace_back(s,d,1.0,t0,t1);
    }
    return e;
}

// ════════ FIGURE 1: Query Latency vs Steps (incremental ingest) ════════
// X: 2000 steps of incremental edge addition
// Y: query latency for narrow/wide
// Methods: Tiered, HBM-Only, DRAM-Only
// Seeds: 3
static void fig1(const std::string& path) {
    std::cout << "[Fig1] Query Latency vs Incremental Steps\n";
    const int NS=2000, NSEED=3, BATCH=500;
    const uint64_t SD[]={42,137,271};
    struct M { const char* n; size_t h,g,d; } ms[] = {
        {"Tiered",512ULL<<20,1024ULL<<20,2048ULL<<20},
        {"HBM-Only",4096ULL<<20,0,0},
        {"DRAM-Only",0,0,4096ULL<<20}
    };
    int NM=3;
    struct Q { const char* n; int32_t lo,hi; } qs[] = {{"narrow",4500,4550},{"wide",0,7500}};
    int NQ=2;

    std::ofstream out(path);
    out << "{\n  \"metadata\":{\"panel\":\"Query Latency vs Incremental Steps\",\"source\":\"Philemon-TSH\",\"n_per_seed\":" << NS << ",\"n_seeds\":" << NSEED << "},\n";
    out << "  \"steps\":["; for(int i=0;i<NS;++i){if(i)out<<",";out<<i*BATCH;} out<<"],\n";
    out << "  \"methods\":{\n";

    for(int m=0;m<NM;++m){
        if(m) out<<",\n";
        out << "    \"" << ms[m].n << "\":{\n";
        std::vector<std::vector<double>> narrow_lat(NSEED), wide_lat(NSEED);

        for(int s=0;s<NSEED;++s){
            std::cout << "  " << ms[m].n << " s" << s;
            size_t h=ms[m].h,g=ms[m].g,d=ms[m].d;
            if(!h&&!g) d=std::max(d,(size_t)(4096ULL<<20));
            TieredAllocator a(h,g,d);
            TierPlacementPolicy pol(100'000'000ULL,1'000'000'000ULL);
            TemporalBridge br(a,pol,50000);
            auto edges = gen_edges(NS*BATCH, 10000, 100000, SD[s]);

            for(int step=0;step<NS;++step){
                size_t base=step*BATCH, end=base+BATCH;
                for(size_t i=base;i<end&&i<edges.size();++i) br.add_edge(edges[i]);
                if((step+1)%10==0 || step==0) br.flush_partitions();

                for(int q=0;q<NQ;++q){
                    uint64_t cnt=0;
                    auto t0=std::chrono::high_resolution_clock::now();
                    int iters=20;
                    for(int it=0;it<iters;++it){ cnt=0;
                        br.temporal_subgraph_query(qs[q].lo,qs[q].hi,[&cnt](const TemporalEdge&){++cnt;});
                    }
                    auto t1=std::chrono::high_resolution_clock::now();
                    double us=std::chrono::duration<double,std::micro>(t1-t0).count()/iters;
                    if(q==0) narrow_lat[s].push_back(us);
                    else wide_lat[s].push_back(us);
                }
                if(step%200==0){std::cout<<".";std::cout.flush();}
            }
            std::cout<<" ";
        }
        std::cout<<"\n";

        // Write narrow
        out << "      \"narrow\":{\n";
        for(int s=0;s<NSEED;++s) out<<"        \"seed_"<<s<<"\":"<<vj(narrow_lat[s])<<",\n";
        std::vector<double> mn(NS,0),sd(NS,0);
        for(int i=0;i<NS;++i){for(int s=0;s<NSEED;++s)mn[i]+=narrow_lat[s][i];mn[i]/=NSEED;
            for(int s=0;s<NSEED;++s){double d=narrow_lat[s][i]-mn[i];sd[i]+=d*d;}sd[i]=sqrt(sd[i]/NSEED);}
        out<<"        \"mean\":"<<vj(mn)<<",\n        \"std\":"<<vj(sd)<<"\n      },\n";

        // Write wide
        out << "      \"wide\":{\n";
        for(int s=0;s<NSEED;++s) out<<"        \"seed_"<<s<<"\":"<<vj(wide_lat[s])<<",\n";
        for(int i=0;i<NS;++i){mn[i]=0;sd[i]=0;for(int s=0;s<NSEED;++s)mn[i]+=wide_lat[s][i];mn[i]/=NSEED;
            for(int s=0;s<NSEED;++s){double d=wide_lat[s][i]-mn[i];sd[i]+=d*d;}sd[i]=sqrt(sd[i]/NSEED);}
        out<<"        \"mean\":"<<vj(mn)<<",\n        \"std\":"<<vj(sd)<<"\n      }\n";
        out << "    }";
    }
    out << "\n  }\n}\n";
    out.close();
    std::cout << "  → " << path << "\n";
}

// ════════ FIGURE 2: Throughput (QPS) vs Steps ════════
static void fig2(const std::string& path) {
    std::cout << "[Fig2] Throughput vs Steps\n";
    const int NS=2000, NSEED=3, QBATCH=100;
    const uint64_t SD[]={42,137,271};
    struct M { const char* n; size_t h,g,d; } ms[] = {
        {"Tiered",512ULL<<20,1024ULL<<20,2048ULL<<20},
        {"HBM-Only",4096ULL<<20,0,0},
        {"DRAM-Only",0,0,4096ULL<<20}
    };
    int NM=3;

    std::ofstream out(path);
    out << "{\n  \"metadata\":{\"panel\":\"QPS vs Sequential Steps\",\"source\":\"Philemon-TSH\",\"n_per_seed\":"<<NS<<",\"n_seeds\":"<<NSEED<<"},\n";
    out << "  \"steps\":["; for(int i=0;i<NS;++i){if(i)out<<",";out<<i;} out<<"],\n";
    out << "  \"methods\":{\n";

    for(int m=0;m<NM;++m){
        if(m) out<<",\n";
        out << "    \"" << ms[m].n << "\":{\n";
        std::vector<std::vector<double>> qps_d(NSEED);
        std::vector<double> time_h;

        for(int s=0;s<NSEED;++s){
            std::cout << "  " << ms[m].n << " s" << s;
            size_t h=ms[m].h,g=ms[m].g,d=ms[m].d;
            if(!h&&!g) d=std::max(d,(size_t)(4096ULL<<20));
            TieredAllocator a(h,g,d);
            TierPlacementPolicy pol(100'000'000ULL,1'000'000'000ULL);
            TemporalBridge br(a,pol,50000);
            auto edges=gen_edges(500000,10000,100000,SD[s]);
            br.add_edges(edges);br.flush_partitions();

            auto gstart=std::chrono::high_resolution_clock::now();
            for(int step=0;step<NS;++step){
                int32_t c=(step*10000)/NS, half=500+(step%200)*5;
                int32_t lo=std::max(0,c-half), hi=std::min(10000,c+half);
                auto t0=std::chrono::high_resolution_clock::now();
                for(int q=0;q<QBATCH;++q){
                    uint64_t cnt=0;
                    br.temporal_subgraph_query(lo,hi,[&cnt](const TemporalEdge&){++cnt;});
                }
                auto t1=std::chrono::high_resolution_clock::now();
                double el=std::chrono::duration<double>(t1-t0).count();
                qps_d[s].push_back(QBATCH/el);
                if(s==0) time_h.push_back(std::chrono::duration<double,std::ratio<3600>>(t1-gstart).count());
                if(step%200==0){std::cout<<".";std::cout.flush();}
            }
            std::cout<<" ";
        }
        std::cout<<"\n";

        out << "      \"time_hours\":" << vj(time_h) << ",\n";
        for(int s=0;s<NSEED;++s) out<<"      \"seed_"<<s<<"\":"<<vj(qps_d[s])<<",\n";
        std::vector<double> mn(NS,0),sd(NS,0);
        for(int i=0;i<NS;++i){for(int s=0;s<NSEED;++s)mn[i]+=qps_d[s][i];mn[i]/=NSEED;
            for(int s=0;s<NSEED;++s){double d=qps_d[s][i]-mn[i];sd[i]+=d*d;}sd[i]=sqrt(sd[i]/NSEED);}
        out<<"      \"mean\":"<<vj(mn)<<",\n      \"std\":"<<vj(sd)<<"\n";
        out << "    }";
    }
    out << "\n  }\n}\n"; out.close();
    std::cout << "  → " << path << "\n";
}

// ════════ FIGURE 3: Memory Utilization vs Steps ════════
static void fig3(const std::string& path) {
    std::cout << "[Fig3] Memory Tier Utilization\n";
    const int NS=2000, NSEED=3, BATCH=500;
    const uint64_t SD[]={42,137,271};

    std::ofstream out(path);
    out << "{\n  \"metadata\":{\"panel\":\"Memory Utilization vs Steps\",\"source\":\"Philemon-TSH\",\"n_per_seed\":"<<NS<<",\"n_seeds\":"<<NSEED<<"},\n";
    out << "  \"steps\":["; for(int i=0;i<NS;++i){if(i)out<<",";out<<i*BATCH;} out<<"],\n";

    std::vector<std::vector<double>> hbm(NSEED),gddr(NSEED),dram(NSEED),nparts(NSEED);
    for(int s=0;s<NSEED;++s){
        std::cout << "  seed " << s;
        auto edges=gen_edges(NS*BATCH,10000,100000,SD[s]);
        TieredAllocator a(512ULL<<20,1024ULL<<20,2048ULL<<20);
        TierPlacementPolicy pol(100'000'000ULL,1'000'000'000ULL);
        TemporalBridge br(a,pol,100000);

        for(int step=0;step<NS;++step){
            size_t base=step*BATCH,end=base+BATCH;
            for(size_t i=base;i<end&&i<edges.size();++i) br.add_edge(edges[i]);
            if((step+1)%10==0||step==0) br.flush_partitions();
            hbm[s].push_back(a.budget(MemoryTier::HBM).used_bytes.load()/(1024.0*1024.0));
            gddr[s].push_back(a.budget(MemoryTier::GDDR).used_bytes.load()/(1024.0*1024.0));
            dram[s].push_back(a.budget(MemoryTier::DRAM).used_bytes.load()/(1024.0*1024.0));
            nparts[s].push_back(br.partition_count());
            if(step%200==0){std::cout<<".";std::cout.flush();}
        }
        std::cout<<"\n";
    }

    auto wm=[&](const char* nm, auto& d, bool last){
        out<<"  \""<<nm<<"\":{\n";
        for(int s=0;s<NSEED;++s)out<<"    \"seed_"<<s<<"\":"<<vj(d[s])<<",\n";
        std::vector<double> mn(NS,0);
        for(int i=0;i<NS;++i){for(int s=0;s<NSEED;++s)mn[i]+=d[s][i];mn[i]/=NSEED;}
        out<<"    \"mean\":"<<vj(mn)<<"\n  }"<<(last?"\n":",\n");
    };
    wm("hbm_mb",hbm,false); wm("gddr_mb",gddr,false); wm("dram_mb",dram,false); wm("partitions",nparts,true);
    out << "}\n"; out.close();
    std::cout << "  → " << path << "\n";
}

// ════════ FIGURE 4: Migration Cost ════════
static void fig4(const std::string& path) {
    std::cout << "[Fig4] Migration Cost\n";
    const int NS=2000, NSEED=3;
    const uint64_t SD[]={42,137,271};

    std::ofstream out(path);
    out << "{\n  \"metadata\":{\"panel\":\"Migration Cost vs Steps\",\"source\":\"Philemon-TSH\",\"n_per_seed\":"<<NS<<",\"n_seeds\":"<<NSEED<<"},\n";
    out << "  \"steps\":["; for(int i=0;i<NS;++i){if(i)out<<",";out<<5000+i*245;} out<<"],\n";

    std::vector<std::vector<double>> mig_us(NSEED),mig_cnt(NSEED);
    for(int s=0;s<NSEED;++s){
        std::cout << "  seed " << s;
        auto edges=gen_edges(500000,10000,100000,SD[s]);
        for(int step=0;step<NS;++step){
            size_t n=5000+step*245;
            TieredAllocator a(512ULL<<20,1024ULL<<20,2048ULL<<20);
            TierPlacementPolicy pol(100'000'000ULL,1'000'000'000ULL);
            TemporalBridge br(a,pol,50000);
            for(size_t i=0;i<n&&i<edges.size();++i)br.add_edge(edges[i]);
            br.flush_partitions();
            // Touch to make hot
            for(int q=0;q<5;++q) br.temporal_subgraph_query(0,10000,[](const TemporalEdge&){});
            auto t0=std::chrono::high_resolution_clock::now();
            size_t mg=br.migration_sweep();
            auto t1=std::chrono::high_resolution_clock::now();
            mig_us[s].push_back(std::chrono::duration<double,std::micro>(t1-t0).count());
            mig_cnt[s].push_back(mg);
            if(step%200==0){std::cout<<".";std::cout.flush();}
        }
        std::cout<<"\n";
    }

    auto wm=[&](const char* nm, auto& d, bool last){
        out<<"  \""<<nm<<"\":{\n";
        for(int s=0;s<NSEED;++s)out<<"    \"seed_"<<s<<"\":"<<vj(d[s])<<",\n";
        std::vector<double> mn(NS,0);
        for(int i=0;i<NS;++i){for(int s=0;s<NSEED;++s)mn[i]+=d[s][i];mn[i]/=NSEED;}
        out<<"    \"mean\":"<<vj(mn)<<"\n  }"<<(last?"\n":",\n");
    };
    wm("migration_us",mig_us,false); wm("partitions_migrated",mig_cnt,true);
    out << "}\n"; out.close();
    std::cout << "  → " << path << "\n";
}

int main() {
    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << "  Philemon-TSH Data Gen (2000 pts/curve, 3 seeds)\n";
    std::cout << "═══════════════════════════════════════════════════════\n\n";
    auto t0=std::chrono::high_resolution_clock::now();
    fig1("philemon_query_latency_2000.json");
    fig2("philemon_qps_2000.json");
    fig3("philemon_memory_util_2000.json");
    fig4("philemon_migration_cost_2000.json");
    auto t1=std::chrono::high_resolution_clock::now();
    std::cout << "\nTotal: " << std::fixed << std::setprecision(1) << std::chrono::duration<double>(t1-t0).count() << "s, RSS: " << get_peak_rss_mb() << " MB\n";
    return 0;
}
