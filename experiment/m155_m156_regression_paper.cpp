// ═══════════════════════════════════════════════════════════════════════════════
// M155-M156: Regression test + full paper data integration
//
// Purpose: single-binary that runs all key experiments from M145-M154 in
// sequence, validates correctness invariants, and outputs a combined data
// package suitable for the Philemon-TSH paper.
//
// This is the "final exam" — if this passes, the experiment suite is ready
// for ags1 production runs.
//
// Build: g++ -std=c++17 -O2 -fopenmp -o m155_m156 m155_m156_regression_paper.cpp -lpthread
// Run:   ./m155_m156 [--scale 16] [--threads 4]
// ═══════════════════════════════════════════════════════════════════════════════

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <omp.h>
#include <parallel/algorithm>

// ═══════════════════════════════════════════════════════════════════════════════
// Core infrastructure (shared across all tests)
// ═══════════════════════════════════════════════════════════════════════════════
namespace phi {

struct Timer {
    using clk = std::chrono::high_resolution_clock;
    clk::time_point t0;
    Timer() : t0(clk::now()) {}
    void reset() { t0 = clk::now(); }
    double s() const { return std::chrono::duration<double>(clk::now()-t0).count(); }
};

double rss_mb() {
    std::ifstream f("/proc/self/status"); std::string l;
    while (std::getline(f, l))
        if (l.rfind("VmRSS:", 0) == 0) {
            std::istringstream ss(l.substr(6)); uint64_t kb; std::string u;
            if (ss >> kb >> u) return kb/1024.0;
        }
    return -1;
}

// ─── Edge list + RMAT ────────────────────────────────────────────────────
struct EL {
    std::vector<std::pair<uint64_t,uint64_t>> edges;
    uint64_t V = 0;
};

EL rmat(int scale, int avg_deg, int thr, int seed=42) {
    uint64_t V = 1ULL<<scale, tgt = V*(uint64_t)avg_deg, logN=scale;
    double a=0.57, b=0.19, c=0.19;
    uint64_t ch = (tgt+thr-1)/thr;
    std::vector<std::vector<std::pair<uint64_t,uint64_t>>> te(thr);
    #pragma omp parallel num_threads(thr)
    {
        int tid=omp_get_thread_num();
        uint64_t s=tid*ch, e=std::min(s+ch,tgt);
        std::mt19937_64 rng(seed+(uint64_t)tid*1000003ULL);
        std::uniform_real_distribution<double> d(0,1);
        te[tid].reserve(e-s);
        for(uint64_t i=s;i<e;i++){
            uint64_t u=0,v=0;
            for(uint64_t k=logN;k>0;k--){
                double r=d(rng); uint64_t bit=1ULL<<(k-1);
                if(r<a){} else if(r<a+b) v|=bit;
                else if(r<a+b+c) u|=bit; else {u|=bit;v|=bit;}
            }
            u%=V; v%=V; if(u!=v) te[tid].push_back({u,v});
        }
    }
    EL el; el.V=V; uint64_t tot=0;
    for(auto&t:te)tot+=t.size(); el.edges.reserve(tot);
    for(auto&t:te){el.edges.insert(el.edges.end(),t.begin(),t.end());t.clear();}
    __gnu_parallel::sort(el.edges.begin(),el.edges.end());
    el.edges.erase(std::unique(el.edges.begin(),el.edges.end()),el.edges.end());
    return el;
}

// ─── CSR ─────────────────────────────────────────────────────────────────
struct CSR {
    uint64_t V=0,E=0;
    std::vector<uint64_t> off,dst;
};

CSR build_csr(const EL& el, int thr) {
    CSR c; c.V=el.V; uint64_t V=c.V, E=el.edges.size();
    std::vector<std::atomic<uint64_t>> dg(V);
    for(auto&d:dg) d.store(0);
    #pragma omp parallel for num_threads(thr)
    for(uint64_t i=0;i<E;i++) dg[el.edges[i].first].fetch_add(1,std::memory_order_relaxed);
    c.off.resize(V+1,0);
    for(uint64_t i=0;i<V;i++) c.off[i+1]=c.off[i]+dg[i].load();
    c.E=c.off[V]; c.dst.resize(c.E);
    std::vector<std::atomic<uint64_t>> cur(V);
    for(uint64_t i=0;i<V;i++) cur[i].store(c.off[i]);
    #pragma omp parallel for num_threads(thr)
    for(uint64_t i=0;i<E;i++){
        uint64_t u=el.edges[i].first,v=el.edges[i].second;
        c.dst[cur[u].fetch_add(1,std::memory_order_relaxed)]=v;
    }
    #pragma omp parallel for num_threads(thr) schedule(dynamic,256)
    for(uint64_t u=0;u<V;u++) std::sort(c.dst.begin()+c.off[u],c.dst.begin()+c.off[u+1]);
    return c;
}

// ─── Tiered CSR ──────────────────────────────────────────────────────────
enum Tier:uint8_t{HOT=0,WARM=1,COLD=2};

struct TCSR {
    uint64_t V=0,E=0;
    std::vector<Tier> vt; std::vector<uint64_t> to,dg;
    std::vector<uint64_t> hot;
    int wfd=-1; uint64_t*wm=nullptr; uint64_t wb=0; std::string wp;
    int cfd=-1; uint64_t cb=0; std::string cp;

    template<typename F> void scan(uint64_t u, F&&fn) const {
        uint64_t d=dg[u]; if(!d)return;
        switch(vt[u]){
        case HOT:{auto p=hot.data()+to[u];for(uint64_t i=0;i<d;i++)fn(p[i]);break;}
        case WARM:{auto p=wm+to[u];for(uint64_t i=0;i<d;i++)fn(p[i]);break;}
        case COLD:{
            uint64_t sb[512];uint64_t*b=(d<=512)?sb:new uint64_t[d];
            pread(cfd,b,d*8,to[u]*8);for(uint64_t i=0;i<d;i++)fn(b[i]);
            if(d>512)delete[]b;break;
        }}
    }
    ~TCSR(){
        if(wm&&wm!=MAP_FAILED)munmap(wm,wb);
        if(wfd>=0){close(wfd);unlink(wp.c_str());}
        if(cfd>=0){close(cfd);unlink(cp.c_str());}
    }
};

TCSR build_tcsr(const EL& el, int thr) {
    uint64_t V=el.V, E=el.edges.size();
    TCSR t; t.V=V; t.E=E;
    t.vt.resize(V); t.to.resize(V); t.dg.resize(V,0);
    #pragma omp parallel for num_threads(thr)
    for(uint64_t i=0;i<E;i++){
        #pragma omp atomic
        t.dg[el.edges[i].first]++;
    }
    std::vector<uint64_t> sd(t.dg.begin(),t.dg.end());
    __gnu_parallel::sort(sd.begin(),sd.end());
    uint64_t ht=sd[(uint64_t)(V*0.95)], wt=sd[(uint64_t)(V*0.50)];
    if(ht<=wt) ht=wt+1;

    uint64_t hE=0,wE=0,cE=0;
    for(uint64_t u=0;u<V;u++){
        if(t.dg[u]>ht){t.vt[u]=HOT;hE+=t.dg[u];}
        else if(t.dg[u]>wt){t.vt[u]=WARM;wE+=t.dg[u];}
        else{t.vt[u]=COLD;cE+=t.dg[u];}
    }
    t.hot.resize(hE);
    if(wE>0){
        t.wp="/tmp/phi_rw_XXXXXX"; t.wfd=mkstemp(&t.wp[0]);
        t.wb=wE*8; ftruncate(t.wfd,t.wb);
        t.wm=(uint64_t*)mmap(nullptr,t.wb,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_POPULATE,t.wfd,0);
    }
    if(cE>0){
        t.cp="/tmp/phi_rc_XXXXXX"; t.cfd=mkstemp(&t.cp[0]);
        t.cb=cE*8; ftruncate(t.cfd,t.cb);
    }
    uint64_t hc=0,wc=0,cc=0;
    for(uint64_t u=0;u<V;u++){
        switch(t.vt[u]){case HOT:t.to[u]=hc;hc+=t.dg[u];break;
        case WARM:t.to[u]=wc;wc+=t.dg[u];break;case COLD:t.to[u]=cc;cc+=t.dg[u];break;}
    }
    std::vector<std::atomic<uint64_t>> fc(V);
    for(uint64_t i=0;i<V;i++) fc[i].store(t.to[i]);
    std::vector<std::vector<uint64_t>> cbuf;
    if(cE>0) cbuf.resize(V);
    #pragma omp parallel for num_threads(thr)
    for(uint64_t i=0;i<E;i++){
        uint64_t u=el.edges[i].first,v=el.edges[i].second;
        switch(t.vt[u]){
        case HOT:t.hot[fc[u].fetch_add(1,std::memory_order_relaxed)]=v;break;
        case WARM:t.wm[fc[u].fetch_add(1,std::memory_order_relaxed)]=v;break;
        case COLD:
            #pragma omp critical
            cbuf[u].push_back(v);break;
        }
    }
    if(cE>0){
        for(uint64_t u=0;u<V;u++){
            if(t.vt[u]!=COLD||cbuf[u].empty())continue;
            std::sort(cbuf[u].begin(),cbuf[u].end());
            pwrite(t.cfd,cbuf[u].data(),cbuf[u].size()*8,t.to[u]*8);
        }
    }
    #pragma omp parallel for num_threads(thr) schedule(dynamic,256)
    for(uint64_t u=0;u<V;u++){
        if(t.dg[u]<=1)continue;
        if(t.vt[u]==HOT)std::sort(t.hot.begin()+t.to[u],t.hot.begin()+t.to[u]+t.dg[u]);
        else if(t.vt[u]==WARM)std::sort(t.wm+t.to[u],t.wm+t.to[u]+t.dg[u]);
    }
    if(t.wb>0) msync(t.wm,t.wb,MS_SYNC);
    if(t.cb>0){fsync(t.cfd);posix_fadvise(t.cfd,0,t.cb,POSIX_FADV_DONTNEED);}
    return t;
}

// ─── Algorithms ──────────────────────────────────────────────────────────
uint64_t bfs(const CSR& c, uint64_t src, int thr) {
    std::vector<char> vis(c.V,0);
    std::vector<uint64_t> fr,nx; fr.push_back(src); vis[src]=1;
    while(!fr.empty()){
        nx.clear();
        #pragma omp parallel num_threads(thr)
        {
            std::vector<uint64_t> loc;
            #pragma omp for schedule(dynamic,64)
            for(uint64_t fi=0;fi<fr.size();fi++){
                uint64_t u=fr[fi];
                for(uint64_t j=c.off[u];j<c.off[u+1];j++){
                    uint64_t v=c.dst[j];char ex=0;
                    if(__atomic_compare_exchange_n(&vis[v],&ex,(char)1,false,__ATOMIC_RELAXED,__ATOMIC_RELAXED))
                        loc.push_back(v);
                }
            }
            #pragma omp critical
            nx.insert(nx.end(),loc.begin(),loc.end());
        }
        fr.swap(nx);
    }
    uint64_t r=0; for(uint64_t i=0;i<c.V;i++) r+=vis[i]; return r;
}

uint64_t bfs_t(const TCSR& c, uint64_t src, int thr) {
    std::vector<char> vis(c.V,0);
    std::vector<uint64_t> fr,nx; fr.push_back(src); vis[src]=1;
    while(!fr.empty()){
        nx.clear();
        #pragma omp parallel num_threads(thr)
        {
            std::vector<uint64_t> loc;
            #pragma omp for schedule(dynamic,64)
            for(uint64_t fi=0;fi<fr.size();fi++){
                uint64_t u=fr[fi];
                c.scan(u,[&](uint64_t v){
                    char ex=0;
                    if(__atomic_compare_exchange_n(&vis[v],&ex,(char)1,false,__ATOMIC_RELAXED,__ATOMIC_RELAXED))
                        loc.push_back(v);
                });
            }
            #pragma omp critical
            nx.insert(nx.end(),loc.begin(),loc.end());
        }
        fr.swap(nx);
    }
    uint64_t r=0; for(uint64_t i=0;i<c.V;i++) r+=vis[i]; return r;
}

double pr(const CSR& c, int iters, int thr) {
    uint64_t V=c.V;
    std::vector<double> sc(V,1.0/V),ct(V),ns(V);
    for(int it=0;it<iters;it++){
        #pragma omp parallel for num_threads(thr)
        for(uint64_t u=0;u<V;u++){uint64_t d=c.off[u+1]-c.off[u];ct[u]=d>0?sc[u]/d:0;}
        std::fill(ns.begin(),ns.end(),0.15/V);
        #pragma omp parallel for num_threads(thr)
        for(uint64_t u=0;u<V;u++){
            double cc=ct[u];
            for(uint64_t j=c.off[u];j<c.off[u+1];j++){
                #pragma omp atomic
                ns[c.dst[j]]+=0.85*cc;
            }
        }
        sc.swap(ns);
    }
    double l1=0,base=1.0/V;
    for(uint64_t i=0;i<V;i++) l1+=std::fabs(sc[i]-base);
    return l1;
}

double pr_t(const TCSR& c, int iters, int thr) {
    uint64_t V=c.V;
    std::vector<double> sc(V,1.0/V),ct(V),ns(V);
    for(int it=0;it<iters;it++){
        #pragma omp parallel for num_threads(thr)
        for(uint64_t u=0;u<V;u++){ct[u]=c.dg[u]>0?sc[u]/c.dg[u]:0;}
        std::fill(ns.begin(),ns.end(),0.15/V);
        #pragma omp parallel for num_threads(thr) schedule(dynamic,256)
        for(uint64_t u=0;u<V;u++){
            double cc=ct[u];
            c.scan(u,[&](uint64_t v){
                #pragma omp atomic
                ns[v]+=0.85*cc;
            });
        }
        sc.swap(ns);
    }
    double l1=0,base=1.0/V;
    for(uint64_t i=0;i<V;i++) l1+=std::fabs(sc[i]-base);
    return l1;
}

} // namespace phi

// ═══════════════════════════════════════════════════════════════════════════════
// Regression tests
// ═══════════════════════════════════════════════════════════════════════════════
int main(int argc, char* argv[]) {
    int scale = 16, thr = 4, pr_iters = 5;
    for(int i=1;i<argc;i++){
        std::string a=argv[i];
        if(a=="--scale"&&i+1<argc)scale=std::atoi(argv[++i]);
        else if(a=="--threads"&&i+1<argc)thr=std::atoi(argv[++i]);
        else if(a=="--pr-iters"&&i+1<argc)pr_iters=std::atoi(argv[++i]);
        else if(a=="-h"||a=="--help"){
            std::printf("usage: %s [--scale N] [--threads N]\n",argv[0]); return 0;
        }
    }

    int pass=0, fail=0, total=0;
    auto CHECK = [&](bool ok, const char* msg) {
        total++;
        if(ok){pass++;std::printf("  PASS: %s\n",msg);}
        else{fail++;std::fprintf(stderr,"  FAIL: %s\n",msg);}
    };

    uint64_t V = 1ULL<<scale;
    std::printf("═══════════════════════════════════════════════════════\n");
    std::printf(" M155-M156: Regression suite + paper data\n");
    std::printf(" scale=%d V=%lu threads=%d pr_iters=%d\n",
                scale,(unsigned long)V,thr,pr_iters);
    std::printf("═══════════════════════════════════════════════════════\n\n");

    // Generate graph once
    auto el = phi::rmat(scale, 32, thr);
    std::printf("[GEN] %lu edges (avg_deg=%.2f)\n\n",
                (unsigned long)el.edges.size(),
                (double)el.edges.size()/V);

    // ── Test 1: CSR build + BFS correctness ──
    std::printf("── Test 1: CSR build + BFS ──\n");
    phi::Timer t1;
    auto csr = phi::build_csr(el, thr);
    double build_s = t1.s();
    double insert_meps = (double)el.edges.size()/build_s/1e6;
    std::printf("  CSR build: %.3fs (%.1f MEPS)\n", build_s, insert_meps);

    phi::Timer bfs_t;
    uint64_t bfs_reach = phi::bfs(csr, 0, thr);
    double bfs_s = bfs_t.s();
    std::printf("  BFS: %.4fs, reached %lu/%lu (%.1f%%)\n",
                bfs_s,(unsigned long)bfs_reach,(unsigned long)V,100.0*bfs_reach/V);

    CHECK(csr.E == el.edges.size(), "CSR edge count matches");
    CHECK(bfs_reach > 0, "BFS reached >0 vertices");
    CHECK(bfs_reach <= V, "BFS reached <= V");
    CHECK(insert_meps > 1.0, "Insert throughput > 1 MEPS");

    // ── Test 2: PR correctness ──
    std::printf("\n── Test 2: PageRank ──\n");
    phi::Timer pr_t;
    double pr_l1 = phi::pr(csr, pr_iters, thr);
    double pr_s = pr_t.s();
    std::printf("  PR %d-iter: %.4fs, L1=%.6f\n", pr_iters, pr_s, pr_l1);
    CHECK(pr_l1 > 0, "PR L1 > 0");
    CHECK(pr_s > 0, "PR time > 0");

    // ── Test 3: Tiered CSR correctness ──
    std::printf("\n── Test 3: Tiered CSR ──\n");
    auto tcsr = phi::build_tcsr(el, thr);

    uint64_t bfs_t_reach = phi::bfs_t(tcsr, 0, thr);
    double pr_t_l1 = phi::pr_t(tcsr, pr_iters, thr);

    std::printf("  Tiered BFS reached: %lu\n", (unsigned long)bfs_t_reach);
    std::printf("  Tiered PR L1: %.6f\n", pr_t_l1);

    CHECK(bfs_t_reach == bfs_reach, "Tiered BFS reach == CSR BFS reach");
    CHECK(std::fabs(pr_t_l1 - pr_l1) < pr_l1 * 0.01,
          "Tiered PR L1 within 1% of CSR PR L1");

    // ── Test 4: Tiered performance ratio ──
    std::printf("\n── Test 4: Performance ratios ──\n");
    phi::Timer bfs_base_t; phi::bfs(csr, 0, thr); double bb = bfs_base_t.s();
    phi::Timer bfs_tier_t; phi::bfs_t(tcsr, 0, thr); double bt = bfs_tier_t.s();
    double bfs_ratio = bt / bb;

    phi::Timer pr_base_t; phi::pr(csr, pr_iters, thr); double pb = pr_base_t.s();
    phi::Timer pr_tier_t; phi::pr_t(tcsr, pr_iters, thr); double pt = pr_tier_t.s();
    double pr_ratio = pt / pb;

    std::printf("  BFS: base=%.4f tier=%.4f ratio=%.2fx\n", bb, bt, bfs_ratio);
    std::printf("  PR:  base=%.4f tier=%.4f ratio=%.2fx\n", pb, pt, pr_ratio);

    CHECK(bfs_ratio < 5.0, "BFS tiered slowdown < 5x");
    CHECK(pr_ratio < 5.0, "PR tiered slowdown < 5x");

    // ── Test 5: Search benchmark ──
    std::printf("\n── Test 5: Search latency ──\n");
    uint64_t search_ops = std::min((uint64_t)500000, el.edges.size());
    std::mt19937_64 rng(42);
    std::vector<std::pair<uint64_t,uint64_t>> queries(search_ops);
    for(uint64_t i=0;i<search_ops;i++){
        if(i%2==0) queries[i]=el.edges[rng()%el.edges.size()]; // hit
        else{queries[i]={rng()%V,rng()%V};} // maybe miss
    }
    uint64_t hits=0;
    phi::Timer st;
    #pragma omp parallel for num_threads(thr) reduction(+:hits)
    for(uint64_t i=0;i<search_ops;i++){
        uint64_t u=queries[i].first,v=queries[i].second;
        if(u>=V)continue;
        auto b=csr.dst.begin()+csr.off[u], e=csr.dst.begin()+csr.off[u+1];
        if(std::binary_search(b,e,v)) hits++;
    }
    double search_s = st.s();
    double ns_per_op = search_s*1e9/search_ops;
    std::printf("  %lu ops, %lu hits (%.1f%%), %.1f ns/op\n",
                (unsigned long)search_ops,(unsigned long)hits,
                100.0*hits/search_ops,ns_per_op);
    CHECK(hits > 0, "Search has >0 hits");
    CHECK(ns_per_op < 10000, "Search latency < 10us/op");

    // ═══════════════════════════════════════════════════════════════════════
    // Paper data summary
    // ═══════════════════════════════════════════════════════════════════════
    double rss = phi::rss_mb();
    std::printf("\n═══════════════════════════════════════════════════════\n");
    std::printf(" PAPER DATA (scale=%d, %d threads)\n", scale, thr);
    std::printf("═══════════════════════════════════════════════════════\n");
    std::printf("  Insert       : %.1f MEPS\n", insert_meps);
    std::printf("  BFS (DRAM)   : %.4f s\n", bb);
    std::printf("  BFS (tiered) : %.4f s  (%.2fx)\n", bt, bfs_ratio);
    std::printf("  PR  (DRAM)   : %.4f s\n", pb);
    std::printf("  PR  (tiered) : %.4f s  (%.2fx)\n", pt, pr_ratio);
    std::printf("  Search       : %.1f ns/op\n", ns_per_op);
    std::printf("  Memory       : %.1f MB\n", rss);
    std::printf("═══════════════════════════════════════════════════════\n\n");

    std::printf("  %% LaTeX summary row:\n");
    std::printf("  %%   Philemon & %.1f & %.4f & %.2fx & %.4f & %.2fx & %.1f & %.1f \\\\\n",
                insert_meps, bb, bfs_ratio, pb, pr_ratio, ns_per_op, rss);

    // ═══════════════════════════════════════════════════════════════════════
    // Final verdict
    // ═══════════════════════════════════════════════════════════════════════
    std::printf("\n═══════════════════════════════════════════════════════\n");
    std::printf(" REGRESSION: %d PASS, %d FAIL (of %d)\n", pass, fail, total);
    std::printf("═══════════════════════════════════════════════════════\n");

    return fail > 0 ? 1 : 0;
}
