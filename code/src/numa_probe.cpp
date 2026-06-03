// numa_probe.cpp -- direct measurement of the NUMA penalty that underlies
// every render result. Two microbenchmarks, each run with the data LOCAL to
// the compute node vs REMOTE on each other node (so the SLIT distances
// 10/12/32 get turned into actual ns and GB/s):
//
//   (1) LATENCY: pointer-chase. A buffer of indices forms one random
//       permutation cycle; each load depends on the previous (no MLP, no
//       prefetch) -> measures raw dependent-load latency. This is the metric
//       that BVH-node and texture random access actually pay.
//   (2) BANDWIDTH: STREAM triad (a = b + k*c), many threads -> sustained GB/s.
//
// For node pair (compute=C, data=D): bind threads to C, allocate data on D.
// D==C is local (dist 10); D!=C gives the cross-socket figure at that pair's
// SLIT distance. Report a table of ns/access and GB/s vs distance, plus the
// remote/local penalty ratio -- the empirical "NUMA tax".
//
// Build: g++ -O3 -std=c++17 -pthread numa_probe.cpp -lnuma -o numa_probe
// Env: NP_LAT_MB (chase buffer, default 512), NP_BW_MB (per-array, default 1024),
//      NP_CHASE (chase steps, default 100M), NP_THREADS (bw threads/node, default = node cores)
#include <numa.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

static std::vector<int> cpus_in_node(int node){
    std::vector<int> out; struct bitmask* bm=numa_allocate_cpumask();
    if(numa_node_to_cpus(node,bm)==0) for(unsigned i=0;i<bm->size;i++) if(numa_bitmask_isbitset(bm,i)) out.push_back((int)i);
    numa_free_cpumask(bm);
    cpu_set_t mine; CPU_ZERO(&mine); sched_getaffinity(0,sizeof(mine),&mine);
    std::vector<int> f; for(int c:out) if(CPU_ISSET(c,&mine)) f.push_back(c); return f;
}
static void pin(int cpu){ cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu,&s); pthread_setaffinity_np(pthread_self(),sizeof(s),&s); }
static int env_int(const char*k,int d){ const char*s=getenv(k); return (!s||!*s)?d:atoi(s); }

// ---- latency: single-thread pointer chase over a random cycle on data node D, thread on node C ----
static double chase_ns(int C, int D, size_t buf_mb, long steps){
    auto cc=cpus_in_node(C); if(cc.empty()) return -1;
    size_t n = buf_mb*1024ULL*1024ULL/sizeof(uint64_t);
    uint64_t* buf=(uint64_t*)numa_alloc_onnode(n*sizeof(uint64_t), D);
    if(!buf) return -2;
    // build one random permutation cycle: buf[i] = next index
    std::vector<uint64_t> perm(n); for(size_t i=0;i<n;i++) perm[i]=i;
    std::mt19937_64 rng(12345); for(size_t i=n-1;i>0;i--){ std::uniform_int_distribution<size_t> d(0,i); std::swap(perm[i],perm[d(rng)]); }
    // first-touch on D is implicit (we write from this thread; but we want pages on D, which numa_alloc_onnode already bound). fill chain:
    for(size_t i=0;i<n;i++) buf[perm[i]] = perm[(i+1)%n];
    double ns=-1;
    std::thread t([&](){
        pin(cc[0]);
        volatile uint64_t idx=0; uint64_t cur=0;
        // warm
        for(long i=0;i<(long)n;i++) cur=buf[cur];
        auto t0=std::chrono::high_resolution_clock::now();
        for(long i=0;i<steps;i++) cur=buf[cur];
        auto t1=std::chrono::high_resolution_clock::now();
        idx=cur; (void)idx;
        ns = std::chrono::duration<double,std::nano>(t1-t0).count()/(double)steps;
    });
    t.join();
    numa_free(buf, n*sizeof(uint64_t));
    return ns;
}

// ---- bandwidth: STREAM triad, T threads on node C, arrays on node D ----
static double triad_gbs(int C, int D, size_t arr_mb, int T, int reps){
    auto cc=cpus_in_node(C); if(cc.empty()) return -1; if(T>(int)cc.size()) T=cc.size();
    size_t n=arr_mb*1024ULL*1024ULL/sizeof(double);
    double*a=(double*)numa_alloc_onnode(n*sizeof(double),D);
    double*b=(double*)numa_alloc_onnode(n*sizeof(double),D);
    double*c=(double*)numa_alloc_onnode(n*sizeof(double),D);
    if(!a||!b||!c) return -2;
    std::atomic<int> go(0); std::vector<std::thread> th;
    for(int t=0;t<T;t++) th.emplace_back([&,t](){
        pin(cc[t]); size_t lo=n*t/T, hi=n*(t+1)/T;
        for(size_t i=lo;i<hi;i++){a[i]=0;b[i]=1;c[i]=2;}
        while(!go.load()) ;
        const double k=3.0;
        for(int r=0;r<reps;r++) for(size_t i=lo;i<hi;i++) a[i]=b[i]+k*c[i];
    });
    usleep(50000);
    auto t0=std::chrono::high_resolution_clock::now(); go.store(1);
    for(auto&x:th) x.join();
    auto t1=std::chrono::high_resolution_clock::now();
    double sec=std::chrono::duration<double>(t1-t0).count();
    double bytes=(double)n*24.0*reps;
    numa_free(a,n*sizeof(double));numa_free(b,n*sizeof(double));numa_free(c,n*sizeof(double));
    return bytes/sec/1e9;
}

int main(){
    if(numa_available()<0){fprintf(stderr,"NUMA unavailable\n");return 2;}
    int nn=numa_num_configured_nodes();
    int lat_mb=env_int("NP_LAT_MB",512), bw_mb=env_int("NP_BW_MB",1024);
    long steps=(long)env_int("NP_CHASE",100)*1000000L;
    int bwT=env_int("NP_THREADS",0);
    printf("NUMA_PROBE nodes=%d lat_buf=%dMB bw_arr=%dMB chase_steps=%ld\n",nn,lat_mb,bw_mb,steps);
    printf("--- node distances (SLIT) ---\n"); fflush(stdout);
    system("numactl --hardware | sed -n '/node distances/,$p'");
    // pick compute node 0; probe data on every node (0=local, others=cross)
    int C=0; auto cc=cpus_in_node(C);
    int T = bwT>0?bwT:(int)cc.size();
    printf("\nLATENCY (thread on node %d, ns/dependent-load):\n", C);
    printf("  %-8s %-10s %-10s %-8s\n","dataNode","ns/access","vs_local","note");
    double lat_local=-1;
    for(int D=0;D<nn;D++){
        double ns=chase_ns(C,D,lat_mb,steps);
        if(D==C) lat_local=ns;
        double ratio = (lat_local>0&&ns>0)? ns/lat_local : 0;
        printf("  %-8d %-10.2f %-10s %s\n", D, ns,
               (ratio>0?[](double r){static char s[16];snprintf(s,16,"%.2fx",r);return s;}(ratio):"-"),
               D==C?"LOCAL":"cross");
        fflush(stdout);
    }
    printf("\nBANDWIDTH (%d threads on node %d, GB/s triad):\n", T, C);
    printf("  %-8s %-10s %-10s %-8s\n","dataNode","GB/s","vs_local","note");
    double bw_local=-1;
    for(int D=0;D<nn;D++){
        double g=triad_gbs(C,D,bw_mb,T,5);
        if(D==C) bw_local=g;
        double ratio=(bw_local>0&&g>0)? g/bw_local:0;
        printf("  %-8d %-10.2f %-10s %s\n", D, g,
               (ratio>0?[](double r){static char s[16];snprintf(s,16,"%.2fx",r);return s;}(ratio):"-"),
               D==C?"LOCAL":"cross");
        fflush(stdout);
    }
    printf("\nNP_DONE\n");
    return 0;
}
