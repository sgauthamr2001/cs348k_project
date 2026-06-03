// rtiow_hairball.cpp -- procedural Hairball stress for NUMA experiments.
//
// Standalone (does NOT include rtiow_pathtrace machinery). Uses triangles
// instead of spheres. Same N/R/P modes, same ray cycling, same xorshift
// path tracer. Per-hit work is much heavier than the sphere version:
// - triangle indexed into a separate vertex array (3 cache-line fetches
//   for positions vs 1 sphere center fetch)
// - per-vertex normals; shade-time barycentric interpolation
// - tiny triangles -> deep narrow BVH with overlapping bounds
//
// Procedural scene: N_STRANDS curves emerging from a small core sphere,
// each tessellated as a tube of SIDES facets with SEGMENTS segments.
// Default 600/120/12 -> ~1.7M triangles. NSTR=1000 SEG=120 SID=15 -> 3.6M.
//
// Build:  g++ -O3 -std=c++17 -pthread rtiow_hairball.cpp -lnuma -o rtiow_hairball
// Env:
//   RTIOW_MODE=naive|replicate|partition|hybrid  (default partition)
//   RTIOW_REPL_DEPTH=K        (hybrid mode: replicate top K BVH levels
//                              per socket; cold tail stays interleaved.
//                              K=0 -> identical to naive; K large -> ~R)
//   RTIOW_PROFILE_BVH=1       (instrument traversal to count per-node
//                              visits; dump histogram by BFS level)
//   RTIOW_NSOCKETS=K                       (default 2)
//   RTIOW_NSTRANDS=N                       (default 800)
//   RTIOW_SEGMENTS=S                       (default 120)
//   RTIOW_SIDES=M                          (default 12)
//   RTIOW_IMAGE_WIDTH=W                    (default 1200)
//   RTIOW_SAMPLES=S                        (default 1)
//   RTIOW_MAX_DEPTH=D                      (default 5)
//   RTIOW_LEAF_MAX=L                       (default 4)
//   RTIOW_SEED=N                           (default 42)
//
// BENCH_STATS line tagged tag=rtiow_hairball.

#include <numa.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

// ---------- math ----------
struct Vec3 {
    float x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float a, float b, float c) : x(a), y(b), z(c) {}
    Vec3 operator+(const Vec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3 operator*(const Vec3& o) const { return {x*o.x, y*o.y, z*o.z}; }
    Vec3 operator*(float s) const { return {x*s, y*s, z*s}; }
    Vec3 operator/(float s) const { return {x/s, y/s, z/s}; }
    float dot(const Vec3& o) const { return x*o.x + y*o.y + z*o.z; }
    Vec3 cross(const Vec3& o) const {
        return {y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x};
    }
    float length() const { return std::sqrt(dot(*this)); }
    float length_sq() const { return dot(*this); }
    Vec3 normalized() const { float l = length(); return l > 0 ? *this / l : *this; }
};

static inline Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - n * (2.0f * v.dot(n));
}
static inline bool near_zero(const Vec3& v) {
    return std::fabs(v.x) < 1e-8f && std::fabs(v.y) < 1e-8f && std::fabs(v.z) < 1e-8f;
}

// ---------- geometry (SPHERES: analytic, compute-bound contrast to hairball) ----------
constexpr uint32_t MAT_LAMBERTIAN = 0, MAT_METAL = 1, MAT_DIELECTRIC = 2;
struct Sphere {
    Vec3 center; float radius;       // 16
    Vec3 albedo; uint32_t material;  // 16
    float fuzz; float ir;            // 8
};  // 40 bytes -- self-describing (no vertex array, no palette)

struct AABB {
    float mn[3], mx[3];
    AABB() { for (int i = 0; i < 3; ++i) { mn[i] = 1e30f; mx[i] = -1e30f; } }
    void expand(const Vec3& p) {
        mn[0] = std::min(mn[0], p.x); mx[0] = std::max(mx[0], p.x);
        mn[1] = std::min(mn[1], p.y); mx[1] = std::max(mx[1], p.y);
        mn[2] = std::min(mn[2], p.z); mx[2] = std::max(mx[2], p.z);
    }
    void expand(const AABB& o) {
        for (int i = 0; i < 3; ++i) {
            mn[i] = std::min(mn[i], o.mn[i]);
            mx[i] = std::max(mx[i], o.mx[i]);
        }
    }
    void expand_sphere(const Sphere& s) {
        Vec3 r{s.radius, s.radius, s.radius};
        expand(s.center - r); expand(s.center + r);
    }
    bool intersect(const Vec3& o, const Vec3& invd, float& t_near, float& t_far) const {
        float tx1 = (mn[0]-o.x)*invd.x, tx2 = (mx[0]-o.x)*invd.x;
        float ty1 = (mn[1]-o.y)*invd.y, ty2 = (mx[1]-o.y)*invd.y;
        float tz1 = (mn[2]-o.z)*invd.z, tz2 = (mx[2]-o.z)*invd.z;
        t_near = std::max({std::min(tx1,tx2), std::min(ty1,ty2), std::min(tz1,tz2), 0.0f});
        t_far  = std::min({std::max(tx1,tx2), std::max(ty1,ty2), std::max(tz1,tz2)});
        return t_far >= t_near;
    }
};

// Path = ray segment + cumulative throughput. Same shape as cp7 sphere version.
struct alignas(8) Path {
    Vec3 o, d, invd;          // 36
    float t_max;              // 4
    int32_t hit_tri_id;       // 4  -- side_id in top 4 bits, tri idx in bottom 28
    int32_t pixel_id;         // 4
    Vec3 throughput;          // 12
    float bary_u, bary_v;     // 8   -- saved at hit for shade-time normal interp
    uint16_t sockets_pending; // 2
    uint8_t depth;            // 1
    uint8_t max_depth;        // 1
};
// 72 bytes

struct BVHNode2 {
    AABB bbox;             // 24
    int32_t left;          // 4
    int32_t right_or_first;// 4
    uint16_t count;        // 2  -- 0 = internal
    uint16_t axis;         // 2
};
// 36 bytes (pads to 40)

// ---------- NUMA helpers ----------
static std::vector<int> cpus_in_node(int node) {
    std::vector<int> out;
    struct bitmask* bm = numa_allocate_cpumask();
    if (numa_node_to_cpus(node, bm) == 0) {
        for (unsigned i = 0; i < bm->size; ++i)
            if (numa_bitmask_isbitset(bm, i)) out.push_back((int)i);
    }
    numa_free_cpumask(bm);
    cpu_set_t mine; CPU_ZERO(&mine);
    sched_getaffinity(0, sizeof(mine), &mine);
    std::vector<int> f;
    for (int c : out) if (CPU_ISSET(c, &mine)) f.push_back(c);
    return f;
}
static void pin_thread_to_cpu(int cpu) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
}
template<typename T>
static T* numa_new(size_t n, int node) {
    void* p = numa_alloc_onnode(n * sizeof(T), node);
    if (!p) { fprintf(stderr, "numa_alloc_onnode failed n=%zu node=%d\n", n, node); exit(1); }
    return (T*)p;
}

// cp13: logical-socket -> physical-NUMA-node mapping. Built at startup from
// the nodes that actually have CPUs in our cpuset (so `numactl
// --cpunodebind=0,2 --membind=0,2` makes the binary use physical nodes 0
// and 2 as its two logical sockets, instead of assuming 0,1). This lets us
// pin a 2-socket run to a chosen NUMA-node PAIR and thereby select the
// cross-socket distance (e.g. {0,1}=12, {0,2}=32 on the 4-socket box).
static std::vector<int> g_phys_node;  // g_phys_node[logical_socket] = physical node id
static inline int phys(int logical_socket) {
    return (logical_socket >= 0 && logical_socket < (int)g_phys_node.size())
           ? g_phys_node[logical_socket] : logical_socket;
}
// Discover NUMA nodes that have >=1 CPU in our cpuset, in ascending order.
static std::vector<int> active_numa_nodes() {
    std::vector<int> out;
    int maxn = numa_num_configured_nodes();
    for (int n = 0; n < maxn; ++n) {
        if (!cpus_in_node(n).empty()) out.push_back(n);
    }
    if (out.empty()) out.push_back(0);
    return out;
}

// ---------- env ----------
static int env_int(const char* k, int def) {
    const char* s = std::getenv(k); return (!s || !*s) ? def : atoi(s);
}
static const char* env_str(const char* k, const char* def) {
    const char* s = std::getenv(k); return (!s || !*s) ? def : s;
}

enum class Mode { Partition, Replicate, Naive, Hybrid };
static Mode parse_mode(const char* s) {
    if (s && !strcmp(s, "replicate")) return Mode::Replicate;
    if (s && !strcmp(s, "naive"))     return Mode::Naive;
    if (s && !strcmp(s, "hybrid"))    return Mode::Hybrid;
    return Mode::Partition;
}
static const char* mode_name(Mode m) {
    switch (m) {
        case Mode::Replicate: return "replicate";
        case Mode::Naive:     return "naive";
        case Mode::Hybrid:    return "hybrid";
        default:              return "partition";
    }
}

// ---------- RNG ----------
struct Rng {
    uint64_t s[2];
    void seed(uint64_t a, uint64_t b) { s[0] = a ? a : 1; s[1] = b ? b : 1; }
    uint64_t next_u64() {
        uint64_t x = s[0], y = s[1];
        s[0] = y; x ^= x << 23;
        s[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
        return s[1] + y;
    }
    float next_f() { return (next_u64() >> 40) * (1.0f / 16777216.0f); }
    float range(float a, float b) { return a + (b - a) * next_f(); }
    Vec3 unit_in_sphere() {
        for (int i = 0; i < 16; ++i) {
            Vec3 p{range(-1,1), range(-1,1), range(-1,1)};
            if (p.length_sq() < 1.0f) return p;
        }
        return Vec3{1,0,0};
    }
    Vec3 random_unit() { return unit_in_sphere().normalized(); }
};

// ---------- HAIRBALL SCENE GENERATION ----------
// N strands radiating from a small core sphere, each a random-walk
// tube of (SEGMENTS+1) cross-sections with SIDES facets per ring.
// Triangles per strand = SEGMENTS * SIDES * 2. Default 800/120/12 ->
// 2.3M triangles.
struct ScenePalette { std::vector<Vec3> colors; };  // unused for spheres (kept for ctx parity)

// RTIOW sphere grid. grid_half=N -> (2N)^2 candidate cells + 4 hero spheres.
// At N=500 ~1M spheres. Matches the cp2/cp5 RTIOW scene RNG order.
static void gen_spheres(int grid_half, uint32_t seed, std::vector<Sphere>& sp) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    sp.push_back({{0,-1000,0},1000,{0.5f,0.5f,0.5f},MAT_LAMBERTIAN,0,1.5f});
    sp.push_back({{0,1,0},1,{0,0,0},MAT_DIELECTRIC,0,1.5f});
    sp.push_back({{-4,1,0},1,{0.4f,0.2f,0.1f},MAT_LAMBERTIAN,0,1.5f});
    sp.push_back({{4,1,0},1,{0.7f,0.6f,0.5f},MAT_METAL,0,1.5f});
    for (int ai=-grid_half; ai<grid_half; ++ai) for (int bi=-grid_half; bi<grid_half; ++bi){
        float choose=u(rng); float jx=u(rng), jz=u(rng);
        Vec3 c{ai+0.9f*jx, 0.2f, bi+0.9f*jz};
        if ((c - Vec3{4,0.2f,0}).length() <= 0.9f) continue;
        if (choose<0.8f){ float r0x=u(rng),r0y=u(rng),r0z=u(rng),r1x=u(rng),r1y=u(rng),r1z=u(rng);
            sp.push_back({c,0.2f,{r0x*r1x,r0y*r1y,r0z*r1z},MAT_LAMBERTIAN,0,1.5f});
        } else if (choose<0.95f){ float ax=u(rng),ay=u(rng),az=u(rng);
            sp.push_back({c,0.2f,{0.5f+0.5f*ax,0.5f+0.5f*ay,0.5f+0.5f*az},MAT_METAL,0.5f*u(rng),1.5f});
        } else sp.push_back({c,0.2f,{0,0,0},MAT_DIELECTRIC,0,1.5f});
    }
}

// ---------- Sphere intersection (analytic) ----------
static inline bool sphere_hit(const Sphere& s, const Vec3& o, const Vec3& d,
                              float t_min, float t_max, float& t_hit) {
    Vec3 oc = o - s.center;
    float a = d.dot(d);
    float half_b = oc.dot(d);
    float c = oc.dot(oc) - s.radius * s.radius;
    float disc = half_b*half_b - a*c;
    if (disc < 0) return false;
    float sq = std::sqrt(disc);
    float root = (-half_b - sq) / a;
    if (root < t_min || root > t_max) { root = (-half_b + sq) / a; if (root < t_min || root > t_max) return false; }
    t_hit = root;
    return true;
}
// ---------- materials / scatter (RTIOW) ----------
static inline Vec3 reflectv(const Vec3& v, const Vec3& n){ return v - n*(2.0f*v.dot(n)); }
static inline Vec3 refractv(const Vec3& uv, const Vec3& n, float eta){
    float ct = std::min(-uv.dot(n), 1.0f);
    Vec3 perp = (uv + n*ct) * eta;
    Vec3 par = n * -std::sqrt(std::fabs(1.0f - perp.length_sq()));
    return perp + par;
}
static inline float schlick(float cos, float ri){ float r0=(1-ri)/(1+ri); r0*=r0; return r0+(1-r0)*std::pow(1-cos,5.0f); }

// ---------- BVH builder over spheres ----------
struct BVH {
    BVHNode2* nodes = nullptr;           // shared/cold (hybrid: hot_nodes for idx<hot_count)
    BVHNode2* hot_nodes = nullptr;       // hybrid: per-socket local copy of top hot_count BFS nodes
    size_t    hot_count = 0;
    std::atomic<uint64_t>* visits = nullptr; // profile: per-node visit counter
    Sphere*   prims = nullptr;
    size_t    n_nodes = 0, n_prims = 0;
    AABB      world_bbox;
    int       owner_node = -1;
    int       side_id = 0;
};

static uint32_t build_rec(std::vector<BVHNode2>& out, std::vector<Sphere>& sp,
                          uint32_t lo, uint32_t hi, int depth, uint32_t leaf_max) {
    uint32_t idx = (uint32_t)out.size();
    out.emplace_back();
    AABB bb;
    for (uint32_t i = lo; i < hi; ++i) bb.expand_sphere(sp[i]);
    uint32_t count = hi - lo;
    if (count <= leaf_max || depth > 60) {
        out[idx].bbox = bb; out[idx].left = -1;
        out[idx].right_or_first = (int32_t)lo;
        out[idx].count = (uint16_t)count; out[idx].axis = 0;
        return idx;
    }
    float ex = bb.mx[0]-bb.mn[0], ey = bb.mx[1]-bb.mn[1], ez = bb.mx[2]-bb.mn[2];
    int ax = 0; float em = ex;
    if (ey > em) { ax = 1; em = ey; }
    if (ez > em) { ax = 2; }
    uint32_t mid = lo + count/2;
    std::nth_element(sp.begin()+lo, sp.begin()+mid, sp.begin()+hi,
        [ax](const Sphere& a, const Sphere& b){
            return (&a.center.x)[ax] < (&b.center.x)[ax];
        });
    uint32_t l = build_rec(out, sp, lo, mid, depth+1, leaf_max);
    uint32_t r = build_rec(out, sp, mid, hi, depth+1, leaf_max);
    out[idx].bbox = bb; out[idx].left = (int32_t)l;
    out[idx].right_or_first = (int32_t)r;
    out[idx].count = 0; out[idx].axis = (uint16_t)ax;
    return idx;
}

static void build_bvh_on(BVH& out, std::vector<Sphere> sp, int node, int side_id, uint32_t leaf_max) {
    out.owner_node = node;
    out.side_id = side_id;
    out.n_prims = sp.size();
    std::vector<BVHNode2> tmp;
    tmp.reserve(sp.size() * 2);
    if (!sp.empty()) build_rec(tmp, sp, 0, (uint32_t)sp.size(), 0, leaf_max);
    out.n_nodes = tmp.size();
    out.prims = numa_new<Sphere>(std::max<size_t>(1, sp.size()), node);
    out.nodes = numa_new<BVHNode2>(std::max<size_t>(1, tmp.size()), node);
    std::thread ft([&]() {
        auto cs = cpus_in_node(node);
        if (!cs.empty()) pin_thread_to_cpu(cs[0]);
        if (!sp.empty())  memcpy(out.prims, sp.data(),  sp.size() * sizeof(Sphere));
        if (!tmp.empty()) memcpy(out.nodes, tmp.data(), tmp.size() * sizeof(BVHNode2));
    });
    ft.join();
    out.world_bbox = tmp.empty() ? AABB{} : tmp[0].bbox;
}

static void build_bvh_interleaved(BVH& out, std::vector<Sphere> sp, uint32_t leaf_max) {
    out.owner_node = -1;
    out.side_id = 0;
    out.n_prims = sp.size();
    std::vector<BVHNode2> tmp;
    tmp.reserve(sp.size() * 2);
    if (!sp.empty()) build_rec(tmp, sp, 0, (uint32_t)sp.size(), 0, leaf_max);
    out.n_nodes = tmp.size();
    size_t pb = std::max<size_t>(64, sp.size() * sizeof(Sphere));
    size_t nb = std::max<size_t>(64, tmp.size() * sizeof(BVHNode2));
    out.prims = (Sphere*)numa_alloc_interleaved(pb);
    out.nodes = (BVHNode2*)numa_alloc_interleaved(nb);
    if (!out.prims || !out.nodes) { fprintf(stderr, "numa_alloc_interleaved failed\n"); exit(1); }
    if (!sp.empty())  memcpy(out.prims, sp.data(),  sp.size() * sizeof(Sphere));
    if (!tmp.empty()) memcpy(out.nodes, tmp.data(), tmp.size() * sizeof(BVHNode2));
    out.world_bbox = tmp.empty() ? AABB{} : tmp[0].bbox;
}

// BFS-reorder a BVH in place: nodes[0..M-1] in BFS order, child indices
// remapped. This makes "top K levels" correspond to the first prefix of
// the array, so hybrid mode can slice it cleanly.
static void bfs_reorder_inplace(BVH& bvh) {
    if (bvh.n_nodes == 0) return;
    size_t N = bvh.n_nodes;
    std::vector<int32_t> bfs_order; bfs_order.reserve(N);
    std::vector<int32_t> old_to_new(N, -1);
    std::vector<int32_t> q; q.push_back(0);
    while (!q.empty()) {
        std::vector<int32_t> next;
        for (int32_t old_idx : q) {
            old_to_new[old_idx] = (int32_t)bfs_order.size();
            bfs_order.push_back(old_idx);
            const BVHNode2& n = bvh.nodes[old_idx];
            if (n.count == 0) {
                next.push_back(n.left);
                next.push_back(n.right_or_first);
            }
        }
        q.swap(next);
    }
    // Rewrite the nodes array in-place via a tmp copy.
    std::vector<BVHNode2> tmp(N);
    for (size_t i = 0; i < N; ++i) {
        BVHNode2 src = bvh.nodes[bfs_order[i]];
        if (src.count == 0) {
            src.left = old_to_new[src.left];
            src.right_or_first = old_to_new[src.right_or_first];
        }
        tmp[i] = src;
    }
    memcpy(bvh.nodes, tmp.data(), N * sizeof(BVHNode2));
}

// Hybrid setup: assumes bvh.nodes already BFS-ordered and shared
// (interleaved). For each socket s, allocate a hot-prefix array of size
// hot_count on node s and copy the first hot_count nodes into it.
// Returns memory used by the per-socket hot blocks (sum across sockets).
static size_t alloc_hot_per_socket(std::vector<BVH>& bvhs, int n_sockets,
                                   size_t hot_count) {
    size_t total_extra = 0;
    for (int s = 0; s < n_sockets; ++s) {
        bvhs[s].hot_count = hot_count;
        if (hot_count == 0) { bvhs[s].hot_nodes = nullptr; continue; }
        bvhs[s].hot_nodes = numa_new<BVHNode2>(hot_count, phys(s));
        // First-touch from a thread pinned to socket s so pages bind locally
        std::thread ft([&, s](){
            auto cs = cpus_in_node(phys(s));
            if (!cs.empty()) pin_thread_to_cpu(cs[0]);
            memcpy(bvhs[s].hot_nodes, bvhs[0].nodes, hot_count * sizeof(BVHNode2));
        });
        ft.join();
        total_extra += hot_count * sizeof(BVHNode2);
    }
    return total_extra;
}

static bool traverse(const BVH& bvh, Path& p) {
    if (bvh.n_nodes == 0) return false;
    int32_t stack[128];
    int sp = 0;
    stack[sp++] = 0;
    float t_max = p.t_max;
    int32_t best = p.hit_tri_id;
    float bu = p.bary_u, bv = p.bary_v;
    bool any = false;
    while (sp > 0) {
        int32_t ni = stack[--sp];
        // In hybrid mode, hot prefix is per-socket-local; cold tail is
        // shared. Single branch on hot_count (constant per render).
        const BVHNode2& n = ((uint32_t)ni < (uint32_t)bvh.hot_count)
                            ? bvh.hot_nodes[ni] : bvh.nodes[ni];
        if (bvh.visits) bvh.visits[ni].fetch_add(1, std::memory_order_relaxed);
        float tn, tf;
        if (!n.bbox.intersect(p.o, p.invd, tn, tf)) continue;
        if (tn > t_max) continue;
        if (n.count > 0) {
            int32_t first = n.right_or_first;
            for (uint16_t k = 0; k < n.count; ++k) {
                uint32_t pi = (uint32_t)first + k;
                float th;
                if (sphere_hit(bvh.prims[pi], p.o, p.d, 0.001f, t_max, th)) {
                    t_max = th;
                    best = (int32_t)((uint32_t)pi | ((uint32_t)bvh.side_id << 28));
                    any = true;
                }
            }
        } else {
            int32_t L = n.left, R = n.right_or_first;
            float dax = (&p.d.x)[n.axis];
            if (dax < 0) std::swap(L, R);
            if (sp + 2 < 128) { stack[sp++] = R; stack[sp++] = L; }
        }
    }
    p.t_max = t_max; p.hit_tri_id = best; p.bary_u = bu; p.bary_v = bv;
    return any;
}

// ---------- Sky ----------
static Vec3 sky_color(const Vec3& dir) {
    Vec3 ud = dir.normalized();
    float t = 0.5f * (ud.y + 1.0f);
    return Vec3{1,1,1} * (1.0f - t) + Vec3{0.5f, 0.7f, 1.0f} * t;
}

// ---------- Queue ----------
// Queue: per-socket ray queue. Two mechanisms, selected by RTIOW_QUEUE:
//  "mutex" (default) -- std::mutex + condition_variable, default-allocated
//      storage (heavyweight; the original cp8-17 path).
//  "local" -- spinlock + spin-poll (no condvar), and the Path storage is
//      first-touched onto the destination socket's local DRAM (reserve once,
//      memset from a thread pinned to that node). This is the Wald-spirit
//      bulk/NUMA-local handoff: producers write the forwarded batch directly
//      into the consumer socket's local memory, consumers read locally, and
//      there is no condvar wakeup latency. Same algorithm + same per-path
//      logic as mutex mode -> identical results (verified on dev box).
struct Queue {
    std::mutex m;
    std::condition_variable cv;
    std::atomic_flag spin = ATOMIC_FLAG_INIT;   // used in local mode
    bool local = false;
    std::vector<Path> paths;
    std::atomic<uint64_t> total_pushed{0};
    std::atomic<uint64_t> total_popped{0};
    inline void slock(){ while (spin.test_and_set(std::memory_order_acquire)) __builtin_ia32_pause(); }
    inline void sunlock(){ spin.clear(std::memory_order_release); }
};

// ---------- Camera ----------
struct Camera {
    Vec3 origin, lower_left, horizontal, vertical;
    Vec3 u_basis, v_basis;
    float lens_radius;
    int width, height;
};
static Camera make_camera(int width) {
    Camera c;
    c.width = width;
    float aspect = 16.0f/9.0f;
    c.height = (int)(width / aspect);
    // Looking at the hairball core. Distance ~6 so the tangle nearly fills view.
    Vec3 lookfrom{4, 2, 6};
    Vec3 lookat{0, 0, 0};
    Vec3 vup{0, 1, 0};
    float vfov = 45.0f * 3.14159265f / 180.0f;
    float focus_dist = 6.0f;
    float h = std::tan(vfov/2);
    float vp_h = 2.0f * h * focus_dist;
    float vp_w = aspect * vp_h;
    Vec3 w = (lookfrom - lookat).normalized();
    Vec3 u = vup.cross(w).normalized();
    Vec3 v = w.cross(u);
    c.origin = lookfrom;
    c.horizontal = u * vp_w;
    c.vertical = v * vp_h;
    c.lower_left = c.origin - c.horizontal*0.5f - c.vertical*0.5f - w*focus_dist;
    c.u_basis = u; c.v_basis = v;
    c.lens_radius = 0.0f; // no DOF
    return c;
}
static Path make_primary(const Camera& c, int x, int y, int pixel_id, uint8_t max_depth, Rng& rng) {
    float fx = (x + rng.next_f()) / (float)c.width;
    float fy = 1.0f - (y + rng.next_f()) / (float)c.height;
    Vec3 origin = c.origin;
    Vec3 dir = (c.lower_left + c.horizontal*fx + c.vertical*fy - origin).normalized();
    Path p;
    p.o = origin; p.d = dir;
    p.invd = {1.0f/dir.x, 1.0f/dir.y, 1.0f/dir.z};
    p.t_max = 1e30f; p.hit_tri_id = -1; p.pixel_id = pixel_id;
    p.throughput = {1,1,1}; p.sockets_pending = 0;
    p.depth = 1; p.max_depth = max_depth;
    p.bary_u = p.bary_v = 0.0f;
    return p;
}

// ---------- Worker ----------
struct WorkerCtx {
    int socket, pin_cpu;
    BVH* bvh;
    Queue* queues;
    int n_queues;
    Mode mode;
    float* fb;
    int img_w, img_h;
    int samples;
    std::atomic<int64_t>* in_flight;
    std::atomic<bool>* shutdown;
    std::atomic<uint64_t>* rays_processed;
    std::atomic<uint64_t>* rays_forwarded;
    std::atomic<uint64_t>* paths_retired;
    const ScenePalette* palette;
    int rng_seed; int worker_id;
    // Texture pool overlay for bandwidth-bound stress (cp10):
    //   tex_pool points to a buffer of n_tex_floats floats (n_tex_floats * 4 bytes).
    //   tex_pool is allocated per-mode: numa_alloc_interleaved (N/H/P) or
    //   numa_alloc_onnode (R, one copy per socket).
    //   On hit, each worker reads K=tex_lookups texels at hashed offsets,
    //   accumulating into the albedo. tex_lookups=0 disables textures
    //   (binary behaves identically to cp9).
    const float* tex_pool;
    size_t       n_tex_floats;
    int          tex_lookups;
    bool         compute_light;  // cp12: terminate on hit (no scatter/bounce)
};

static inline void retire(const Path& p, const Vec3& contrib, float* fb, int img_w, int img_h,
                          std::atomic<uint64_t>* paths_retired, std::atomic<int64_t>* in_flight) {
    int x = p.pixel_id % img_w;
    int y = p.pixel_id / img_w;
    if (x >= 0 && x < img_w && y >= 0 && y < img_h) {
        float* pix = fb + (y * img_w + x) * 3;
        pix[0] += p.throughput.x * contrib.x;
        pix[1] += p.throughput.y * contrib.y;
        pix[2] += p.throughput.z * contrib.z;
    }
    paths_retired->fetch_add(1, std::memory_order_relaxed);
    in_flight->fetch_sub(1, std::memory_order_acq_rel);
}

static uint16_t initial_mask(const Path& p, const std::vector<BVH>& bvhs,
                             int n_sockets, Mode mode, int rr, int& first) {
    uint16_t mask = 0; first = -1;
    if (mode == Mode::Replicate || mode == Mode::Naive || mode == Mode::Hybrid) {
        float tn, tf;
        if (bvhs[0].world_bbox.intersect(p.o, p.invd, tn, tf) && tf > 0) {
            int sk = rr % n_sockets;
            mask = (uint16_t)(1u << sk); first = sk;
        }
    } else {
        float best = 1e30f;
        for (int sk = 0; sk < n_sockets; ++sk) {
            float tn, tf;
            if (bvhs[sk].world_bbox.intersect(p.o, p.invd, tn, tf) && tf > 0) {
                mask |= (uint16_t)(1u << sk);
                if (tn < best) { best = tn; first = sk; }
            }
        }
    }
    return mask;
}

static void push_batch(Queue& q, std::vector<Path>& batch) {
    if (batch.empty()) return;
    size_t n = batch.size();
    if (q.local) {
        // Spinlock append into the destination-local buffer (the forwarded
        // batch lands on the consumer socket's DRAM). No condvar.
        q.slock();
        q.paths.insert(q.paths.end(),
            std::make_move_iterator(batch.begin()),
            std::make_move_iterator(batch.end()));
        q.total_pushed.fetch_add(n);
        q.sunlock();
    } else {
        {
            std::lock_guard<std::mutex> lk(q.m);
            q.paths.insert(q.paths.end(),
                std::make_move_iterator(batch.begin()),
                std::make_move_iterator(batch.end()));
            q.total_pushed.fetch_add(n);
        }
        q.cv.notify_all();
    }
    batch.clear();
}

static void worker_main(WorkerCtx ctx, BVH* all_bvhs, int n_bvh) {
    pin_thread_to_cpu(ctx.pin_cpu);
    Rng rng;
    rng.seed(0x9E3779B97F4A7C15ULL ^ (uint64_t)ctx.worker_id * 0xC6BC279692B5C323ULL,
             0xBF58476D1CE4E5B9ULL ^ (uint64_t)ctx.rng_seed);
    std::vector<Path> batch;
    std::vector<Path> fwd[8];

    while (!ctx.shutdown->load(std::memory_order_acquire) || ctx.in_flight->load() > 0) {
        batch.clear();
        Queue& Q = ctx.queues[ctx.socket];
        if (Q.local) {
            // Spinlock pop (no condvar). If empty, spin-poll with pause and
            // re-check termination at the top of the while loop.
            Q.slock();
            if (!Q.paths.empty()) {
                size_t take = std::min(Q.paths.size(), (size_t)512);
                batch.insert(batch.end(),
                    std::make_move_iterator(Q.paths.end() - take),
                    std::make_move_iterator(Q.paths.end()));
                Q.paths.resize(Q.paths.size() - take);
                Q.total_popped.fetch_add(take);
            }
            Q.sunlock();
            if (batch.empty()) { for (volatile int i = 0; i < 128; ++i) __builtin_ia32_pause(); continue; }
        } else {
            std::unique_lock<std::mutex> lk(Q.m);
            Q.cv.wait_for(lk, std::chrono::milliseconds(2),
                [&]{ return !Q.paths.empty() || ctx.shutdown->load(); });
            if (Q.paths.empty()) continue;
            size_t take = std::min(Q.paths.size(), (size_t)512);
            batch.insert(batch.end(),
                std::make_move_iterator(Q.paths.end() - take),
                std::make_move_iterator(Q.paths.end()));
            Q.paths.resize(Q.paths.size() - take);
            Q.total_popped.fetch_add(take);
        }
        for (int s = 0; s < ctx.n_queues; ++s) fwd[s].clear();
        uint64_t proc = 0, fwdc = 0;

        for (auto& p : batch) {
            traverse(*ctx.bvh, p);
            p.sockets_pending &= ~(uint16_t)(1u << ctx.socket);
            proc++;
            if (p.sockets_pending != 0) {
                int next = __builtin_ctz(p.sockets_pending);
                if (next < ctx.n_queues) { fwd[next].push_back(p); fwdc++; }
                else { retire(p, sky_color(p.d), ctx.fb, ctx.img_w, ctx.img_h, ctx.paths_retired, ctx.in_flight); }
                continue;
            }
            if (p.hit_tri_id < 0) {
                retire(p, sky_color(p.d), ctx.fb, ctx.img_w, ctx.img_h, ctx.paths_retired, ctx.in_flight);
                continue;
            }
            int side = (int)(((uint32_t)p.hit_tri_id) >> 28);
            uint32_t ti = ((uint32_t)p.hit_tri_id) & 0x0FFFFFFFu;
            if (side < 0 || side >= n_bvh || ti >= all_bvhs[side].n_prims) {
                retire(p, Vec3{1,0,1}, ctx.fb, ctx.img_w, ctx.img_h, ctx.paths_retired, ctx.in_flight);
                continue;
            }
            const Sphere& s = all_bvhs[side].prims[ti];
            Vec3 hp = p.o + p.d * p.t_max;
            Vec3 outward = (hp - s.center) / s.radius;   // analytic sphere normal
            Vec3 n = outward;
            Vec3 albedo = s.albedo;
            // Optional texture overlay (kept for parity; off by default for spheres).
            if (ctx.tex_lookups > 0 && ctx.n_tex_floats > 0) {
                uint32_t h = (uint32_t)(ti * 0x9E3779B9u);
                Vec3 tex_acc{0,0,0};
                for (int i = 0; i < ctx.tex_lookups; ++i) {
                    h = h * 0x6C8E9CF5u + 1u;
                    size_t off = (size_t)h % (ctx.n_tex_floats - 4);
                    tex_acc.x += ctx.tex_pool[off + 0];
                    tex_acc.y += ctx.tex_pool[off + 1];
                    tex_acc.z += ctx.tex_pool[off + 2];
                }
                float inv = 1.0f / (float)ctx.tex_lookups;
                albedo.x *= tex_acc.x*inv; albedo.y *= tex_acc.y*inv; albedo.z *= tex_acc.z*inv;
            }
            if (p.depth >= p.max_depth) {
                retire(p, Vec3{0,0,0}, ctx.fb, ctx.img_w, ctx.img_h, ctx.paths_retired, ctx.in_flight);
                continue;
            }
            // compute-light: terminate on hit with shaded albedo (no scatter).
            if (ctx.compute_light) {
                float shade = std::fabs(n.dot(p.d));
                retire(p, albedo * shade, ctx.fb, ctx.img_w, ctx.img_h,
                       ctx.paths_retired, ctx.in_flight);
                continue;
            }
            // ---- RTIOW material scatter ----
            Vec3 dir; Vec3 atten;
            Vec3 ud = p.d.normalized();
            if (s.material == MAT_LAMBERTIAN) {
                if (n.dot(p.d) > 0) n = Vec3{-n.x,-n.y,-n.z};
                dir = n + rng.random_unit(); if (near_zero(dir)) dir = n; dir = dir.normalized();
                atten = albedo;
            } else if (s.material == MAT_METAL) {
                if (n.dot(p.d) > 0) n = Vec3{-n.x,-n.y,-n.z};
                Vec3 refl = reflectv(ud, n);
                dir = (refl + rng.unit_in_sphere() * s.fuzz).normalized();
                atten = albedo;
                if (dir.dot(n) <= 0) { retire(p, Vec3{0,0,0}, ctx.fb, ctx.img_w, ctx.img_h, ctx.paths_retired, ctx.in_flight); continue; }
            } else { // DIELECTRIC
                bool front = ud.dot(outward) < 0;
                Vec3 nn = front ? outward : Vec3{-outward.x,-outward.y,-outward.z};
                float ri = front ? (1.0f/s.ir) : s.ir;
                float ct = std::min(-ud.dot(nn), 1.0f);
                float st = std::sqrt(std::max(0.0f, 1.0f - ct*ct));
                if (ri*st > 1.0f || schlick(ct, ri) > rng.next_f()) dir = reflectv(ud, nn);
                else dir = refractv(ud, nn, ri);
                dir = dir.normalized();
                atten = Vec3{1,1,1};
            }
            p.throughput = p.throughput * atten;
            p.depth++;
            Vec3 dir2 = dir; (void)dir2;
            p.o = hp; p.d = dir;
            p.invd = {1.0f/dir.x, 1.0f/dir.y, 1.0f/dir.z};
            p.t_max = 1e30f; p.hit_tri_id = -1;
            int first = -1;
            std::vector<BVH> view(all_bvhs, all_bvhs + n_bvh);
            uint16_t mask = initial_mask(p, view, ctx.n_queues, ctx.mode, (int)proc, first);
            p.sockets_pending = mask;
            if (mask == 0) {
                retire(p, sky_color(p.d), ctx.fb, ctx.img_w, ctx.img_h, ctx.paths_retired, ctx.in_flight);
            } else if (first < 0 || first >= ctx.n_queues) {
                retire(p, sky_color(p.d), ctx.fb, ctx.img_w, ctx.img_h, ctx.paths_retired, ctx.in_flight);
            } else {
                fwd[first].push_back(p); fwdc++;
            }
        }
        ctx.rays_processed->fetch_add(proc);
        ctx.rays_forwarded->fetch_add(fwdc);
        for (int s = 0; s < ctx.n_queues; ++s) if (!fwd[s].empty()) push_batch(ctx.queues[s], fwd[s]);
    }
}

// ---------- main ----------
int main(int argc, char** argv) {
    if (numa_available() < 0) { fprintf(stderr, "NUMA unavailable\n"); return 2; }
    std::string out_file = (argc > 1) ? argv[1] : "rtiow_hairball.ppm";

    int grid_half = env_int("RTIOW_GRID_HALF", 500);   // RTIOW sphere grid half-extent
    int img_w     = env_int("RTIOW_IMAGE_WIDTH", 1200);
    int samples   = env_int("RTIOW_SAMPLES", 1);
    int max_depth = env_int("RTIOW_MAX_DEPTH", 5);
    int seed      = env_int("RTIOW_SEED", 42);
    int max_nodes = numa_num_configured_nodes();
    // cp13: build logical-socket -> physical-node map from the cpuset. Under
    // `numactl --cpunodebind=A,B` only nodes A,B have cpus, so we use those
    // as logical sockets 0,1 (selecting the cross-socket distance via the
    // pair). Without numactl, this is just [0,1,...] as before.
    g_phys_node = active_numa_nodes();
    int avail_sockets = (int)g_phys_node.size();
    int n_sockets = std::min(avail_sockets, env_int("RTIOW_NSOCKETS", 2));
    if (n_sockets < 1) n_sockets = 1;
    g_phys_node.resize(n_sockets);  // only the first n_sockets are used
    { fprintf(stderr, "cp13: logical sockets -> physical nodes:");
      for (int s = 0; s < n_sockets; ++s) fprintf(stderr, " %d->%d", s, g_phys_node[s]);
      fprintf(stderr, " (of %d configured)\n", max_nodes); }
    int leaf_max = env_int("RTIOW_LEAF_MAX", 4);
    if (leaf_max < 1) leaf_max = 1;
    if (leaf_max > 255) leaf_max = 255;
    Mode mode = parse_mode(env_str("RTIOW_MODE", "partition"));
    int repl_depth = env_int("RTIOW_REPL_DEPTH", 16);
    bool profile_bvh = env_int("RTIOW_PROFILE_BVH", 0) != 0;
    int tex_mb = env_int("RTIOW_TEX_MB", 0);              // texture pool size in MB
    int tex_lookups = env_int("RTIOW_TEX_LOOKUPS", 4);    // texels read per hit
    // cp11: per-socket memory budget. 0 = unlimited. If a mode's predicted
    // per-socket footprint exceeds this, the run is declared INFEASIBLE and
    // exits before building (models a NUMA node with limited local DRAM, or
    // a cgroup cpuset.mems + memory cap).
    long socket_budget_mb = (long)env_int("RTIOW_SOCKET_BUDGET_MB", 0);
    if (tex_mb < 0) tex_mb = 0;
    if (tex_lookups < 0) tex_lookups = 0;
    if (tex_mb == 0) tex_lookups = 0;  // no pool -> no lookups
    // cp12: cap total worker threads (0 = use all cores in cpuset), spread
    // evenly across sockets so all memory controllers stay active. Lets us
    // sweep thread count {32,64,128,256} on one allocation to find the
    // fabric-saturation point where replication starts to beat interleave.
    int max_threads = env_int("RTIOW_NTHREADS", 0);
    // cp18: queue mechanism. "mutex" = heavyweight (mutex+condvar, default
    // alloc). "local" = spinlock + spin-poll + destination-NUMA-local storage
    // (Wald-spirit bulk/local handoff). Lets us measure how much of
    // partition's cost is fixable mechanism vs fundamental fabric.
    const char* queue_mode = env_str("RTIOW_QUEUE", "mutex");
    bool use_local_queue = (strcmp(queue_mode, "local") == 0);
    // cp12: compute-light shading -- on hit, accumulate albedo (x texture)
    // and terminate the path (no scatter, no bounce). Maximizes the
    // memory:compute ratio to push toward bandwidth-bound.
    bool compute_light = env_int("RTIOW_COMPUTE_LIGHT", 0) != 0;

    fprintf(stderr, "rtiow_spheres: mode=%s grid_half=%d "
                    "samples=%d depth=%d leaf_max=%d ns=%d (of %d) "
                    "repl_depth=%d profile=%d tex_MB=%d tex_lookups=%d\n",
            mode_name(mode), grid_half, samples, max_depth,
            leaf_max, n_sockets, max_nodes, repl_depth, profile_bvh ? 1 : 0,
            tex_mb, tex_lookups);

    auto t0 = std::chrono::high_resolution_clock::now();

    // Scene gen
    auto tg0 = std::chrono::high_resolution_clock::now();
    ScenePalette palette;  // unused (kept for ctx parity)
    std::vector<Sphere> tris;  // named 'tris' to minimize edits; holds spheres
    gen_spheres(grid_half, (uint32_t)seed, tris);
    auto tg1 = std::chrono::high_resolution_clock::now();
    size_t n_tris = tris.size(), n_verts = 0;
    fprintf(stderr, "scene: spheres=%zu\n", n_tris);

    // ---- cp11: per-socket footprint accounting + feasibility gate ----
    // Predict per-socket peak bytes for this mode before building anything.
    //   G        = one full copy of geometry (tris + verts + BVH nodes)
    //   est_nodes ~= 2 * ceil(n_tris / leaf_max)   (binary BVH, leaf_max prims/leaf)
    //   tex_b    = texture pool bytes
    // Per-socket footprint by mode:
    //   Naive     : G is ONE interleaved copy -> ~G/ns per node; tex interleaved -> tex_b/ns
    //   Partition : tris+nodes split (~ (tri_b+node_b)/ns) BUT full vertex array per
    //               socket (our builder replicates verts); tex interleaved -> tex_b/ns
    //   Replicate : full G per node + full tex_b per node
    //   Hybrid    : G/ns (interleaved) + hot-prefix bytes; tex interleaved -> tex_b/ns
    {
        size_t tri_b  = n_tris  * sizeof(Sphere);
        size_t vert_b = 0;  // spheres: no vertex array
        size_t est_nodes = 2 * (n_tris / (size_t)std::max(1, leaf_max) + 1);
        size_t node_b = est_nodes * sizeof(BVHNode2);
        size_t G = tri_b + vert_b + node_b;
        size_t tex_b = (size_t)tex_mb * 1024ULL * 1024ULL;
        size_t ns = (size_t)n_sockets;
        // hot-prefix bytes for hybrid (top-K levels, capped at full tree)
        size_t hot_nodes = 0;
        if (mode == Mode::Hybrid) {
            if (repl_depth >= 30) hot_nodes = est_nodes;
            else if (repl_depth > 0) hot_nodes = std::min((size_t)((1ull<<repl_depth)-1), est_nodes);
        }
        size_t hot_b = hot_nodes * sizeof(BVHNode2);

        size_t per_socket = 0;
        switch (mode) {
            case Mode::Naive:
                per_socket = G/ns + tex_b/ns; break;
            case Mode::Partition:
                per_socket = (tri_b + node_b)/ns + vert_b + tex_b/ns; break;
            case Mode::Replicate:
                per_socket = G + tex_b; break;
            case Mode::Hybrid:
                per_socket = G/ns + hot_b + tex_b/ns; break;
        }
        double G_mb = G / 1048576.0;
        double ps_mb = per_socket / 1048576.0;
        fprintf(stderr, "footprint: geometry_G=%.1fMB tex=%dMB per_socket_pred=%.1fMB "
                        "budget=%ldMB\n", G_mb, tex_mb, ps_mb, socket_budget_mb);

        if (socket_budget_mb > 0 && ps_mb > (double)socket_budget_mb) {
            // Infeasible: emit a parseable line and a sentinel BENCH_STATS,
            // then exit cleanly so the sweep continues.
            printf("FEASIBILITY mode=%s feasible=0 n_triangles=%zu geometry_MB=%.1f "
                   "per_socket_pred_MB=%.1f budget_MB=%ld n_sockets=%d tex_MB=%d\n",
                   mode_name(mode), n_tris, G_mb, ps_mb, socket_budget_mb, n_sockets, tex_mb);
            printf("BENCH_STATS tag=rtiow_hairball mode=%s feasible=0 leaf_max=%d "
                   "n_triangles=%zu n_sockets=%d tex_MB=%d per_socket_pred_MB=%.1f "
                   "budget_MB=%ld render_ms=-1 total_ms=-1\n",
                   mode_name(mode), leaf_max, n_tris, n_sockets, tex_mb, ps_mb,
                   socket_budget_mb);
            return 0;
        }
        printf("FEASIBILITY mode=%s feasible=1 n_triangles=%zu geometry_MB=%.1f "
               "per_socket_pred_MB=%.1f budget_MB=%ld n_sockets=%d tex_MB=%d\n",
               mode_name(mode), n_tris, G_mb, ps_mb, socket_budget_mb, n_sockets, tex_mb);
    }

    // Build BVH(s)
    std::vector<BVH> bvhs(n_sockets);
    auto tb0 = std::chrono::high_resolution_clock::now();
    size_t hybrid_extra_bytes = 0;
    if (mode == Mode::Naive) {
        build_bvh_interleaved(bvhs[0], tris, (uint32_t)leaf_max);
        for (int s = 1; s < n_sockets; ++s) bvhs[s] = bvhs[0];
    } else if (mode == Mode::Hybrid) {
        // Build one shared BVH (interleaved), BFS-reorder so top K levels
        // are the first prefix, then allocate per-socket hot-prefix copies.
        build_bvh_interleaved(bvhs[0], tris, (uint32_t)leaf_max);
        bfs_reorder_inplace(bvhs[0]);
        size_t hot_count;
        if (repl_depth <= 0) {
            hot_count = 0;
        } else if (repl_depth >= 30) {
            hot_count = bvhs[0].n_nodes;  // effectively full replicate
        } else {
            uint64_t cap = (1ull << repl_depth) - 1ull;
            hot_count = std::min((size_t)cap, bvhs[0].n_nodes);
        }
        for (int s = 1; s < n_sockets; ++s) bvhs[s] = bvhs[0]; // shallow copy of shared pointers
        hybrid_extra_bytes = alloc_hot_per_socket(bvhs, n_sockets, hot_count);
        fprintf(stderr, "hybrid: repl_depth=%d hot_count=%zu hot_MB_per_socket=%.2f "
                        "total_hot_MB=%.2f\n",
                repl_depth, hot_count,
                hot_count * sizeof(BVHNode2) / 1048576.0,
                hybrid_extra_bytes / 1048576.0);
    } else if (mode == Mode::Replicate) {
        for (int s = 0; s < n_sockets; ++s) build_bvh_on(bvhs[s], tris, phys(s), s, (uint32_t)leaf_max);
    } else {
        // Partition by sphere-center x-median.
        std::vector<float> xs; xs.reserve(tris.size());
        for (const auto& sph : tris) xs.push_back(sph.center.x);
        std::vector<float> sorted_xs = xs;
        std::nth_element(sorted_xs.begin(), sorted_xs.begin() + sorted_xs.size()/2, sorted_xs.end());
        float x_split = sorted_xs[sorted_xs.size()/2];
        std::vector<std::vector<Sphere>> per_sock(n_sockets);
        for (size_t i = 0; i < tris.size(); ++i) {
            int sid = (xs[i] < x_split) ? 0 : 1;
            if (sid >= n_sockets) sid = n_sockets - 1;
            per_sock[sid].push_back(tris[i]);
        }
        for (int s = 0; s < n_sockets; ++s) {
            build_bvh_on(bvhs[s], per_sock[s], phys(s), s, (uint32_t)leaf_max);
        }
        fprintf(stderr, "partition: x_split=%.3f", x_split);
        for (int s = 0; s < n_sockets; ++s) fprintf(stderr, " sock%d=%zu sph", s, per_sock[s].size());
        fprintf(stderr, "\n");
    }
    auto tb1 = std::chrono::high_resolution_clock::now();

    // Profile-mode visit counter (shared, lives on socket 0's allocator).
    std::atomic<uint64_t>* visit_array = nullptr;
    if (profile_bvh) {
        visit_array = new std::atomic<uint64_t>[bvhs[0].n_nodes];
        for (size_t i = 0; i < bvhs[0].n_nodes; ++i) visit_array[i].store(0);
        for (int s = 0; s < n_sockets; ++s) bvhs[s].visits = visit_array;
    }

    // -------- Texture pool allocation (cp10 bandwidth overlay) --------
    // Allocated per-socket: interleaved for N/H/P (one shared pool), or
    // numa_alloc_onnode per socket for R (one local copy each). Filled
    // with deterministic pseudo-random floats so cache-line reads return
    // meaningful values rather than zeros (zero-fill is a known cheat
    // some allocators use that would defeat the bandwidth test).
    std::vector<const float*> tex_pool_per_socket(n_sockets, nullptr);
    size_t n_tex_floats = (size_t)tex_mb * 1024ULL * 1024ULL / sizeof(float);
    if (tex_mb > 0) {
        size_t tex_bytes = (size_t)tex_mb * 1024ULL * 1024ULL;
        if (mode == Mode::Replicate) {
            // One copy per socket, allocated locally
            for (int s = 0; s < n_sockets; ++s) {
                float* p = (float*)numa_alloc_onnode(tex_bytes, phys(s));
                if (!p) { fprintf(stderr, "tex_pool numa_alloc_onnode(%zu, %d) failed\n", tex_bytes, s); return 1; }
                // First-touch from a thread on socket s
                std::thread ft([&, s, p](){
                    auto cs = cpus_in_node(phys(s));
                    if (!cs.empty()) pin_thread_to_cpu(cs[0]);
                    for (size_t i = 0; i < n_tex_floats; ++i) {
                        uint32_t h = (uint32_t)(i * 0x9E3779B9u) ^ (uint32_t)s;
                        p[i] = (h & 0xFFFFFF) * (1.0f / 16777216.0f); // [0,1)
                    }
                });
                ft.join();
                tex_pool_per_socket[s] = p;
            }
            fprintf(stderr, "tex_pool: REPLICATED, %d MB per socket, %d sockets -> %d MB total\n",
                    tex_mb, n_sockets, tex_mb * n_sockets);
        } else {
            // Single interleaved pool, shared across all sockets
            float* p = (float*)numa_alloc_interleaved(tex_bytes);
            if (!p) { fprintf(stderr, "tex_pool numa_alloc_interleaved(%zu) failed\n", tex_bytes); return 1; }
            for (size_t i = 0; i < n_tex_floats; ++i) {
                uint32_t h = (uint32_t)(i * 0x9E3779B9u);
                p[i] = (h & 0xFFFFFF) * (1.0f / 16777216.0f);
            }
            for (int s = 0; s < n_sockets; ++s) tex_pool_per_socket[s] = p;
            fprintf(stderr, "tex_pool: INTERLEAVED, %d MB shared across %d sockets\n", tex_mb, n_sockets);
        }
    }

    // Camera + framebuffer + initial paths
    Camera cam = make_camera(img_w);
    int img_h = cam.height;
    std::vector<float> fb((size_t)img_w * img_h * 3, 0.0f);

    std::vector<Queue> queues(n_sockets);
    Rng cam_rng; cam_rng.seed((uint64_t)seed * 0x9E3779B9 + 1, 0x12345678);
    std::vector<std::vector<Path>> initial(n_sockets);
    int64_t total_paths = (int64_t)img_w * img_h * samples;
    uint64_t init_none = 0;
    auto tgr0 = std::chrono::high_resolution_clock::now();
    int rr = 0;
    for (int y = 0; y < img_h; ++y) {
        for (int x = 0; x < img_w; ++x) {
            for (int s = 0; s < samples; ++s) {
                Path p = make_primary(cam, x, y, y*img_w + x, (uint8_t)max_depth, cam_rng);
                int first = -1;
                uint16_t mask = initial_mask(p, bvhs, n_sockets, mode, rr++, first);
                p.sockets_pending = mask;
                if (mask == 0) {
                    int px = p.pixel_id % img_w, py = p.pixel_id / img_w;
                    Vec3 sk = sky_color(p.d);
                    float* pix = fb.data() + (py * img_w + px) * 3;
                    pix[0] += sk.x; pix[1] += sk.y; pix[2] += sk.z;
                    init_none++;
                } else {
                    initial[first].push_back(p);
                }
            }
        }
    }
    auto tgr1 = std::chrono::high_resolution_clock::now();
    int64_t paths_enq = 0;
    for (int s = 0; s < n_sockets; ++s) paths_enq += (int64_t)initial[s].size();
    fprintf(stderr, "primary paths enqueued=%lld sky_immediate=%llu (of %lld)\n",
            (long long)paths_enq, (unsigned long long)init_none, (long long)total_paths);

    std::atomic<int64_t> in_flight(paths_enq);
    std::atomic<bool> shutdown(false);
    std::vector<std::atomic<uint64_t>> rays_proc(n_sockets), rays_fwd(n_sockets), paths_ret(n_sockets);
    for (int s = 0; s < n_sockets; ++s) { rays_proc[s].store(0); rays_fwd[s].store(0); paths_ret[s].store(0); }
    if (use_local_queue) {
        // Reserve each queue's storage and first-touch it onto the socket's
        // local node (page placement = first-touch policy), so the queue data
        // lives in the consumer socket's DRAM. Reserve >= worst-case in-flight
        // (total paths) so no later realloc moves the pages off-node. Then
        // COPY initial paths in (a move-assign would swap in node-0 storage
        // and destroy the local placement).
        size_t cap = (size_t)paths_enq + 4096;
        for (int s = 0; s < n_sockets; ++s) {
            queues[s].local = true;
            queues[s].paths.reserve(cap);
            int nodeS = phys(s);
            std::thread ft([&queues, s, nodeS, cap](){
                auto cs = cpus_in_node(nodeS);
                if (!cs.empty()) pin_thread_to_cpu(cs[0]);
                // First-touch the full reserved buffer on node s (one write per
                // 4KB page binds the page to this node under first-touch policy).
                volatile char* base = (volatile char*)queues[s].paths.data();
                size_t bytes = cap * sizeof(Path);
                for (size_t off = 0; off < bytes; off += 4096) base[off] = 0;
            });
            ft.join();
        }
        for (int s = 0; s < n_sockets; ++s) {
            queues[s].slock();
            queues[s].paths.insert(queues[s].paths.end(), initial[s].begin(), initial[s].end());
            queues[s].total_pushed.fetch_add(queues[s].paths.size());
            queues[s].sunlock();
            initial[s].clear();
        }
        fprintf(stderr, "queue=local: reserved+first-touched %zu paths/socket on local DRAM\n", cap);
    } else {
        for (int s = 0; s < n_sockets; ++s) {
            std::lock_guard<std::mutex> lk(queues[s].m);
            queues[s].paths = std::move(initial[s]);
            queues[s].total_pushed.fetch_add(queues[s].paths.size());
        }
    }

    auto tr0 = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> workers;
    int worker_id = 0;
    // cp12: thread cap. per_sock_cap = ceil(max_threads / n_sockets) so the
    // chosen threads are spread evenly across sockets (keeps all memory
    // controllers active). max_threads<=0 -> use all cores in cpuset.
    int per_sock_cap = 0;
    if (max_threads > 0) per_sock_cap = (max_threads + n_sockets - 1) / n_sockets;
    int spawned_total = 0;
    for (int s = 0; s < n_sockets; ++s) {
        auto cs = cpus_in_node(phys(s));
        int spawned_this_sock = 0;
        for (int c : cs) {
            if (max_threads > 0 && (spawned_total >= max_threads ||
                                    spawned_this_sock >= per_sock_cap)) break;
            WorkerCtx ctx;
            ctx.socket = s; ctx.pin_cpu = c; ctx.bvh = &bvhs[s];
            ctx.queues = queues.data(); ctx.n_queues = n_sockets; ctx.mode = mode;
            ctx.fb = fb.data(); ctx.img_w = img_w; ctx.img_h = img_h;
            ctx.samples = samples; ctx.in_flight = &in_flight; ctx.shutdown = &shutdown;
            ctx.rays_processed = &rays_proc[s]; ctx.rays_forwarded = &rays_fwd[s];
            ctx.paths_retired = &paths_ret[s];
            ctx.palette = &palette;
            ctx.tex_pool = tex_pool_per_socket[s];
            ctx.n_tex_floats = n_tex_floats;
            ctx.tex_lookups = tex_lookups;
            ctx.compute_light = compute_light;
            ctx.rng_seed = seed; ctx.worker_id = worker_id++;
            BVH* all = bvhs.data(); int nb = n_sockets;
            workers.emplace_back([ctx, all, nb]() mutable { worker_main(ctx, all, nb); });
            spawned_this_sock++; spawned_total++;
        }
    }
    fprintf(stderr, "spawned %d workers across %d sockets (cap=%d)\n",
            spawned_total, n_sockets, max_threads);

    while (in_flight.load() > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        for (auto& q : queues) q.cv.notify_all();
    }
    shutdown.store(true);
    for (auto& q : queues) q.cv.notify_all();
    for (auto& t : workers) t.join();
    auto tr1 = std::chrono::high_resolution_clock::now();

    // Output
    auto tw0 = std::chrono::high_resolution_clock::now();
    {
        std::vector<uint8_t> out((size_t)img_w * img_h * 3);
        float inv_s = 1.0f / (float)samples;
        for (int i = 0; i < img_w * img_h; ++i) {
            float r = std::sqrt(std::max(0.0f, fb[i*3+0] * inv_s));
            float g = std::sqrt(std::max(0.0f, fb[i*3+1] * inv_s));
            float b = std::sqrt(std::max(0.0f, fb[i*3+2] * inv_s));
            out[i*3+0] = (uint8_t)(255.0f * std::min(1.0f, r));
            out[i*3+1] = (uint8_t)(255.0f * std::min(1.0f, g));
            out[i*3+2] = (uint8_t)(255.0f * std::min(1.0f, b));
        }
        std::ofstream f(out_file, std::ios::binary);
        f << "P6\n" << img_w << " " << img_h << "\n255\n";
        f.write((const char*)out.data(), out.size());
    }
    auto tw1 = std::chrono::high_resolution_clock::now();

    auto ms = [](auto a, auto b){ return std::chrono::duration_cast<std::chrono::milliseconds>(b-a).count(); };
    long gen_ms = ms(tg0, tg1), bvh_ms = ms(tb0, tb1), genray_ms = ms(tgr0, tgr1);
    long render_ms = ms(tr0, tr1), write_ms = ms(tw0, tw1), total_ms = ms(t0, tw1);

    uint64_t total_proc = 0, total_fwd = 0, total_ret = 0;
    for (int s = 0; s < n_sockets; ++s) {
        total_proc += rays_proc[s].load();
        total_fwd  += rays_fwd[s].load();
        total_ret  += paths_ret[s].load();
    }
    double fwd_rate = (paths_enq > 0) ? (double)total_fwd / (double)paths_enq : 0.0;
    double avg_visits = (paths_enq > 0) ? (double)total_proc / (double)paths_enq : 0.0;
    size_t bvh_bytes_per = bvhs[0].n_nodes * sizeof(BVHNode2)
                        + bvhs[0].n_prims * sizeof(Sphere);

    printf("BENCH_STATS tag=rtiow_spheres mode=%s leaf_max=%d "
           "grid_half=%d n_spheres=%zu n_vertices=%zu "
           "samples=%d max_depth=%d seed=%d n_sockets=%d image_width=%d image_height=%d "
           "bvh_nodes_sock0=%zu bvh_bytes_per_socket=%zu hot_count=%zu "
           "hybrid_extra_MB=%.2f repl_depth=%d tex_MB=%d tex_lookups=%d "
           "tex_replicated=%d feasible=1 budget_MB=%ld queue=%s "
           "actual_per_socket_MB=%.1f "
           "paths_enqueued=%lld rays_processed=%llu rays_forwarded=%llu "
           "paths_retired=%llu forwarding_rate=%.4f avg_visits_per_path=%.3f "
           "gen_ms=%ld bvh_build_ms=%ld genray_ms=%ld render_ms=%ld write_ms=%ld total_ms=%ld\n",
           mode_name(mode), leaf_max,
           grid_half, n_tris, n_verts,
           samples, max_depth, seed, n_sockets, img_w, img_h,
           bvhs[0].n_nodes, bvh_bytes_per, bvhs[0].hot_count,
           hybrid_extra_bytes / 1048576.0,
           (mode == Mode::Hybrid) ? repl_depth : 0,
           tex_mb, tex_lookups,
           (tex_mb > 0 && mode == Mode::Replicate) ? 1 : 0,
           socket_budget_mb, queue_mode,
           // actual per-socket footprint: bvh bytes (per socket) + tex (per socket if replicated)
           ( (mode == Mode::Replicate)
               ? (bvh_bytes_per + (size_t)tex_mb*1048576ULL)
               : (bvh_bytes_per + (mode==Mode::Hybrid ? (size_t)(hybrid_extra_bytes/std::max(1,n_sockets)) : 0)) )
             / 1048576.0,
           (long long)paths_enq, (unsigned long long)total_proc, (unsigned long long)total_fwd,
           (unsigned long long)total_ret, fwd_rate, avg_visits,
           gen_ms, bvh_ms, genray_ms, render_ms, write_ms, total_ms);

    // Profile dump: per-BFS-level visit histogram (only meaningful if BVH
    // was bfs_reorder'd, which happens in Hybrid mode; for other modes the
    // "level" mapping by array-index is only approximate).
    if (visit_array) {
        size_t N = bvhs[0].n_nodes;
        std::vector<uint64_t> by_level(64, 0);
        std::vector<uint64_t> count_by_level(64, 0);
        for (size_t i = 0; i < N; ++i) {
            int lvl = 0;
            uint64_t v = i + 1;
            while (v > 1) { v >>= 1; lvl++; }
            count_by_level[lvl]++;
            by_level[lvl] += visit_array[i].load();
        }
        uint64_t total = 0;
        for (auto v : by_level) total += v;
        printf("BVH_PROFILE total_visits=%llu\n", (unsigned long long)total);
        printf("BVH_PROFILE level nodes visits visits_per_node cum_pct_visits cum_pct_nodes\n");
        uint64_t cv = 0, cn = 0;
        for (int l = 0; l < 64; ++l) {
            if (count_by_level[l] == 0) continue;
            cv += by_level[l];
            cn += count_by_level[l];
            double vpn = (double)by_level[l] / std::max<uint64_t>(1, count_by_level[l]);
            double pct = total > 0 ? 100.0 * (double)cv / (double)total : 0.0;
            double npct = N > 0 ? 100.0 * (double)cn / (double)N : 0.0;
            printf("BVH_PROFILE %2d %10llu %15llu %10.1f %8.3f %8.5f\n",
                   l, (unsigned long long)count_by_level[l],
                   (unsigned long long)by_level[l], vpn, pct, npct);
        }
        delete[] visit_array;
    }

    int free_count = (mode == Mode::Naive || mode == Mode::Hybrid) ? 1 : n_sockets;
    for (int i = 0; i < free_count; ++i) {
        BVH& b = bvhs[i];
        if (b.prims) numa_free(b.prims, b.n_prims * sizeof(Sphere));
        if (b.nodes) numa_free(b.nodes, b.n_nodes * sizeof(BVHNode2));
    }
    if (mode == Mode::Hybrid) {
        for (int s = 0; s < n_sockets; ++s) {
            if (bvhs[s].hot_nodes) numa_free(bvhs[s].hot_nodes,
                                             bvhs[s].hot_count * sizeof(BVHNode2));
        }
    }
    // Free texture pool(s)
    if (tex_mb > 0) {
        size_t tex_bytes = (size_t)tex_mb * 1024ULL * 1024ULL;
        if (mode == Mode::Replicate) {
            for (int s = 0; s < n_sockets; ++s)
                if (tex_pool_per_socket[s]) numa_free((void*)tex_pool_per_socket[s], tex_bytes);
        } else if (tex_pool_per_socket[0]) {
            numa_free((void*)tex_pool_per_socket[0], tex_bytes);
        }
    }
    return 0;
}
