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

// ---------- geometry ----------
struct Vertex { Vec3 pos; Vec3 normal; };  // 24 bytes

struct Triangle {
    uint32_t v[3];        // 12  -- indices into Vertex array
    uint32_t albedo_idx;  // 4   -- packed 0xRRGGBB00 for compactness, or strand id
};
// 16 bytes (one quarter cache line; 4 tris per CL)

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
    void expand_tri(const Triangle& t, const Vertex* verts) {
        expand(verts[t.v[0]].pos);
        expand(verts[t.v[1]].pos);
        expand(verts[t.v[2]].pos);
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
static double env_dbl(const char* k, double def) {
    const char* s = std::getenv(k); return (!s || !*s) ? def : atof(s);
}

enum class Mode { Partition, Replicate, Naive, Hybrid, LeafHybrid };
static Mode parse_mode(const char* s) {
    if (s && !strcmp(s, "replicate"))  return Mode::Replicate;
    if (s && !strcmp(s, "naive"))      return Mode::Naive;
    if (s && !strcmp(s, "hybrid"))     return Mode::Hybrid;
    if (s && !strcmp(s, "leafhybrid")) return Mode::LeafHybrid;
    return Mode::Partition;
}
static const char* mode_name(Mode m) {
    switch (m) {
        case Mode::Replicate:  return "replicate";
        case Mode::Naive:      return "naive";
        case Mode::Hybrid:     return "hybrid";
        case Mode::LeafHybrid: return "leafhybrid";
        default:               return "partition";
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
struct ScenePalette {
    std::vector<Vec3> colors;  // looked up by triangle.albedo_idx (one entry per strand)
};

static void gen_hairball(int n_strands, int segs, int sides, uint32_t seed,
                         std::vector<Vertex>& verts, std::vector<Triangle>& tris,
                         ScenePalette& palette) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::normal_distribution<float> turn_dist(0.0f, 0.25f);
    const float CORE_R = 0.5f;
    const float TUBE_R = 0.012f;
    const float STEP = 0.04f;

    verts.reserve((size_t)n_strands * (segs + 1) * sides);
    tris.reserve((size_t)n_strands * segs * sides * 2);
    palette.colors.reserve(n_strands);

    for (int s = 0; s < n_strands; ++s) {
        // Start uniformly on small sphere surface
        float th = u(rng) * 6.2831853f;
        float cp = 2.0f * u(rng) - 1.0f;
        float sp = std::sqrt(1.0f - cp * cp);
        Vec3 pos{CORE_R * sp * std::cos(th), CORE_R * sp * std::sin(th), CORE_R * cp};
        Vec3 dir = pos.normalized();
        Vec3 up{0, 1, 0};
        if (std::fabs(dir.dot(up)) > 0.95f) up = Vec3{1, 0, 0};
        Vec3 right = dir.cross(up).normalized();
        up = right.cross(dir).normalized();

        Vec3 albedo{0.3f + 0.6f * u(rng), 0.3f + 0.6f * u(rng), 0.3f + 0.6f * u(rng)};
        palette.colors.push_back(albedo);
        uint32_t strand_id = (uint32_t)s;
        uint32_t base = (uint32_t)verts.size();

        for (int i = 0; i <= segs; ++i) {
            for (int j = 0; j < sides; ++j) {
                float a = j * (6.2831853f / sides);
                float ca = std::cos(a), sa = std::sin(a);
                Vec3 off = right * (TUBE_R * ca) + up * (TUBE_R * sa);
                Vec3 vp = pos + off;
                Vec3 vn = off.normalized();
                verts.push_back({vp, vn});
            }
            if (i < segs) {
                Vec3 turn{turn_dist(rng), turn_dist(rng), turn_dist(rng)};
                dir = (dir + turn * 0.3f).normalized();
                pos = pos + dir * STEP;
                if (std::fabs(dir.dot(up)) > 0.95f) up = right;
                right = dir.cross(up).normalized();
                up = right.cross(dir).normalized();
            }
        }
        // tube quads -> 2 triangles each
        for (int i = 0; i < segs; ++i) {
            for (int j = 0; j < sides; ++j) {
                int jn = (j + 1) % sides;
                uint32_t i0 = base + (uint32_t)(i * sides + j);
                uint32_t i1 = base + (uint32_t)(i * sides + jn);
                uint32_t i2 = base + (uint32_t)((i + 1) * sides + j);
                uint32_t i3 = base + (uint32_t)((i + 1) * sides + jn);
                tris.push_back({{i0, i2, i1}, strand_id});
                tris.push_back({{i1, i2, i3}, strand_id});
            }
        }
    }
}

// ---------- Triangle intersection (Moller-Trumbore) ----------
// verts = shared vertex array (ORIGINAL coherent order, never reordered).
// leafhybrid (v2): vslot[i]>=0 -> vertex i is replicated; read it from the
// per-socket local packed array hotv at that slot. Slots are assigned in
// ASCENDING original index, so a triangle's near-contiguous vertices map to
// near-contiguous slots -> cache/prefetch coherence is PRESERVED (the thing
// v1's hotness-reorder destroyed). For all other modes vslot==nullptr so
// vget==verts[i] exactly (bit-identical).
static inline const Vertex& vget(const Vertex* verts, const Vertex* hotv,
                                 const int32_t* vslot, uint32_t i) {
    if (vslot) { int32_t s = vslot[i]; if (s >= 0) return hotv[s]; }
    return verts[i];
}
static inline bool tri_hit(const Triangle& t, const Vertex* verts,
                           const Vertex* hotv, const int32_t* vslot,
                           const Vec3& o, const Vec3& d,
                           float t_min, float t_max,
                           float& t_hit, float& u_out, float& v_out) {
    const Vec3& v0 = vget(verts, hotv, vslot, t.v[0]).pos;
    const Vec3& v1 = vget(verts, hotv, vslot, t.v[1]).pos;
    const Vec3& v2 = vget(verts, hotv, vslot, t.v[2]).pos;
    Vec3 e1 = v1 - v0, e2 = v2 - v0;
    Vec3 p = d.cross(e2);
    float det = e1.dot(p);
    if (det > -1e-7f && det < 1e-7f) return false;
    float inv = 1.0f / det;
    Vec3 tv = o - v0;
    float u = tv.dot(p) * inv;
    if (u < 0.0f || u > 1.0f) return false;
    Vec3 q = tv.cross(e1);
    float v = d.dot(q) * inv;
    if (v < 0.0f || u + v > 1.0f) return false;
    float th = e2.dot(q) * inv;
    if (th < t_min || th > t_max) return false;
    t_hit = th; u_out = u; v_out = v;
    return true;
}

// ---------- BVH builder over triangles ----------
struct BVH {
    BVHNode2* nodes = nullptr;           // shared/cold (hybrid: still contains all nodes,
                                         //   but workers prefer hot_nodes for idx < hot_count)
                                         //   or full per-socket (R) or interleaved (N) or half (P)
    BVHNode2* hot_nodes = nullptr;       // hybrid: per-socket local copy of top hot_count BFS nodes
    size_t    hot_count = 0;             // 0 unless hybrid
    std::atomic<uint64_t>* visits = nullptr; // profile mode: per-node visit counter (in shared idx space)
    Vertex*   hot_verts = nullptr;       // leafhybrid(v2): per-socket local packed copy of the hot vertices
    size_t    hot_vert_count = 0;        // 0 unless leafhybrid (= #hot verts = local array size)
    int32_t*  vert_slot = nullptr;       // leafhybrid(v2): shared map orig-vert-idx -> local slot (-1 if cold)
    Triangle* prims = nullptr;
    Vertex*   verts = nullptr;
    size_t    n_nodes = 0, n_prims = 0, n_verts = 0;
    AABB      world_bbox;
    int       owner_node = -1;
    int       side_id = 0;
};

static uint32_t build_rec(std::vector<BVHNode2>& out, std::vector<Triangle>& tris,
                          const Vertex* verts, uint32_t lo, uint32_t hi, int depth,
                          uint32_t leaf_max) {
    uint32_t idx = (uint32_t)out.size();
    out.emplace_back();
    AABB bb;
    for (uint32_t i = lo; i < hi; ++i) bb.expand_tri(tris[i], verts);
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
    std::nth_element(tris.begin()+lo, tris.begin()+mid, tris.begin()+hi,
        [verts, ax](const Triangle& a, const Triangle& b){
            float ca = (&verts[a.v[0]].pos.x)[ax] + (&verts[a.v[1]].pos.x)[ax] + (&verts[a.v[2]].pos.x)[ax];
            float cb = (&verts[b.v[0]].pos.x)[ax] + (&verts[b.v[1]].pos.x)[ax] + (&verts[b.v[2]].pos.x)[ax];
            return ca < cb;
        });
    uint32_t l = build_rec(out, tris, verts, lo, mid, depth+1, leaf_max);
    uint32_t r = build_rec(out, tris, verts, mid, hi, depth+1, leaf_max);
    out[idx].bbox = bb; out[idx].left = (int32_t)l;
    out[idx].right_or_first = (int32_t)r;
    out[idx].count = 0; out[idx].axis = (uint16_t)ax;
    return idx;
}

static void build_bvh_on(BVH& out, std::vector<Triangle> tris, std::vector<Vertex> verts_in,
                         int node, int side_id, uint32_t leaf_max) {
    out.owner_node = node;
    out.side_id = side_id;
    out.n_prims = tris.size();
    out.n_verts = verts_in.size();
    std::vector<BVHNode2> tmp;
    tmp.reserve(tris.size() * 2);
    if (!tris.empty()) build_rec(tmp, tris, verts_in.data(), 0, (uint32_t)tris.size(), 0, leaf_max);
    out.n_nodes = tmp.size();
    out.prims = numa_new<Triangle>(std::max<size_t>(1, tris.size()), node);
    out.verts = numa_new<Vertex>(std::max<size_t>(1, verts_in.size()), node);
    out.nodes = numa_new<BVHNode2>(std::max<size_t>(1, tmp.size()), node);
    std::thread ft([&]() {
        auto cs = cpus_in_node(node);
        if (!cs.empty()) pin_thread_to_cpu(cs[0]);
        if (!tris.empty())     memcpy(out.prims, tris.data(),    tris.size() * sizeof(Triangle));
        if (!verts_in.empty()) memcpy(out.verts, verts_in.data(),verts_in.size() * sizeof(Vertex));
        if (!tmp.empty())      memcpy(out.nodes, tmp.data(),     tmp.size() * sizeof(BVHNode2));
    });
    ft.join();
    out.world_bbox = tmp.empty() ? AABB{} : tmp[0].bbox;
}

static void build_bvh_interleaved(BVH& out, std::vector<Triangle> tris,
                                  std::vector<Vertex> verts_in, uint32_t leaf_max) {
    out.owner_node = -1;
    out.side_id = 0;
    out.n_prims = tris.size();
    out.n_verts = verts_in.size();
    std::vector<BVHNode2> tmp;
    tmp.reserve(tris.size() * 2);
    if (!tris.empty()) build_rec(tmp, tris, verts_in.data(), 0, (uint32_t)tris.size(), 0, leaf_max);
    out.n_nodes = tmp.size();
    size_t pb = std::max<size_t>(64, tris.size() * sizeof(Triangle));
    size_t vb = std::max<size_t>(64, verts_in.size() * sizeof(Vertex));
    size_t nb = std::max<size_t>(64, tmp.size() * sizeof(BVHNode2));
    out.prims = (Triangle*)numa_alloc_interleaved(pb);
    out.verts = (Vertex*)numa_alloc_interleaved(vb);
    out.nodes = (BVHNode2*)numa_alloc_interleaved(nb);
    if (!out.prims || !out.verts || !out.nodes) { fprintf(stderr, "numa_alloc_interleaved failed\n"); exit(1); }
    if (!tris.empty())     memcpy(out.prims, tris.data(),     tris.size() * sizeof(Triangle));
    if (!verts_in.empty()) memcpy(out.verts, verts_in.data(), verts_in.size() * sizeof(Vertex));
    if (!tmp.empty())      memcpy(out.nodes, tmp.data(),      tmp.size() * sizeof(BVHNode2));
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
                float th, u, v;
                if (tri_hit(bvh.prims[pi], bvh.verts, bvh.hot_verts,
                            bvh.vert_slot, p.o, p.d, 0.001f, t_max, th, u, v)) {
                    t_max = th;
                    best = (int32_t)((uint32_t)pi | ((uint32_t)bvh.side_id << 28));
                    bu = u; bv = v;
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
            const Triangle& t = all_bvhs[side].prims[ti];
            const Vertex* verts = all_bvhs[side].verts;
            const Vertex* hotv  = all_bvhs[side].hot_verts;
            const int32_t* vslot = all_bvhs[side].vert_slot;
            // Barycentric-interpolated normal (read 3 vertex normals) -- routed
            // through the slot map so leafhybrid serves these locally too.
            float w = 1.0f - p.bary_u - p.bary_v;
            Vec3 n = vget(verts, hotv, vslot, t.v[0]).normal * w
                   + vget(verts, hotv, vslot, t.v[1]).normal * p.bary_u
                   + vget(verts, hotv, vslot, t.v[2]).normal * p.bary_v;
            n = n.normalized();
            Vec3 hp = p.o + p.d * p.t_max;
            if (p.depth >= p.max_depth) {
                retire(p, Vec3{0,0,0}, ctx.fb, ctx.img_w, ctx.img_h, ctx.paths_retired, ctx.in_flight);
                continue;
            }
            Vec3 albedo = ctx.palette->colors[t.albedo_idx];
            // -------- Texture-overlay sampling (cp10) --------
            // K random fetches into the texture pool, hashed from
            // (strand_id, barycentric). Each fetch is a near-guaranteed
            // L3 miss when tex_pool is larger than L3. Accumulate into
            // albedo so the work isn't dead-code-eliminated.
            if (ctx.tex_lookups > 0 && ctx.n_tex_floats > 0) {
                uint32_t h = (uint32_t)t.albedo_idx * 0x9E3779B9u;
                h ^= ((uint32_t)(p.bary_u * 1.0e6f)) * 0xC6BC2796u;
                h ^= ((uint32_t)(p.bary_v * 1.0e6f)) * 0xBF584767u;
                Vec3 tex_acc{0,0,0};
                for (int i = 0; i < ctx.tex_lookups; ++i) {
                    h = h * 0x6C8E9CF5u + 1u;
                    // Force step to land on a different cache line each lookup
                    size_t off = (size_t)h % (ctx.n_tex_floats - 4);
                    // Treat each lookup as RGB triple, accumulate
                    tex_acc.x += ctx.tex_pool[off + 0];
                    tex_acc.y += ctx.tex_pool[off + 1];
                    tex_acc.z += ctx.tex_pool[off + 2];
                }
                float inv = 1.0f / (float)ctx.tex_lookups;
                albedo.x *= tex_acc.x * inv;
                albedo.y *= tex_acc.y * inv;
                albedo.z *= tex_acc.z * inv;
            }
            // cp12 compute-light: terminate on hit with shaded albedo, no
            // bounce. Keeps the full per-hit memory cost (triangle + verts +
            // texture fetches above) while removing scatter compute, to
            // maximize the memory:compute ratio.
            if (ctx.compute_light) {
                float shade = std::fabs(n.dot(p.d));  // cheap facing term
                retire(p, albedo * shade, ctx.fb, ctx.img_w, ctx.img_h,
                       ctx.paths_retired, ctx.in_flight);
                continue;
            }
            if (n.dot(p.d) > 0) n = Vec3{-n.x, -n.y, -n.z};
            // Lambertian scatter
            Vec3 dir = n + rng.random_unit();
            if (near_zero(dir)) dir = n;
            dir = dir.normalized();
            p.throughput = p.throughput * albedo;
            p.depth++;
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

    int n_strands = env_int("RTIOW_NSTRANDS", 800);
    int segments  = env_int("RTIOW_SEGMENTS", 120);
    int sides     = env_int("RTIOW_SIDES",    12);
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
    // leafhybrid: fraction of vertices (ranked by primary-visibility hotness)
    // replicated into per-socket local DRAM. 0 -> behaves like naive; 1.0 ->
    // every vertex local (but tris/nodes still interleaved). Sweeping this
    // traces speedup-vs-replicated-MB: does a small hot fraction recover R?
    double leaf_repl_frac = env_dbl("RTIOW_LEAF_REPL_FRAC", 0.10);
    if (leaf_repl_frac < 0.0) leaf_repl_frac = 0.0;
    if (leaf_repl_frac > 1.0) leaf_repl_frac = 1.0;
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

    fprintf(stderr, "rtiow_hairball: mode=%s strands=%d segs=%d sides=%d "
                    "samples=%d depth=%d leaf_max=%d ns=%d (of %d) "
                    "repl_depth=%d profile=%d tex_MB=%d tex_lookups=%d\n",
            mode_name(mode), n_strands, segments, sides, samples, max_depth,
            leaf_max, n_sockets, max_nodes, repl_depth, profile_bvh ? 1 : 0,
            tex_mb, tex_lookups);

    auto t0 = std::chrono::high_resolution_clock::now();

    // Scene gen
    auto tg0 = std::chrono::high_resolution_clock::now();
    std::vector<Vertex> verts;
    std::vector<Triangle> tris;
    ScenePalette palette;
    gen_hairball(n_strands, segments, sides, (uint32_t)seed, verts, tris, palette);
    auto tg1 = std::chrono::high_resolution_clock::now();
    size_t n_tris = tris.size(), n_verts = verts.size();
    fprintf(stderr, "scene: triangles=%zu vertices=%zu palette=%zu\n",
            n_tris, n_verts, palette.colors.size());

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
        size_t tri_b  = n_tris  * sizeof(Triangle);
        size_t vert_b = n_verts * sizeof(Vertex);
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
        // leafhybrid: per-socket hot-vertex prefix bytes
        size_t leaf_hot_b = (mode == Mode::LeafHybrid)
            ? (size_t)(leaf_repl_frac * (double)n_verts) * sizeof(Vertex) : 0;

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
            case Mode::LeafHybrid:
                per_socket = G/ns + leaf_hot_b + tex_b/ns; break;
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
        build_bvh_interleaved(bvhs[0], tris, verts, (uint32_t)leaf_max);
        for (int s = 1; s < n_sockets; ++s) bvhs[s] = bvhs[0];
    } else if (mode == Mode::Hybrid) {
        // Build one shared BVH (interleaved), BFS-reorder so top K levels
        // are the first prefix, then allocate per-socket hot-prefix copies.
        build_bvh_interleaved(bvhs[0], tris, verts, (uint32_t)leaf_max);
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
    } else if (mode == Mode::LeafHybrid) {
        // ---- LEAF-GEOMETRY replication, COHERENCE-PRESERVING (v2). Build the
        // shared interleaved BVH (verts stay in original strand-build order --
        // NO reorder, so prefetch coherence is intact). Profile primary
        // visibility -> per-vertex hotness. Pick the hottest frac of vertices,
        // but pack them into the per-socket local array in ASCENDING ORIGINAL
        // INDEX (a triangle's near-contiguous verts -> near-contiguous slots).
        // A shared slot map vslot[i] routes hot verts to the local copy, cold
        // verts to the shared interleaved array. tris/nodes stay interleaved,
        // so this isolates the leaf-geometry LOCALITY benefit WITHOUT the
        // cache-layout regression that v1's hotness-reorder caused. ----
        build_bvh_interleaved(bvhs[0], tris, verts, (uint32_t)leaf_max);
        size_t NV = bvhs[0].n_verts, NN = bvhs[0].n_nodes;
        std::vector<uint64_t> vhot(NV, 0);
        if (NV > 0 && NN > 0) {
            // profiling pass: count node visits over PRIMARY rays (depth 1),
            // single-threaded; stride huge images to cap cost ~400k rays.
            auto* pv = new std::atomic<uint64_t>[NN];
            for (size_t i = 0; i < NN; ++i) pv[i].store(0);
            bvhs[0].visits = pv;
            Camera pcam = make_camera(img_w);
            long npix = (long)img_w * pcam.height;
            int stride = (int)std::max(1L, npix / 400000L);
            Rng prng; prng.seed((uint64_t)seed * 0x9E3779B9u + 7, 0xABCDEF01u);
            for (long idx = 0; idx < npix; idx += stride) {
                int x = (int)(idx % img_w), y = (int)(idx / img_w);
                Path p = make_primary(pcam, x, y, y*img_w + x, (uint8_t)1, prng);
                traverse(bvhs[0], p);
            }
            bvhs[0].visits = nullptr;
            for (size_t ni = 0; ni < NN; ++ni) {
                const BVHNode2& nd = bvhs[0].nodes[ni];
                if (nd.count == 0) continue;                 // interior node
                uint64_t c = pv[ni].load(); if (c == 0) continue;
                uint32_t first = (uint32_t)nd.right_or_first;
                for (uint16_t k = 0; k < nd.count; ++k) {
                    const Triangle& t = bvhs[0].prims[first + k];
                    vhot[t.v[0]] += c; vhot[t.v[1]] += c; vhot[t.v[2]] += c;
                }
            }
            delete[] pv;
        }
        size_t hvc = (size_t)(leaf_repl_frac * (double)NV);
        if (hvc > NV) hvc = NV;
        double coverage = 0.0;
        if (hvc > 0 && NV > 0) {
            // select the hvc hottest original indices, then SORT ASCENDING so
            // slot order == original order (coherence preserved).
            std::vector<uint32_t> idx(NV);
            for (uint32_t i = 0; i < NV; ++i) idx[i] = i;
            std::nth_element(idx.begin(), idx.begin() + hvc, idx.end(),
                             [&](uint32_t a, uint32_t b){ return vhot[a] > vhot[b]; });
            uint64_t tot = 0, cov = 0;
            for (uint32_t i = 0; i < NV; ++i) tot += vhot[i];
            for (size_t r = 0; r < hvc; ++r) cov += vhot[idx[r]];
            coverage = (tot > 0) ? (double)cov / (double)tot : 0.0;
            std::sort(idx.begin(), idx.begin() + hvc);   // ascending original index
            // shared slot map (interleaved, like the cold verts)
            int32_t* vslot = (int32_t*)numa_alloc_interleaved(NV * sizeof(int32_t));
            if (!vslot) { fprintf(stderr, "vslot alloc failed\n"); return 1; }
            memset(vslot, 0xFF, NV * sizeof(int32_t));   // 0xFFFFFFFF = -1 (cold)
            std::vector<Vertex> packed(hvc);
            for (size_t r = 0; r < hvc; ++r) { vslot[idx[r]] = (int32_t)r; packed[r] = bvhs[0].verts[idx[r]]; }
            for (int s = 1; s < n_sockets; ++s) bvhs[s] = bvhs[0]; // share coherent geometry
            for (int s = 0; s < n_sockets; ++s) {
                bvhs[s].vert_slot = vslot;          // shared map
                bvhs[s].hot_vert_count = hvc;
                bvhs[s].hot_verts = numa_new<Vertex>(hvc, phys(s));
                Vertex* dst = bvhs[s].hot_verts; const Vertex* src = packed.data();
                std::thread ft([&, s, dst, src, hvc](){
                    auto cs = cpus_in_node(phys(s));
                    if (!cs.empty()) pin_thread_to_cpu(cs[0]);
                    memcpy(dst, src, hvc * sizeof(Vertex));   // first-touch local
                });
                ft.join();
            }
            hybrid_extra_bytes = hvc * sizeof(Vertex) * (size_t)n_sockets
                               + NV * sizeof(int32_t);   // + shared slot map
        } else {
            for (int s = 1; s < n_sockets; ++s) bvhs[s] = bvhs[0]; // frac=0 -> naive
        }
        fprintf(stderr, "leafhybrid(v2): frac=%.3f hot_verts=%zu/%zu (%.1f%%) "
                        "hot_MB_per_socket=%.2f slot_MB=%.2f access_coverage=%.3f\n",
                leaf_repl_frac, hvc, NV, NV ? 100.0*hvc/NV : 0.0,
                hvc * sizeof(Vertex) / 1048576.0,
                NV * sizeof(int32_t) / 1048576.0, coverage);
    } else if (mode == Mode::Replicate) {
        for (int s = 0; s < n_sockets; ++s) build_bvh_on(bvhs[s], tris, verts, phys(s), s, (uint32_t)leaf_max);
    } else {
        // Partition by triangle centroid x-median.
        std::vector<float> xs; xs.reserve(tris.size());
        for (const auto& t : tris) {
            float c = (verts[t.v[0]].pos.x + verts[t.v[1]].pos.x + verts[t.v[2]].pos.x) / 3.0f;
            xs.push_back(c);
        }
        std::vector<float> sorted_xs = xs;
        std::nth_element(sorted_xs.begin(), sorted_xs.begin() + sorted_xs.size()/2, sorted_xs.end());
        float x_split = sorted_xs[sorted_xs.size()/2];
        std::vector<std::vector<Triangle>> per_sock(n_sockets);
        for (size_t i = 0; i < tris.size(); ++i) {
            int sid = (xs[i] < x_split) ? 0 : 1;
            if (sid >= n_sockets) sid = n_sockets - 1;
            per_sock[sid].push_back(tris[i]);
        }
        // Each socket still gets the FULL vertex array (cheap, ~MB-scale),
        // because we partition by triangle but a triangle on socket s may
        // reference vertices that geometrically belong to either side.
        for (int s = 0; s < n_sockets; ++s) {
            build_bvh_on(bvhs[s], per_sock[s], verts, phys(s), s, (uint32_t)leaf_max);
        }
        fprintf(stderr, "partition: x_split=%.3f", x_split);
        for (int s = 0; s < n_sockets; ++s) fprintf(stderr, " sock%d=%zu tris", s, per_sock[s].size());
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
                        + bvhs[0].n_prims * sizeof(Triangle)
                        + bvhs[0].n_verts * sizeof(Vertex);

    printf("BENCH_STATS tag=rtiow_hairball mode=%s leaf_max=%d "
           "n_strands=%d segments=%d sides=%d n_triangles=%zu n_vertices=%zu "
           "samples=%d max_depth=%d seed=%d n_sockets=%d image_width=%d image_height=%d "
           "bvh_nodes_sock0=%zu bvh_bytes_per_socket=%zu hot_count=%zu "
           "hybrid_extra_MB=%.2f repl_depth=%d "
           "leaf_repl_frac=%.3f hot_vert_count=%zu leaf_repl_MB=%.2f "
           "tex_MB=%d tex_lookups=%d "
           "tex_replicated=%d feasible=1 budget_MB=%ld queue=%s "
           "actual_per_socket_MB=%.1f "
           "paths_enqueued=%lld rays_processed=%llu rays_forwarded=%llu "
           "paths_retired=%llu forwarding_rate=%.4f avg_visits_per_path=%.3f "
           "gen_ms=%ld bvh_build_ms=%ld genray_ms=%ld render_ms=%ld write_ms=%ld total_ms=%ld\n",
           mode_name(mode), leaf_max,
           n_strands, segments, sides, n_tris, n_verts,
           samples, max_depth, seed, n_sockets, img_w, img_h,
           bvhs[0].n_nodes, bvh_bytes_per, bvhs[0].hot_count,
           hybrid_extra_bytes / 1048576.0,
           (mode == Mode::Hybrid) ? repl_depth : 0,
           (mode == Mode::LeafHybrid) ? leaf_repl_frac : 0.0,
           bvhs[0].hot_vert_count,
           bvhs[0].hot_vert_count * sizeof(Vertex) / 1048576.0,
           tex_mb, tex_lookups,
           (tex_mb > 0 && mode == Mode::Replicate) ? 1 : 0,
           socket_budget_mb, queue_mode,
           // actual per-socket footprint: bvh bytes (per socket) + tex (per socket if replicated)
           ( (mode == Mode::Replicate)
               ? (bvh_bytes_per + (size_t)tex_mb*1048576ULL)
               : (bvh_bytes_per + ((mode==Mode::Hybrid || mode==Mode::LeafHybrid)
                                   ? (size_t)(hybrid_extra_bytes/std::max(1,n_sockets)) : 0)) )
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
        if (b.prims) numa_free(b.prims, b.n_prims * sizeof(Triangle));
        if (b.verts) numa_free(b.verts, b.n_verts * sizeof(Vertex));
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
