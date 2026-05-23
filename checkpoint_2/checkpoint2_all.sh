#!/bin/bash
# checkpoint2_all.sh - all Checkpoint 2 experiments in one driver.
#
# Writes rtiow_native.cpp + pointer_chase.cpp via embedded heredocs,
# builds them, runs five experiments end-to-end, emits CSV per experiment
# plus a final summary table. Designed for a 2-NUMA-socket Sherlock node.
#
# Experiments:
#   E1  hardware characterization (numactl, lscpu, NUMA distances)
#   E2  synthetic pointer-chase (random access, NUMA latency upper bound)
#   E3  RTIOW single-thread A/B (local vs remote DRAM)
#   E4  RTIOW multi-thread C/D (interleave vs sharded)
#   E5  RTIOW shadow-only (memory-bound regime - sharding should clearly win)
#
# Wall time: ~30-45 minutes depending on allocation size.

set -e

OUT_DIR="checkpoint2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR" && cd "$OUT_DIR"

echo "===================================================================="
echo " Checkpoint 2 driver — output dir: $OUT_DIR"
echo " Hostname: $(hostname)  Date: $(date)"
echo "===================================================================="

echo ""
echo "[1/3] Writing rtiow_native.cpp ..."
cat > rtiow_native.cpp <<'RTIOW_EOF'
// rtiow_native.cpp - Standalone RTIOW path tracer in pure C++.
//
// Does NOT link against main.o or depend on the bonsai compiler.
// Produces the same scene (same RNG seed -> same sphere placement) and uses
// the SAME memory layout as the bonsai variant
//
//   * MaterialSphere = 80 bytes, vec3 fields 16-byte aligned (matches IR's
//     non-packed { <3 x float>, float, i32, <3 x float>, float } layout)
//   * BVH node      = 22 bytes packed, scalar cx,cy,cz + radius + nPrims +
//     axis + u32 offset/pOffset slot (matches the u32-BVH IR's
//     <{ <3 x float>, float, i8, i8, <4 x i8> }>)
//
//
// Build:
//   clang++ -std=c++20 -O3 -march=native rtiow_native.cpp -o rtiow_native
//
// For multi-threaded runs add -fopenmp:
//   clang++ -std=c++20 -O3 -march=native -fopenmp rtiow_native.cpp -o rtiow_native
//
// Run (same env vars as bonsai.out):
//   RTIOW_GRID_HALF=11 RTIOW_SAMPLES=10 RTIOW_MAX_DEPTH=10 \
//       ./rtiow_native /tmp/test.ppm
//
//   numactl --membind=0 env RTIOW_GRID_HALF=500 RTIOW_SAMPLES=1 \
//       RTIOW_MAX_DEPTH=5 ./rtiow_native /tmp/local.ppm
//   numactl --membind=1 env RTIOW_GRID_HALF=500 RTIOW_SAMPLES=1 \
//       RTIOW_MAX_DEPTH=5 ./rtiow_native /tmp/remote.ppm

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---- vec3 (portable across GCC and Clang). GCC rejects vector_size(12)
//          because 3 isn't a power of 2, so we use a 16-byte-aligned struct
//          with manual operators. Memory layout (size 16, align 16) matches
//          what Clang's vector_size(12) typedef produced, which in turn
//          matches the IR's <3 x float> in non-packed structs. -----------
struct alignas(16) vec3 {
    float x, y, z;
    float _pad = 0;

    float  operator[](int i) const { return (&x)[i]; }
    float &operator[](int i)       { return (&x)[i]; }
    vec3 operator-() const         { return {-x, -y, -z, 0}; }
    vec3 operator+(vec3 b) const   { return {x + b.x, y + b.y, z + b.z, 0}; }
    vec3 operator-(vec3 b) const   { return {x - b.x, y - b.y, z - b.z, 0}; }
    vec3 operator*(vec3 b) const   { return {x * b.x, y * b.y, z * b.z, 0}; }
    vec3 operator/(vec3 b) const   { return {x / b.x, y / b.y, z / b.z, 0}; }
    // Scalar broadcasts (used by `vec3 / float` and similar).
    vec3 operator*(float s) const  { return {x * s, y * s, z * s, 0}; }
    vec3 operator/(float s) const  { return {x / s, y / s, z / s, 0}; }
    vec3 operator+(float s) const  { return {x + s, y + s, z + s, 0}; }
    vec3 operator-(float s) const  { return {x - s, y - s, z - s, 0}; }
};
static inline vec3 operator*(float s, vec3 v) { return v * s; }
static_assert(sizeof(vec3) == 16, "vec3 must be 16 bytes (12 data + 4 pad)");

static inline float vget(const vec3 &v, int i) { return v[i]; }
static inline vec3 vmake(float x, float y, float z) { return {x, y, z, 0}; }
static inline float vdot(vec3 a, vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
static inline vec3 vcross(vec3 a, vec3 b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x,
            0};
}
static inline float vlen(vec3 v) { return std::sqrt(vdot(v, v)); }
static inline vec3 vunit(vec3 v) {
    float l = vlen(v);
    return l > 0.0f ? v / vmake(l, l, l) : vmake(0, 0, 0);
}
static inline vec3 vmin(vec3 a, vec3 b) {
    return {std::fminf(a.x, b.x), std::fminf(a.y, b.y),
            std::fminf(a.z, b.z), 0};
}
static inline vec3 vmax(vec3 a, vec3 b) {
    return {std::fmaxf(a.x, b.x), std::fmaxf(a.y, b.y),
            std::fmaxf(a.z, b.z), 0};
}

// ---- Scene types (LAYOUT MUST MATCH bonsai for NUMA-comparable benchmarks) --
constexpr uint32_t LAMBERTIAN = 0;
constexpr uint32_t METAL = 1;
constexpr uint32_t DIELECTRIC = 2;

struct Sphere {
    vec3 center;
    float radius;
};
static_assert(sizeof(Sphere) == 32, "Sphere must be 32 bytes");

struct MaterialSphere {
    Sphere s;
    uint32_t material;
    vec3 albedo;
    float fuzz; // metal: fuzz; dielectric: refraction index
};
static_assert(sizeof(MaterialSphere) == 80, "MaterialSphere must be 80 bytes");

// Packed BVH node, matches u32-BVH IR layout (22 bytes).
// split0on_nPrims is overlaid with offset (interior) or pOffset (leaf).
struct BVHNode {
    float cx, cy, cz;
    float radius;
    uint8_t nPrims; // 0 = interior, >0 = leaf with nPrims (up to 255)
    uint8_t axis;
    uint32_t split; // interior: rel offset to right child; leaf: pOffset
} __attribute__((packed));
static_assert(sizeof(BVHNode) == 22, "BVHNode must be 22 bytes");

constexpr uint32_t MAX_LEAF_PRIMS = 1;
static_assert(MAX_LEAF_PRIMS >= 1 && MAX_LEAF_PRIMS <= 255);
constexpr uint32_t MAX_TREE_DEPTH = 96;

// ---- RNG ----
static std::mt19937 g_scene_rng;
// thread_local so each OpenMP worker has its own RNG state (no races).
// With single-threaded builds it's just one instance.
thread_local std::mt19937 g_thread_rng;
static inline void seed_scene(uint32_t s) { g_scene_rng.seed(s); }
static inline void seed_thread(uint32_t s) { g_thread_rng.seed(s); }
static inline float rand01_scene() {
    std::uniform_real_distribution<float> d(0.0f, 1.0f);
    return d(g_scene_rng);
}
static inline float rand_range_scene(float lo, float hi) {
    return lo + (hi - lo) * rand01_scene();
}
static inline float rand01() {
    std::uniform_real_distribution<float> d(0.0f, 1.0f);
    return d(g_thread_rng);
}
static inline float rand_range(float lo, float hi) {
    return lo + (hi - lo) * rand01();
}
static inline vec3 random_unit_vector() {
    // Rejection sampling on the unit ball, then normalize.
    while (true) {
        vec3 p{rand_range(-1, 1), rand_range(-1, 1), rand_range(-1, 1)};
        float l2 = vdot(p, p);
        if (l2 > 1e-10f && l2 <= 1.0f) return p / std::sqrt(l2);
    }
}
static inline vec3 random_in_unit_disk() {
    while (true) {
        vec3 p{rand_range(-1, 1), rand_range(-1, 1), 0.0f};
        if (vdot(p, p) < 1.0f) return p;
    }
}

// ---- Bounding-sphere fold (smallest enclosing sphere of two) --------------
static inline void merge_sphere(float &ox, float &oy, float &oz, float &orad,
                                float ax, float ay, float az, float ar,
                                float bx, float by, float bz, float br) {
    float dx = bx - ax, dy = by - ay, dz = bz - az;
    float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (ar >= dist + br) {
        ox = ax; oy = ay; oz = az; orad = ar;
    } else if (br >= dist + ar) {
        ox = bx; oy = by; oz = bz; orad = br;
    } else {
        orad = 0.5f * (dist + ar + br);
        float dir = dist > 0.0f ? 1.0f / dist : 0.0f;
        float t = orad - ar;
        ox = ax + t * dx * dir;
        oy = ay + t * dy * dir;
        oz = az + t * dz * dir;
    }
}

// ---- BVH build (matches main_hook.cpp's algorithm) -------------------------
struct BVH {
    std::vector<BVHNode> nodes;
    MaterialSphere *prims;
    uint32_t pCount;
};

static BVH build_bvh(std::vector<MaterialSphere> &spheres) {
    BVH bvh;
    bvh.pCount = static_cast<uint32_t>(spheres.size());
    bvh.prims = spheres.data();
    // Upper bound on node count
    size_t leaf_count =
        (bvh.pCount + (MAX_LEAF_PRIMS - 1)) / MAX_LEAF_PRIMS;
    bvh.nodes.resize(2 * leaf_count); // generous upper bound

    uint32_t next_node = 0;

    std::function<uint32_t(uint32_t, uint32_t, uint32_t)> rec =
        [&](uint32_t low, uint32_t high, uint32_t depth) -> uint32_t {
        assert(depth < MAX_TREE_DEPTH);
        uint32_t count = high - low;
        uint32_t idx = next_node++;
        if (idx >= bvh.nodes.size()) bvh.nodes.resize(bvh.nodes.size() * 2);

        if (count <= MAX_LEAF_PRIMS) {
            BVHNode &n = bvh.nodes[idx];
            n.nPrims = static_cast<uint8_t>(count);
            n.axis = 0;
            n.split = static_cast<uint32_t>(low); // pOffset
            float cx = spheres[low].s.center[0];
            float cy = spheres[low].s.center[1];
            float cz = spheres[low].s.center[2];
            float r = spheres[low].s.radius;
            for (uint32_t i = low + 1; i < high; ++i) {
                merge_sphere(cx, cy, cz, r, cx, cy, cz, r,
                             spheres[i].s.center[0], spheres[i].s.center[1],
                             spheres[i].s.center[2], spheres[i].s.radius);
            }
            n.cx = cx;
            n.cy = cy;
            n.cz = cz;
            n.radius = r;
            return idx;
        }

        bvh.nodes[idx].nPrims = 0;

        // Choose axis = longest extent of centers
        vec3 mn = spheres[low].s.center;
        vec3 mx = mn;
        for (uint32_t i = low + 1; i < high; ++i) {
            mn = vmin(mn, spheres[i].s.center);
            mx = vmax(mx, spheres[i].s.center);
        }
        vec3 ext = mx - mn;
        int ax = 0;
        if (ext[1] > ext[0]) ax = 1;
        if (ext[2] > ext[ax]) ax = 2;
        bvh.nodes[idx].axis = static_cast<uint8_t>(ax);

        uint32_t mid = low + count / 2;
        std::nth_element(
            spheres.begin() + low, spheres.begin() + mid, spheres.begin() + high,
            [ax](const MaterialSphere &a, const MaterialSphere &b) {
                return a.s.center[ax] < b.s.center[ax];
            });

        /*uint32_t left =*/ rec(low, mid, depth + 1);
        uint32_t right = rec(mid, high, depth + 1);
        bvh.nodes[idx].split = right - idx;

        float cx, cy, cz, r;
        const BVHNode &L = bvh.nodes[idx + 1];
        const BVHNode &R = bvh.nodes[right];
        merge_sphere(cx, cy, cz, r,
                     L.cx, L.cy, L.cz, L.radius,
                     R.cx, R.cy, R.cz, R.radius);
        bvh.nodes[idx].cx = cx;
        bvh.nodes[idx].cy = cy;
        bvh.nodes[idx].cz = cz;
        bvh.nodes[idx].radius = r;
        return idx;
    };
    rec(0, bvh.pCount, 0);
    bvh.nodes.resize(next_node); // trim
    return bvh;
}

// ---- Ray + intersection ----------------------------------------------------
struct Ray {
    vec3 o, d;
};
struct Hit {
    float t;          // ray parameter at hit
    vec3 p;           // hit point
    vec3 n;           // outward normal
    bool front_face;
    uint32_t mat;
    vec3 albedo;
    float fuzz;
};

// Returns smallest positive t in (t_min, t_max), or -1 if no hit.
static inline float sphere_intersect(const Ray &r, const Sphere &s,
                                     float t_min, float t_max) {
    vec3 oc = r.o - s.center;
    float a = vdot(r.d, r.d);
    float half_b = vdot(oc, r.d);
    float c = vdot(oc, oc) - s.radius * s.radius;
    float disc = half_b * half_b - a * c;
    if (disc < 0) return -1.0f;
    float sd = std::sqrt(disc);
    float t = (-half_b - sd) / a;
    if (t < t_min || t > t_max) {
        t = (-half_b + sd) / a;
        if (t < t_min || t > t_max) return -1.0f;
    }
    return t;
}

// Sphere bounding-volume test for BVH nodes — same math, no closest-hit needed.
static inline bool sphere_overlap_ray(const Ray &r, float cx, float cy,
                                      float cz, float radius, float t_max) {
    vec3 c = vmake(cx, cy, cz);
    vec3 oc = r.o - c;
    float a = vdot(r.d, r.d);
    float half_b = vdot(oc, r.d);
    float cc = vdot(oc, oc) - radius * radius;
    float disc = half_b * half_b - a * cc;
    if (disc < 0) return false;
    float sd = std::sqrt(disc);
    float t1 = (-half_b - sd) / a;
    float t2 = (-half_b + sd) / a;
    // Hit if any t in (eps, t_max). t2 >= t1 always.
    return t2 > 0.001f && t1 < t_max;
}

// ---- BVH traversal (explicit stack, single-threaded) -----------------------
static bool bvh_hit(const BVH &bvh, const Ray &r, float t_min, float t_max,
                    Hit &out) {
    bool hit_any = false;
    float closest = t_max;
    // Iterative stack of node indices to visit.
    uint32_t stack[MAX_TREE_DEPTH + 1];
    int sp = 0;
    stack[sp++] = 0; // root
    const BVHNode *nodes = bvh.nodes.data();
    const MaterialSphere *prims = bvh.prims;
    while (sp > 0) {
        uint32_t idx = stack[--sp];
        const BVHNode &n = nodes[idx];
        if (!sphere_overlap_ray(r, n.cx, n.cy, n.cz, n.radius, closest))
            continue;
        if (n.nPrims == 0) {
            // Interior: push both children. Right is at idx + offset, left at idx + 1.
            uint32_t left = idx + 1;
            uint32_t right = idx + n.split;
            stack[sp++] = left;
            stack[sp++] = right;
        } else {
            // Leaf: linear scan
            uint32_t p_off = n.split;
            for (uint32_t i = 0; i < n.nPrims; ++i) {
                const MaterialSphere &ms = prims[p_off + i];
                float t =
                    sphere_intersect(r, ms.s, t_min, closest);
                if (t > 0) {
                    closest = t;
                    hit_any = true;
                    out.t = t;
                    out.p = r.o + r.d * vmake(t, t, t);
                    vec3 outward = (out.p - ms.s.center) / ms.s.radius;
                    out.front_face = vdot(r.d, outward) < 0;
                    out.n = out.front_face ? outward : -outward;
                    out.mat = ms.material;
                    out.albedo = ms.albedo;
                    out.fuzz = ms.fuzz;
                }
            }
        }
    }
    return hit_any;
}

// ---- Materials -------------------------------------------------------------
static inline bool near_zero(vec3 v) {
    const float eps = 1e-8f;
    return std::fabs(v[0]) < eps && std::fabs(v[1]) < eps &&
           std::fabs(v[2]) < eps;
}
static inline vec3 reflect(vec3 v, vec3 n) { return v - n * vmake(2 * vdot(v, n), 2 * vdot(v, n), 2 * vdot(v, n)); }
static inline vec3 refract(vec3 uv, vec3 n, float eta) {
    float cos_theta = std::fminf(vdot(-uv, n), 1.0f);
    vec3 r_perp = (uv + n * vmake(cos_theta, cos_theta, cos_theta)) *
                  vmake(eta, eta, eta);
    float k = -std::sqrt(std::fmaxf(1.0f - vdot(r_perp, r_perp), 0.0f));
    vec3 r_parallel = n * vmake(k, k, k);
    return r_perp + r_parallel;
}
static inline float schlick(float cosine, float ref_idx) {
    float r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * std::pow(1 - cosine, 5.0f);
}

struct ScatterResult {
    Ray scattered;
    vec3 attenuation;
    bool ok;
};
static inline ScatterResult scatter(const Ray &r, const Hit &h) {
    if (h.mat == LAMBERTIAN) {
        vec3 dir = h.n + random_unit_vector();
        if (near_zero(dir)) dir = h.n;
        return {{h.p, dir}, h.albedo, true};
    } else if (h.mat == METAL) {
        vec3 ref = vunit(reflect(r.d, h.n));
        vec3 dir = ref + random_unit_vector() * vmake(h.fuzz, h.fuzz, h.fuzz);
        bool ok = vdot(dir, h.n) > 0;
        return {{h.p, dir}, h.albedo, ok};
    } else { // DIELECTRIC
        float ri = h.front_face ? (1.0f / h.fuzz) : h.fuzz;
        vec3 ud = vunit(r.d);
        float cos_t = std::fminf(vdot(-ud, h.n), 1.0f);
        float sin_t = std::sqrt(1.0f - cos_t * cos_t);
        bool cannot_refract = (ri * sin_t > 1.0f) || (schlick(cos_t, ri) > rand01());
        vec3 dir = cannot_refract ? reflect(ud, h.n) : refract(ud, h.n, ri);
        return {{h.p, dir}, vmake(1, 1, 1), true};
    }
}

// Global flag — set once in main() from RTIOW_NO_SKY env var. When true, the
// "ray escapes to sky" path returns black, forcing every ray to bounce until
// it hits max_depth. Used to maximize secondary-ray incoherence for the
// sharding benchmarks (rays can't escape early into the sky).
static bool g_no_sky = false;

// Global flag — set once in main() from RTIOW_SHADOW_ONLY env var. When true,
// rays do NOT scatter or recurse — they just test "hit anything?" and return
// black (hit) or white (miss). This strips per-ray compute down to the bare
// minimum (one BVH traversal + sphere intersection tests), removing the
// compute that masks remote-DRAM latency. Used to expose the memory-bound
// regime where sharding gives a clean win.
static bool g_shadow_only = false;

// ---- sample() -- iterative ray tracing, semantically matches bonsai's
// recursive sample() (each bounce multiplies `mult` by attenuation; on miss
// we return `mult * sky_color`; on max-depth bottom-out we return 0).
static vec3 sample_ray(Ray r, int max_depth, const BVH &bvh) {
    if (g_shadow_only) {
        // Shadow-ray mode: just test for any hit, no shading, no recursion.
        Hit h;
        return bvh_hit(bvh, r, 0.001f, 1e30f, h) ? vmake(0, 0, 0) : vmake(1, 1, 1);
    }
    vec3 mult = vmake(1, 1, 1);
    for (int d = 0; d < max_depth; ++d) {
        Hit h;
        if (!bvh_hit(bvh, r, 0.001f, 1e30f, h)) {
            if (g_no_sky) return vmake(0, 0, 0);
            vec3 ud = vunit(r.d);
            float a = 0.5f * (ud[1] + 1.0f);
            vec3 sky = vmake(1, 1, 1) * vmake(1 - a, 1 - a, 1 - a) +
                       vmake(0.5f, 0.7f, 1.0f) * vmake(a, a, a);
            return mult * sky;
        }
        ScatterResult sr = scatter(r, h);
        if (!sr.ok) return vmake(0, 0, 0);
        mult = mult * sr.attenuation;
        r = sr.scattered;
    }
    return vmake(0, 0, 0);
}

// ---- Camera ----------------------------------------------------------------
struct Camera {
    vec3 origin;
    vec3 lower_left;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lens_radius;
};
static Camera make_camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov_deg,
                          float aspect, float defocus_angle_deg,
                          float focus_dist) {
    Camera c;
    float theta = vfov_deg * 3.14159265358979f / 180.0f;
    float h = std::tan(theta / 2.0f);
    float viewport_h = 2.0f * h * focus_dist;
    float viewport_w = aspect * viewport_h;
    c.w = vunit(lookfrom - lookat);
    c.u = vunit(vcross(vup, c.w));
    c.v = vcross(c.w, c.u);
    c.origin = lookfrom;
    c.horizontal = c.u * vmake(viewport_w, viewport_w, viewport_w);
    c.vertical = c.v * vmake(viewport_h, viewport_h, viewport_h);
    c.lower_left = c.origin -
                   c.horizontal * vmake(0.5f, 0.5f, 0.5f) -
                   c.vertical * vmake(0.5f, 0.5f, 0.5f) -
                   c.w * vmake(focus_dist, focus_dist, focus_dist);
    // Match bonsai's defocus interpretation: angle in degrees, lens radius
    // derived from the focus distance.
    float defocus_radius =
        focus_dist * std::tan((defocus_angle_deg * 3.14159265358979f / 180.0f) / 2.0f);
    c.lens_radius = defocus_radius;
    return c;
}
static inline Ray camera_ray(const Camera &c, float s, float t) {
    vec3 rd = random_in_unit_disk() *
              vmake(c.lens_radius, c.lens_radius, c.lens_radius);
    vec3 offset = c.u * vmake(rd[0], rd[0], rd[0]) +
                  c.v * vmake(rd[1], rd[1], rd[1]);
    Ray r;
    r.o = c.origin + offset;
    r.d = c.lower_left +
          c.horizontal * vmake(s, s, s) +
          c.vertical * vmake(t, t, t) -
          c.origin - offset;
    return r;
}

// ---- env helpers -----------------------------------------------------------
static int env_int(const char *name, int dflt) {
    const char *s = std::getenv(name);
    if (!s || !*s) return dflt;
    char *end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0') return dflt;
    return static_cast<int>(v);
}

// ---- main ------------------------------------------------------------------
int main(int argc, char *argv[]) {
    using clock = std::chrono::high_resolution_clock;
    std::string out_path = (argc == 2) ? argv[1] : "rtiow-native-image.ppm";

    const int grid_half = std::max(1, env_int("RTIOW_GRID_HALF", 11));
    const int image_w = std::max(1, env_int("RTIOW_IMAGE_WIDTH", 1200));
    const int samples = std::max(1, env_int("RTIOW_SAMPLES", 50));
    const int max_depth = std::max(1, env_int("RTIOW_MAX_DEPTH", 1));
    const int seed = env_int("RTIOW_SEED", 42);
    // Tile mode: process only rows [tile_start, tile_end). Lets one process per
    // NUMA node render its slice with its own local copy of the BVH+prims, for
    // sharded benchmarks (no cross-socket access).
    const int tile_start_env = env_int("RTIOW_TILE_START", 0);
    const int tile_end_env   = env_int("RTIOW_TILE_END", -1);  // -1 = full image
    // Band-interleaved sharding: render a pixel iff (axis / SHARD_BAND) % SHARD_OF == SHARD_ID.
    // SHARD_AXIS = 0 -> band by row (axis = j); 1 -> band by column (axis = i).
    // Default SHARD_OF=1 means "render everything" (no shard filter).
    const int shard_band  = std::max(1, env_int("RTIOW_SHARD_BAND", 8));
    const int shard_of    = std::max(1, env_int("RTIOW_SHARD_OF", 1));
    const int shard_id    = std::max(0, env_int("RTIOW_SHARD_ID", 0));
    const int shard_axis  = std::max(0, env_int("RTIOW_SHARD_AXIS", 0));  // 0=rows, 1=columns

    std::cerr << "RTIOW_GRID_HALF=" << grid_half << " (grid cells per axis "
              << (2 * grid_half) << ", ~" << (2 * grid_half) * (2 * grid_half)
              << " positions)\n";
    std::cerr << "RTIOW_IMAGE_WIDTH=" << image_w << '\n';
    std::cerr << "RTIOW_SAMPLES=" << samples << '\n';
    std::cerr << "RTIOW_MAX_DEPTH=" << max_depth << '\n';
    std::cerr << "RTIOW_SEED=" << seed << '\n';

    seed_scene(static_cast<uint32_t>(seed));
    // Each worker thread seeds its own g_thread_rng below, derived from this base.
    const uint32_t thread_seed_base = static_cast<uint32_t>(seed) ^ 0xdeadbeefu;
    seed_thread(thread_seed_base); // covers the single-threaded case + scene init

    auto t0 = clock::now();

    // ---- env knobs for incoherent-secondary-ray experiments ----
    // RTIOW_ALL_LAMBERTIAN=1 forces every sphere to be Lambertian, so every
    // bounce randomizes the ray direction uniformly over the hemisphere.
    // RTIOW_NO_SKY=1 makes the sky-miss return black, preventing early ray
    // termination — rays must keep bouncing until MAX_DEPTH runs out, which
    // maximizes the number of incoherent BVH descents per pixel.
    const bool all_lambertian = (env_int("RTIOW_ALL_LAMBERTIAN", 0) != 0);
    const bool no_sky         = (env_int("RTIOW_NO_SKY", 0) != 0);
    const bool shadow_only    = (env_int("RTIOW_SHADOW_ONLY", 0) != 0);
    g_no_sky = no_sky;
    g_shadow_only = shadow_only;
    std::cerr << "ALL_LAMBERTIAN=" << all_lambertian
              << " NO_SKY=" << no_sky
              << " SHADOW_ONLY=" << shadow_only << '\n';

    // ---- scene ----
    std::vector<MaterialSphere> spheres;
    if (all_lambertian) {
        spheres.push_back({{ vmake(0, -1000, 0), 1000 }, LAMBERTIAN, vmake(0.5, 0.5, 0.5), 0.0});
        spheres.push_back({{ vmake(0, 1, 0),     1    }, LAMBERTIAN, vmake(0.8, 0.3, 0.3), 0.0});
        spheres.push_back({{ vmake(-4, 1, 0),    1    }, LAMBERTIAN, vmake(0.4, 0.2, 0.1), 0.0});
        spheres.push_back({{ vmake(4, 1, 0),     1    }, LAMBERTIAN, vmake(0.7, 0.6, 0.5), 0.0});
    } else {
        spheres.push_back({{ vmake(0, -1000, 0), 1000 }, LAMBERTIAN, vmake(0.5, 0.5, 0.5), 0.0});
        spheres.push_back({{ vmake(0, 1, 0),     1    }, DIELECTRIC, vmake(0, 0, 0),       1.5});
        spheres.push_back({{ vmake(-4, 1, 0),    1    }, LAMBERTIAN, vmake(0.4, 0.2, 0.1), 0.0});
        spheres.push_back({{ vmake(4, 1, 0),     1    }, METAL,      vmake(0.7, 0.6, 0.5), 0.0});
    }
    for (int ai = -grid_half; ai < grid_half; ++ai) {
        for (int bi = -grid_half; bi < grid_half; ++bi) {
            float choose = rand01_scene();
            vec3 center = vmake(static_cast<float>(ai + 0.9 * rand01_scene()),
                                0.2f,
                                static_cast<float>(bi + 0.9 * rand01_scene()));
            vec3 anchor = vmake(4, 0.2, 0);
            if (vlen(center - anchor) > 0.9f) {
                if (all_lambertian || choose < 0.8f) {
                    vec3 r0 = vmake(rand01_scene(), rand01_scene(), rand01_scene());
                    vec3 r1 = vmake(rand01_scene(), rand01_scene(), rand01_scene());
                    spheres.push_back({{center, 0.2}, LAMBERTIAN, r0 * r1, 0.0});
                } else if (choose < 0.95f) {
                    vec3 alb = vmake(rand_range_scene(0.5f, 1.0f),
                                     rand_range_scene(0.5f, 1.0f),
                                     rand_range_scene(0.5f, 1.0f));
                    float fz = rand_range_scene(0.0f, 0.5f);
                    spheres.push_back({{center, 0.2}, METAL, alb, fz});
                } else {
                    spheres.push_back({{center, 0.2}, DIELECTRIC, vmake(0, 0, 0), 1.5});
                }
            }
        }
    }

    BVH bvh = build_bvh(spheres);

    std::cerr << "n_spheres=" << spheres.size()
              << " n_nodes=" << bvh.nodes.size()
              << " bvh_bytes=" << (bvh.nodes.size() * sizeof(BVHNode))
              << " prims_bytes=" << (spheres.size() * sizeof(MaterialSphere))
              << " total_scene_bytes=" << (bvh.nodes.size() * sizeof(BVHNode) +
                                           spheres.size() * sizeof(MaterialSphere))
              << '\n';

    // ---- camera ----
    const float aspect = 16.0f / 9.0f;
    int image_h = static_cast<int>(image_w / aspect);
    if (image_h < 1) image_h = 1;
    // Clamp tile range to [0, image_h).
    const int tile_lo = std::max(0, tile_start_env);
    const int tile_hi = (tile_end_env < 0) ? image_h
                                            : std::min(image_h, tile_end_env);
    std::cerr << "tile rows: [" << tile_lo << ", " << tile_hi << ") of "
              << image_h << '\n';
    Camera cam = make_camera(vmake(13, 2, 3), vmake(0, 0, 0), vmake(0, 1, 0),
                             20.0f, aspect, 0.6f, 10.0f);

    auto t1 = clock::now();

    // ---- render ----
    // Single-threaded by default. Compile with -fopenmp to enable threading
    // (do NOT use threading for NUMA bandwidth measurements).
    // Build the list of pixels this process is responsible for (applies shard
    // filter once, not inside the hot loop). Optional shuffle for incoherence
    // experiments — randomized pixel order defeats prefetcher and exposes
    // memory latency, which is the regime where sharding wins decisively.
    const bool shuffle = (env_int("RTIOW_SHUFFLE", 0) != 0);
    std::cerr << "shuffle pixel order: " << (shuffle ? "yes" : "no") << '\n';
    std::vector<std::pair<int, int>> pixel_order;
    pixel_order.reserve(static_cast<size_t>(tile_hi - tile_lo) * image_w);
    for (int j = tile_lo; j < tile_hi; ++j) {
        if (shard_of > 1 && shard_axis == 0 &&
            ((j / shard_band) % shard_of) != shard_id) continue;
        for (int i = 0; i < image_w; ++i) {
            if (shard_of > 1 && shard_axis == 1 &&
                ((i / shard_band) % shard_of) != shard_id) continue;
            pixel_order.emplace_back(j, i);
        }
    }
    if (shuffle) {
        std::mt19937 sh_rng(static_cast<uint32_t>(seed) ^ 0xc0ffeeu);
        std::shuffle(pixel_order.begin(), pixel_order.end(), sh_rng);
    }

    std::vector<int> img(image_w * image_h * 3);
#ifdef _OPENMP
    std::cerr << "OpenMP threads: " << omp_get_max_threads() << '\n';
#pragma omp parallel
    {
        // Each thread gets its own RNG stream, deterministic for a given seed.
        g_thread_rng.seed(thread_seed_base +
                          static_cast<uint32_t>(omp_get_thread_num()) * 7919u);
#pragma omp for schedule(dynamic, 256)
        for (size_t pidx = 0; pidx < pixel_order.size(); ++pidx) {
            int j = pixel_order[pidx].first;
            int i = pixel_order[pidx].second;
            vec3 acc = vmake(0, 0, 0);
            for (int k = 0; k < samples; ++k) {
                float s = (i + rand01()) / float(image_w);
                float t = 1.0f - (j + rand01()) / float(image_h);
                Ray r = camera_ray(cam, s, t);
                acc = acc + sample_ray(r, max_depth, bvh);
            }
            acc = acc / vmake(float(samples), float(samples), float(samples));
            int ir = std::min(255, int(256.0f * std::sqrt(std::fmaxf(acc[0], 0.0f))));
            int ig = std::min(255, int(256.0f * std::sqrt(std::fmaxf(acc[1], 0.0f))));
            int ib = std::min(255, int(256.0f * std::sqrt(std::fmaxf(acc[2], 0.0f))));
            img[(j * image_w + i) * 3 + 0] = ir;
            img[(j * image_w + i) * 3 + 1] = ig;
            img[(j * image_w + i) * 3 + 2] = ib;
        }
    }
#else
    for (size_t pidx = 0; pidx < pixel_order.size(); ++pidx) {
        int j = pixel_order[pidx].first;
        int i = pixel_order[pidx].second;
        vec3 acc = vmake(0, 0, 0);
        for (int k = 0; k < samples; ++k) {
            float s = (i + rand01()) / float(image_w);
            float t = 1.0f - (j + rand01()) / float(image_h);
            Ray r = camera_ray(cam, s, t);
            acc = acc + sample_ray(r, max_depth, bvh);
        }
        acc = acc / vmake(float(samples), float(samples), float(samples));
        int ir = std::min(255, int(256.0f * std::sqrt(std::fmaxf(acc[0], 0.0f))));
        int ig = std::min(255, int(256.0f * std::sqrt(std::fmaxf(acc[1], 0.0f))));
        int ib = std::min(255, int(256.0f * std::sqrt(std::fmaxf(acc[2], 0.0f))));
        img[(j * image_w + i) * 3 + 0] = ir;
        img[(j * image_w + i) * 3 + 1] = ig;
        img[(j * image_w + i) * 3 + 2] = ib;
    }
#endif

    auto t2 = clock::now();

    std::ofstream out(out_path);
    out << "P3\n" << image_w << ' ' << image_h << "\n255\n";
    for (int j = 0; j < image_h; ++j) {
        for (int i = 0; i < image_w; ++i) {
            int *p = &img[(j * image_w + i) * 3];
            out << p[0] << ' ' << p[1] << ' ' << p[2] << '\n';
        }
    }
    auto t3 = clock::now();

    auto ms = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(b - a)
            .count();
    };
    std::cout << "Setup time: " << ms(t0, t1) << " ms\n";
    std::cout << "Render time: " << ms(t1, t2) << " ms\n";
    std::cout << "Write-to-output time: " << ms(t2, t3) << " ms\n";
    std::cout << "n_spheres: " << spheres.size() << '\n';
    std::cout << "n_tree_nodes: " << bvh.nodes.size() << '\n';
    std::cout << "tree_node_bytes: " << sizeof(BVHNode) << '\n';
    std::cout << "tree_total_bytes: " << (bvh.nodes.size() * sizeof(BVHNode)) << '\n';
    std::cout << "prims_total_bytes: " << (spheres.size() * sizeof(MaterialSphere)) << '\n';

    {
        size_t bvh_b = bvh.nodes.size() * sizeof(BVHNode);
        size_t pr_b = spheres.size() * sizeof(MaterialSphere);
        std::cout << "BENCH_STATS"
                  << " grid_half=" << grid_half
                  << " image_width=" << image_w
                  << " samples=" << samples
                  << " max_depth=" << max_depth
                  << " seed=" << seed
                  << " n_spheres=" << spheres.size()
                  << " n_nodes=" << bvh.nodes.size()
                  << " bvh_bytes=" << bvh_b
                  << " prims_bytes=" << pr_b
                  << " total_scene_bytes=" << (bvh_b + pr_b)
                  << " setup_ms=" << ms(t0, t1)
                  << " render_ms=" << ms(t1, t2)
                  << " write_ms=" << ms(t2, t3)
                  << '\n';
    }
    return 0;
}
RTIOW_EOF
echo "    $(wc -l < rtiow_native.cpp) lines"

echo "[1.5/3] Writing pointer_chase.cpp ..."
cat > pointer_chase.cpp <<'PC_EOF'
// pointer_chase.cpp — synthetic random-access benchmark for NUMA latency.
// Allocates an array of N indices forming a single random Hamiltonian cycle,
// then walks the cycle for ITERS hops. Each hop is one cache-line miss with
// no compute to hide it. Run under numactl --membind to measure local-vs-
// remote DRAM latency cleanly.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main(int argc, char **argv) {
    size_t bytes  = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : (1ULL << 30);
    size_t iters  = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 200000000ULL;
    size_t n = bytes / sizeof(size_t);
    std::vector<size_t> arr(n);

    // Build a single random cycle through arr indices.
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = i;
    std::shuffle(order.begin(), order.end(), std::mt19937_64(42));
    for (size_t i = 0; i < n; ++i) arr[order[i]] = order[(i + 1) % n];

    auto t0 = std::chrono::high_resolution_clock::now();
    size_t pos = 0;
    for (size_t i = 0; i < iters; ++i) pos = arr[pos];
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ns = ms * 1e6 / iters;
    printf("BENCH_PC bytes=%zu iters=%zu wall_ms=%.1f ns_per_chase=%.2f final=%zu\n",
        bytes, iters, ms, ns, pos);
    return 0;
}
PC_EOF
echo "    $(wc -l < pointer_chase.cpp) lines"

# ---- Build ----
echo "[2/3] Building ..."
module load gcc/14.2.0 2>/dev/null || true
g++ -std=c++20 -O3 -march=native -fopenmp rtiow_native.cpp -o rtiow_native_mt
g++ -std=c++20 -O3 -march=native            pointer_chase.cpp -o pointer_chase
echo "    rtiow_native_mt + pointer_chase ready"

# ---- Auto-detect per-socket core count ----
N_SOCK0=$(numactl --hardware | awk '/node 0 cpus:/{print NF-3}')
N_SOCK1=$(numactl --hardware | awk '/node 1 cpus:/{print NF-3}')
N_TOTAL=$(( N_SOCK0 + N_SOCK1 ))
echo "    NUMA: socket0=${N_SOCK0}c  socket1=${N_SOCK1}c  total=${N_TOTAL}c"

extract() { awk -F"$1=" "/$1=/{print \$2}" "$2" 2>/dev/null | awk '{print $1}' | head -1; }

# All output PPMs land here so they're preserved per cell + can be inspected
mkdir -p images
echo "    Output images -> $(pwd)/images/"

# Stitch s0+s1 PPMs (each shard writes its own full-size buffer; non-shard
# pixels are zero, so pixel-wise add reconstructs the full image).
stitch() {
    local s0="$1" s1="$2" out="$3"
    python3 - "$s0" "$s1" "$out" 2>/dev/null <<'PYEOF' || true
import sys
def rd(p):
    with open(p) as f: lines = f.readlines()
    return lines[:3], [list(map(int, l.split())) for l in lines[3:]]
s0_p, s1_p, out_p = sys.argv[1:4]
h, s0 = rd(s0_p); _, s1 = rd(s1_p)
combined = [[a + b for a, b in zip(p0, p1)] for p0, p1 in zip(s0, s1)]
with open(out_p, 'w') as f:
    f.writelines(h)
    for r in combined: f.write(' '.join(map(str, r)) + '\n')
PYEOF
}

echo ""
echo "[3/3] Running experiments ..."
T_TOTAL_START=$(date +%s)

# ---- E1: Hardware characterization ----
echo ""
echo "==== E1: HARDWARE ===="
{
    echo "--- numactl --hardware ---"
    numactl --hardware
    echo
    echo "--- lscpu ---"
    lscpu | grep -E "Model name|Socket|Thread|Core|L1d|L2|L3|NUMA"
    echo
    echo "--- /sys NUMA distance ---"
    cat /sys/devices/system/node/distance 2>/dev/null
    echo
    echo "--- /proc/cpuinfo flags (1 cpu) ---"
    grep "model name\|cpu MHz\|cache size\|flags" /proc/cpuinfo | head -5
} | tee e1_hardware.txt | head -30

# ---- E2: Synthetic pointer-chase, local vs remote ----
echo ""
echo "==== E2: POINTER CHASE (synthetic memory latency) ===="
echo "exp,membind,bytes,iters,wall_ms,ns_per_chase" > e2_pointer_chase.csv
for BYTES in 100000000 1000000000 4000000000; do   # 100MB, 1GB, 4GB
    for MB in 0 1; do
        OUT=$(numactl --cpunodebind=0 --membind=$MB ./pointer_chase $BYTES 200000000 2>&1)
        WALL=$(echo "$OUT" | awk -F'wall_ms=' '/BENCH_PC/{print $2}' | awk '{print $1}')
        NS=$(echo "$OUT" | awk -F'ns_per_chase=' '/BENCH_PC/{print $2}' | awk '{print $1}')
        echo "E2,$MB,$BYTES,200000000,$WALL,$NS" | tee -a e2_pointer_chase.csv
    done
done

# ---- E3: RTIOW single-thread A (local) vs B (remote) ----
echo ""
echo "==== E3: RTIOW single-thread A=local B=remote ===="
echo "exp,config,grid,depth,samples,setup_ms,render_ms,wall_ms" > e3_AB.csv
for GRID in 500 1500 3000; do
    for MB in 0 1; do
        TAG=$([ $MB = 0 ] && echo "A_local" || echo "B_remote")
        PPM="images/E3_${TAG}_g${GRID}.ppm"
        T0=$(date +%s%3N)
        OMP_NUM_THREADS=1 numactl --cpunodebind=0 --membind=$MB \
            env RTIOW_GRID_HALF=$GRID RTIOW_SAMPLES=1 RTIOW_MAX_DEPTH=5 \
            ./rtiow_native_mt "$PPM" > /tmp/x.log 2>&1
        T1=$(date +%s%3N)
        RENDER=$(extract render_ms /tmp/x.log)
        SETUP=$(extract setup_ms /tmp/x.log)
        echo "E3,$TAG,$GRID,5,1,$SETUP,$RENDER,$((T1-T0))" | tee -a e3_AB.csv
    done
done

# ---- E4: RTIOW multi-thread C (interleave) vs D (sharded), normal scene ----
echo ""
echo "==== E4: RTIOW multi-thread C=interleave D=sharded (normal scene) ===="
echo "exp,config,grid,depth,samples,setup_ms,render_ms,wall_ms" > e4_CD.csv
run_C() {
    local g="$1" d="$2"
    local PPM="images/E4_C_g${g}_d${d}.ppm"
    T0=$(date +%s%3N)
    OMP_NUM_THREADS=$N_TOTAL numactl --interleave=all \
        env RTIOW_GRID_HALF=$g RTIOW_SAMPLES=4 RTIOW_MAX_DEPTH=$d \
        ./rtiow_native_mt "$PPM" > /tmp/c.log 2>&1
    T1=$(date +%s%3N)
    R=$(extract render_ms /tmp/c.log); S=$(extract setup_ms /tmp/c.log)
    echo "E4,C,$g,$d,4,$S,$R,$((T1-T0))" >> e4_CD.csv
    echo "  C g=$g d=$d wall=$((T1-T0))ms render=${R}ms"
}
run_D() {
    local g="$1" d="$2"
    local PPM0="images/E4_D_s0_g${g}_d${d}.ppm"
    local PPM1="images/E4_D_s1_g${g}_d${d}.ppm"
    local PPMC="images/E4_D_combined_g${g}_d${d}.ppm"
    T0=$(date +%s%3N)
    ( OMP_NUM_THREADS=$N_SOCK0 numactl --cpunodebind=0 --membind=0 \
        env RTIOW_SHARD_BAND=1 RTIOW_SHARD_OF=2 RTIOW_SHARD_ID=0 RTIOW_SHARD_AXIS=1 \
            RTIOW_GRID_HALF=$g RTIOW_SAMPLES=4 RTIOW_MAX_DEPTH=$d \
        ./rtiow_native_mt "$PPM0" > /tmp/s0.log 2>&1 ) &
    ( OMP_NUM_THREADS=$N_SOCK1 numactl --cpunodebind=1 --membind=1 \
        env RTIOW_SHARD_BAND=1 RTIOW_SHARD_OF=2 RTIOW_SHARD_ID=1 RTIOW_SHARD_AXIS=1 \
            RTIOW_GRID_HALF=$g RTIOW_SAMPLES=4 RTIOW_MAX_DEPTH=$d \
        ./rtiow_native_mt "$PPM1" > /tmp/s1.log 2>&1 ) &
    wait
    T1=$(date +%s%3N)
    R0=$(extract render_ms /tmp/s0.log); R1=$(extract render_ms /tmp/s1.log)
    R=$([ "${R0:-0}" -gt "${R1:-0}" ] 2>/dev/null && echo "$R0" || echo "$R1")
    S=$(extract setup_ms /tmp/s0.log)
    echo "E4,D,$g,$d,4,$S,$R,$((T1-T0))" >> e4_CD.csv
    echo "  D g=$g d=$d wall=$((T1-T0))ms render_max=${R}ms (s0=${R0} s1=${R1})"
    stitch "$PPM0" "$PPM1" "$PPMC"
}
for GRID in 500 1500 3000; do
    for DEPTH in 1 5 10; do
        echo "--- E4 GRID=$GRID DEPTH=$DEPTH ---"
        run_C "$GRID" "$DEPTH"
        run_D "$GRID" "$DEPTH"
    done
done

# ---- E5: RTIOW shadow-only mode (low compute, high memory pressure) ----
echo ""
echo "==== E5: RTIOW shadow-only C vs D (memory-bound regime) ===="
echo "exp,config,grid,samples,setup_ms,render_ms,wall_ms" > e5_shadow.csv
run_C_shadow() {
    local g="$1" sam="$2"
    local PPM="images/E5_C_g${g}_s${sam}.ppm"
    T0=$(date +%s%3N)
    OMP_NUM_THREADS=$N_TOTAL numactl --interleave=all \
        env RTIOW_GRID_HALF=$g RTIOW_SAMPLES=$sam RTIOW_MAX_DEPTH=1 \
            RTIOW_SHADOW_ONLY=1 \
        ./rtiow_native_mt "$PPM" > /tmp/c.log 2>&1
    T1=$(date +%s%3N)
    R=$(extract render_ms /tmp/c.log); S=$(extract setup_ms /tmp/c.log)
    echo "E5,C,$g,$sam,$S,$R,$((T1-T0))" >> e5_shadow.csv
    echo "  C g=$g s=$sam wall=$((T1-T0))ms render=${R}ms"
}
run_D_shadow() {
    local g="$1" sam="$2"
    local PPM0="images/E5_D_s0_g${g}_s${sam}.ppm"
    local PPM1="images/E5_D_s1_g${g}_s${sam}.ppm"
    local PPMC="images/E5_D_combined_g${g}_s${sam}.ppm"
    T0=$(date +%s%3N)
    ( OMP_NUM_THREADS=$N_SOCK0 numactl --cpunodebind=0 --membind=0 \
        env RTIOW_SHARD_BAND=1 RTIOW_SHARD_OF=2 RTIOW_SHARD_ID=0 RTIOW_SHARD_AXIS=1 \
            RTIOW_GRID_HALF=$g RTIOW_SAMPLES=$sam RTIOW_MAX_DEPTH=1 \
            RTIOW_SHADOW_ONLY=1 \
        ./rtiow_native_mt "$PPM0" > /tmp/s0.log 2>&1 ) &
    ( OMP_NUM_THREADS=$N_SOCK1 numactl --cpunodebind=1 --membind=1 \
        env RTIOW_SHARD_BAND=1 RTIOW_SHARD_OF=2 RTIOW_SHARD_ID=1 RTIOW_SHARD_AXIS=1 \
            RTIOW_GRID_HALF=$g RTIOW_SAMPLES=$sam RTIOW_MAX_DEPTH=1 \
            RTIOW_SHADOW_ONLY=1 \
        ./rtiow_native_mt "$PPM1" > /tmp/s1.log 2>&1 ) &
    wait
    T1=$(date +%s%3N)
    R0=$(extract render_ms /tmp/s0.log); R1=$(extract render_ms /tmp/s1.log)
    R=$([ "${R0:-0}" -gt "${R1:-0}" ] 2>/dev/null && echo "$R0" || echo "$R1")
    S=$(extract setup_ms /tmp/s0.log)
    echo "E5,D,$g,$sam,$S,$R,$((T1-T0))" >> e5_shadow.csv
    echo "  D g=$g s=$sam wall=$((T1-T0))ms render_max=${R}ms (s0=${R0} s1=${R1})"
    stitch "$PPM0" "$PPM1" "$PPMC"
}
for GRID in 500 1500 3000; do
    for SAM in 16 64; do
        echo "--- E5 GRID=$GRID SAMPLES=$SAM (shadow-only) ---"
        run_C_shadow "$GRID" "$SAM"
        run_D_shadow "$GRID" "$SAM"
    done
done

T_TOTAL_END=$(date +%s)
echo ""
echo "==== ALL DONE in $((T_TOTAL_END - T_TOTAL_START))s ===="
echo "Results in $(pwd)/:"
ls -la *.csv *.txt

echo ""
echo "==== SUMMARY ===="
echo ""
echo "-- E2 pointer-chase (local vs remote latency) --"
awk -F, 'NR>1{printf "  bytes=%s membind=%s  %s ns/chase\n",$3,$2,$6}' e2_pointer_chase.csv

echo ""
echo "-- E3 RTIOW single-thread A=local B=remote (render_ms only) --"
awk -F, 'NR>1{printf "  grid=%-5s %s  render=%sms\n",$3,$2,$7}' e3_AB.csv

echo ""
echo "-- E4 RTIOW multi-thread C vs D (normal scene) --"
awk -F, '
NR>1 {key=$3"_"$4; r[$2"_"key]=$7; s[$2"_"key]=$6; keys[key]=1}
END {
  printf "  %-12s | %-8s | %-12s %-12s | %s\n","GRID_DEPTH","setup","C_render","D_render","D/C"
  for (k in keys) { split(k,p,"_"); g=p[1]; d=p[2]
    c=r["C_"k]; dd=r["D_"k]
    if (c+0>0 && dd+0>0) printf "  g=%-4s d=%-4s | %-8s | %-12s %-12s | %.3fx\n",g,d,s["C_"k],c,dd,dd/c }
}' e4_CD.csv | sort

echo ""
echo "-- E5 RTIOW shadow-only C vs D (memory-bound regime) --"
awk -F, '
NR>1 {key=$3"_"$4; r[$2"_"key]=$6; keys[key]=1}
END {
  printf "  %-15s | %-12s %-12s | %s\n","GRID_SAMPLES","C_render","D_render","D/C"
  for (k in keys) { split(k,p,"_"); g=p[1]; sam=p[2]
    c=r["C_"k]; dd=r["D_"k]
    if (c+0>0 && dd+0>0) printf "  g=%-4s s=%-3s | %-12s %-12s | %.3fx\n",g,sam,c,dd,dd/c }
}' e5_shadow.csv | sort

echo ""
echo "==== Converting PPM -> PNG (if imagemagick available) ===="
if command -v convert >/dev/null 2>&1; then
    N=0
    for ppm in images/*.ppm; do
        png="${ppm%.ppm}.png"
        convert "$ppm" "$png" 2>/dev/null && N=$((N+1))
    done
    echo "  converted $N images to PNG"
else
    echo "  imagemagick 'convert' not found; PPMs only (PNGs ~10x smaller for download)"
    echo "  try: module load imagemagick"
fi

echo ""
echo "==== Packaging results tarball ===="
PARENT=$(dirname $(pwd))
THIS=$(basename $(pwd))
TARBALL="${PARENT}/${THIS}.tar.gz"
tar czf "$TARBALL" -C "$PARENT" "$THIS"
ls -lah "$TARBALL"
echo ""
echo "To download: scp <sherlock>:${TARBALL} ."
