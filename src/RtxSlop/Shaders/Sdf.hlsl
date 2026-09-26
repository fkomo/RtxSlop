// Stage 5: signed distance field ray marching (sphere tracing) - no triangles, no rasterizer, no RT cores.
// The room is ~20 analytic primitives (half-spaces, rotated boxes, spheres) evaluated per step, plus the
// loaded mesh as a baked 3D distance texture (see SdfBake.hlsl).
//  - sphere tracing for primary and reflection rays (mirrors, metals, polished floor)
//  - soft shadows from the SDF itself (penumbra estimated from the closest miss distance)
//  - ambient occlusion by sampling the distance field along the normal
//  - glass: reflection + refraction ray tree; inside the glass the marcher follows the negated distance
// Performance notes: the 6 room half-spaces are the first primitives. Primary/reflection rays intersect
// them analytically and only march the objects - otherwise rays grazing the floor/walls crawl along in
// tiny steps (the classic sphere-tracing worst case). Shadows ignore them (walls can't occlude the room).
// Primitives are cached in groupshared memory once per thread group.
// Root constants: rc0 = HDR UAV, rc1 = primitive buffer SRV, rc2 = primitive count, rc3 = rays per pixel,
//                 rc4 = number of leading half-space primitives

#include "Common.hlsli"

struct SdfPrim
{
    float3 center; uint type;     // plane: center.x = d
    float3 size;   uint material; // box: half extents, sphere: x = radius, plane: normal
    float cosR;    float sinR;    float2 pad;
};

#define SDF_PLANE  0u
#define SDF_BOX    1u
#define SDF_SPHERE 2u
#define SDF_MESH   3u
#define LIGHT_MAT  0x10000u      // material ids >= this are light bulb (LIGHT_MAT + index)

float MeshBoxDistance(float3 p) { return length(p - clamp(p, gMeshSdfMin, gMeshSdfMax)); }

// Baked mesh: distance to the grid box + trilinear sample at the closest point in the box. The 0.9 keeps
// sphere tracing conservative (trilinear interpolation can overestimate slightly near thin features).
// Far away just the box distance - the switch happens where it no longer matters for soft shadows.
float MeshDistance(float3 p)
{
    float3 q = clamp(p, gMeshSdfMin, gMeshSdfMax);
    float outside = length(p - q);
    if (outside > 1.0) return outside;
    Texture3D<float> sdf = ResourceDescriptorHeap[gMeshSdfIndex];
    float d = sdf.SampleLevel(sLinear, (q - gMeshSdfMin) / (gMeshSdfMax - gMeshSdfMin), 0);
    return outside + 0.9 * d;
}

float PrimDistance(SdfPrim pr, float3 p)
{
    if (pr.type == SDF_PLANE) return dot(p, pr.size) + pr.center.x;
    if (pr.type == SDF_MESH) return MeshDistance(p);
    float3 w = p - pr.center;
    if (pr.type == SDF_SPHERE) return length(w) - pr.size.x;
    // box rotated around Y: bring p into the box's local frame
    float3 q = float3(w.x * pr.cosR - w.z * pr.sinR, w.y, w.x * pr.sinR + w.z * pr.cosR);
    float3 d = abs(q) - pr.size;
    return length(max(d, 0)) + min(max(d.x, max(d.y, d.z)), 0);
}

#define MAX_PRIMS 64
groupshared SdfPrim gsPrims[MAX_PRIMS];

struct MapResult { float d; uint mat; bool mesh; };

#define MESH_INCLUDE 0u
#define MESH_EXCLUDE 1u

MapResult Map(float3 p, bool withLights, bool withPlanes = false, uint meshMode = MESH_INCLUDE)
{
    MapResult r;
    r.d = 1e9;
    r.mat = 0;
    r.mesh = false;
    uint count = min(rc2, MAX_PRIMS);
    for (uint i = withPlanes ? 0 : rc4; i < count; i++)
    {
        bool isMesh = gsPrims[i].type == SDF_MESH;
        if (isMesh && (meshMode == MESH_EXCLUDE || MeshBoxDistance(p) >= r.d)) continue;   // skip the texture when it can't win
        float d = PrimDistance(gsPrims[i], p);
        if (d < r.d) { r.d = d; r.mat = gsPrims[i].material; r.mesh = isMesh; }
    }
    if (withLights)
    {
        for (uint li = 0; li < gNumLights; li++)
        {
            float d = length(p - gLights[li].pos) - gLights[li].radius;
            if (d < r.d) { r.d = d; r.mat = LIGHT_MAT + li; }
        }
    }
    return r;
}

// Analytic hit with the room shell (inside of the half-spaces).
float IntersectPlanes(float3 o, float3 d, out uint mat, out float3 n)
{
    float tBest = 1e9;
    mat = 0;
    n = float3(0, 1, 0);
    for (uint i = 0; i < rc4; i++)
    {
        SdfPrim pr = gsPrims[i];
        float dn = dot(d, pr.size);
        if (dn >= 0) continue;
        float t = -(dot(o, pr.size) + pr.center.x) / dn;
        if (t > 0 && t < tBest) { tBest = t; mat = pr.material; n = pr.size; }
    }
    return tBest;
}

// Sphere tracing against the objects, capped by the analytic room hit.
bool March(float3 o, float3 d, out float t, out uint mat, out bool hitPlane, out float3 planeN, out bool hitMesh)
{
    hitMesh = false;
    float tMax = IntersectPlanes(o, d, mat, planeN);
    hitPlane = false;
    t = 0.0;
    // hit threshold grows with distance ~ half a pixel footprint
    uint lastMat = 0;
    for (int i = 0; i < 160 && t < tMax; i++)
    {
        MapResult m = Map(o + d * t, true);
        if (m.d < 1e-4 + 6e-4 * t) { mat = m.mat; hitMesh = m.mesh; return true; }
        lastMat = m.mat;
        hitMesh = m.mesh;
        t += m.d;
    }
    // Out of steps while still in front of the wall = ray grazing an edge: it is right next to that
    // surface, so take it (falling through to the wall behind would draw streaks along edges).
    if (t < tMax) { mat = lastMat; return true; }
    hitMesh = false;
    if (tMax < 1e9) { t = tMax; hitPlane = true; return true; }
    return false;
}

// Inside the glass: march to the exit using the negated mesh distance.
float MarchInside(float3 o, float3 d)
{
    float t = 0.004;
    for (int i = 0; i < 128; i++)
    {
        float h = -MeshDistance(o + d * t);
        if (h < 3e-4) break;
        t += max(h, 0.002);
    }
    return t;
}

float3 MeshNormal(float3 p)
{
    // gradient over about half a voxel: smooths the piecewise-trilinear field
    float3 voxel = (gMeshSdfMax - gMeshSdfMin) / 128.0;
    const float2 k = float2(1, -1);
    float h = 0.5 * max(voxel.x, max(voxel.y, voxel.z));
    return normalize(k.xyy * MeshDistance(p + k.xyy * h) + k.yyx * MeshDistance(p + k.yyx * h) +
                     k.yxy * MeshDistance(p + k.yxy * h) + k.xxx * MeshDistance(p + k.xxx * h));
}

float3 SdfNormal(float3 p)
{
    // tetrahedron technique: 4 taps instead of 6
    const float2 k = float2(1, -1);
    const float h = 5e-4;
    return normalize(k.xyy * Map(p + k.xyy * h, false).d +
                     k.yyx * Map(p + k.yyx * h, false).d +
                     k.yxy * Map(p + k.yxy * h, false).d +
                     k.xxx * Map(p + k.xxx * h, false).d);
}

// Soft shadow (Quilez, improved): the closest a shadow ray passes to any surface, relative to the
// distance travelled, approximates how much of the light's disk is covered.
float SoftShadow(float3 o, float3 L, float maxT, float k, uint meshMode = MESH_INCLUDE)
{
    float res = 1.0;
    float t = 0.02;   // start a bit away so a surface lit at grazing angles doesn't shadow itself
    float ph = 1e10;
    for (int i = 0; i < 64 && t < maxT; i++)
    {
        float h = Map(o + L * t, false, false, meshMode).d;
        if (h < 1e-4) return 0.0;
        float y = h * h / (2.0 * ph);
        float dd = sqrt(max(0.0, h * h - y * y));
        res = min(res, k * dd / max(1e-4, t - y));
        ph = h;
        t += clamp(h, 0.015, 0.5);
    }
    res = saturate(res);
    return res * res * (3.0 - 2.0 * res);
}

float AmbientOcclusion(float3 p, float3 n)
{
    float occ = 0.0;
    float sca = 1.0;
    for (int i = 0; i < 5; i++)
    {
        float h = 0.02 + 0.15 * i / 4.0;
        float d = Map(p + n * h, false, true).d;
        occ += (h - d) * sca;
        sca *= 0.9;
    }
    return saturate(1.0 - 2.5 * occ);
}

bool MeshIsGlassSdf() { return gMeshMaterial != NO_MESH && LoadMaterial(gMeshMaterial).type == MAT_GLASS; }

// Soft shadow towards a light; a glass mesh casts a tinted, partial shadow instead of a black one
// (same cheat as the Whitted ray tracer).
float3 LightShadow(float3 p, float3 N, uint li)
{
    float3 P = p + N * 0.01;
    float3 Lv = gLights[li].pos - P;
    float dist = length(Lv);
    float3 L = Lv / dist;
    float maxT = dist - gLights[li].radius;
    // penumbra sharpness ~ 1 / angular size of the bulb
    float k = clamp(0.5 * dist / gLights[li].radius, 2.0, 64.0);
    if (!MeshIsGlassSdf()) return SoftShadow(P, L, maxT, k);
    float opaque = SoftShadow(P, L, maxT, k, MESH_EXCLUDE);
    if (opaque <= 0) return 0;
    // mesh-only visibility: the glass lets 65% of its tint through
    float t = 0.02, glassVis = 1;
    for (int i = 0; i < 48 && t < maxT; i++)
    {
        float h = MeshDistance(P + L * t);
        if (h < 1e-3) { glassVis = 0; break; }
        t += max(h, 0.01);
    }
    float3 tint = LoadMaterial(gMeshMaterial).albedo * 0.65;
    return opaque * lerp(tint, 1.0, glassVis);
}

// Same Whitted ray tree as the ray tracing mode (explicit stack), with SDF marching instead of RayQuery.
struct RayTask
{
    float3 o;
    float3 d;
    float3 throughput;
    uint depth;
    bool inside;   // travelling inside the glass mesh
};

#define MAX_STACK 8
#define MAX_RAYS 32

float3 TracePixel(float2 pixel)
{
    RayTask stack[MAX_STACK];
    int sp = 0;
    RayTask first;
    first.o = gCamPos;
    first.d = CameraRayDir(pixel);
    first.throughput = 1;
    first.depth = 0;
    first.inside = false;
    stack[sp++] = first;
    float3 radiance = 0;

    for (int n = 0; n < MAX_RAYS && sp > 0; n++)
    {
        RayTask r = stack[--sp];
        float3 o = r.o, d = r.d;
        float t;
        uint mat;
        float3 N;
        bool front = true;
        bool onMesh;

        if (r.inside)
        {
            t = MarchInside(o, d);
            N = -MeshNormal(o + d * t);   // outward normal flipped to face the ray
            mat = gMeshMaterial;
            front = false;
            onMesh = true;
        }
        else
        {
            bool onPlane;
            float3 planeN;
            if (!March(o, d, t, mat, onPlane, planeN, onMesh)) continue;
            if (mat >= LIGHT_MAT)
            {
                radiance += r.throughput * LightEmission(mat - LIGHT_MAT);
                continue;
            }
            float3 hp = o + d * t;
            N = onPlane ? planeN : (onMesh ? MeshNormal(hp) : SdfNormal(hp));
            if (dot(N, d) > 0) N = -N;
        }

        float3 p = o + d * t;
        Material m = LoadMaterial(mat);
        float3 albedo = MaterialAlbedo(m, p);
        float3 V = -d;
        // glass gets extra depth: thin parts bounce light around inside a few times
        bool canRecurse = r.depth < gMaxBounces + (m.type == MAT_GLASS ? 4u : 0u);

        if (m.type == MAT_GLASS)
        {
            float3 thr = r.throughput;
            if (!front) thr *= GlassAbsorption(albedo, t);
            if (!canRecurse) continue;
            float eta = front ? 1.0 / m.ior : m.ior;
            float F = FresnelDielectric(saturate(dot(N, V)), eta);
            RayTask c;
            c.depth = r.depth + 1;
            if (F < 1.0 && sp < MAX_STACK)
            {
                c.o = p - N * 2e-3;
                c.d = refract(d, N, eta);
                c.throughput = thr * (1.0 - F);
                c.inside = front;   // entering -> now inside, leaving -> outside
                stack[sp++] = c;
            }
            if (sp < MAX_STACK && max(thr.r, max(thr.g, thr.b)) * F > 0.01)
            {
                c.o = p + N * 2e-3;
                c.d = reflect(d, N);
                c.throughput = thr * F;
                c.inside = !front;  // internal reflection stays inside
                stack[sp++] = c;
            }
            continue;
        }

        bool reflective = (m.roughness < 0.1 || m.metallic > 0.5) && canRecurse;
        float3 direct = 0;
        for (uint li = 0; li < gNumLights; li++)
        {
            float3 c = ShadePointLight(li, p, N, V, albedo, m, !reflective);
            if (any(c > 0)) c *= LightShadow(p, N, li);
            direct += c;
        }
        // stronger ambient than the other modes so the SDF ambient occlusion is visible
        direct += 2.5 * gAmbient * albedo * (1.0 - m.metallic) * AmbientOcclusion(p, N);
        radiance += r.throughput * direct;

        if (reflective && sp < MAX_STACK)
        {
            float3 F = F_Schlick(SpecularF0(albedo, m.metallic), saturate(dot(N, V)));
            RayTask c;
            c.o = p + N * 2e-3;
            c.d = reflect(d, N);
            c.throughput = r.throughput * F;
            c.depth = r.depth + 1;
            c.inside = false;
            if (max(c.throughput.r, max(c.throughput.g, c.throughput.b)) > 0.01)
                stack[sp++] = c;
        }
    }
    return radiance;
}

static const float2 kRotatedGrid[4] = { float2(0.375, 0.125), float2(0.875, 0.375), float2(0.125, 0.625), float2(0.625, 0.875) };

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void CS_Sdf(uint2 id : SV_DispatchThreadID, uint gi : SV_GroupIndex)
{
    StructuredBuffer<SdfPrim> prims = ResourceDescriptorHeap[rc1];
    if (gi < min(rc2, MAX_PRIMS)) gsPrims[gi] = prims[gi];
    GroupMemoryBarrierWithGroupSync();
    if (any(id >= (uint2)gScreenSize)) return;
    RWTexture2D<float4> output = ResourceDescriptorHeap[rc0];
    float3 c = 0;
    if (rc3 >= 4)
    {
        float wsum = 0;
        [unroll] for (int i = 0; i < 4; i++)
        {
            float3 r = TracePixel(id + kRotatedGrid[i]);
            float w = 1.0 / (1.0 + Luminance(r));
            c += r * w;
            wsum += w;
        }
        c /= wsum;
    }
    else
    {
        c = TracePixel(id + 0.5);
    }
    output[id] = float4(c, 1);
}
