// Stage 5: signed distance field ray marching (sphere tracing) - no triangles, no rasterizer, no RT cores.
// The room is ~20 analytic primitives (half-spaces, rotated boxes, spheres) evaluated per step, plus the
// model as a baked 3D distance texture (see SdfBake.hlsl). Sponza is nothing but one big baked mesh grid.
// Mesh hits are shaded with the exact nearest triangle (stored per voxel by the bake): uv, normal,
// material and sky visibility come from the triangle, only visibility comes from the distance field.
//  - sphere tracing for primary and reflection rays (mirrors, metals, polished floor)
//  - soft shadows from the SDF itself (penumbra estimated from the closest miss distance)
//  - ambient occlusion by sampling the distance field along the normal
//  - glass (any primitive): reflection + refraction ray tree; inside, the marcher follows that
//    primitive's negated distance
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

// Mesh prims: center/size = grid bounds, cosR/sinR = bit-cast SRVs of the distance / triangle-id grids,
// pad.x = bit-cast flags (bit 0 = signed), pad.y = voxel size.
float MeshBoxDistance(SdfPrim pr, float3 p) { return length(max(abs(p - pr.center) - pr.size, 0)); }

// Baked mesh: distance to the grid box + trilinear sample at the closest point in the box. The 0.9 keeps
// sphere tracing conservative (trilinear interpolation can overestimate slightly near thin features).
// Far away just the box distance - the switch happens where it no longer matters for soft shadows.
// Unsigned grids (open architecture) become a thin shell: surfaces are where the distance < 0.6 voxel.
float MeshDistance(SdfPrim pr, float3 p)
{
    float3 bmin = pr.center - pr.size, bmax = pr.center + pr.size;
    float3 q = clamp(p, bmin, bmax);
    float outside = length(p - q);
    if (outside > 1.0) return outside;
    Texture3D<float> sdf = ResourceDescriptorHeap[asuint(pr.cosR)];
    float d = sdf.SampleLevel(sLinear, (q - bmin) / (bmax - bmin), 0);
    if ((asuint(pr.pad.x) & 1) == 0) d -= 0.6 * pr.pad.y;
    return outside + 0.9 * d;
}

float PrimDistance(SdfPrim pr, float3 p)
{
    if (pr.type == SDF_PLANE) return dot(p, pr.size) + pr.center.x;
    if (pr.type == SDF_MESH) return MeshDistance(pr, p);
    float3 w = p - pr.center;
    if (pr.type == SDF_SPHERE) return length(w) - pr.size.x;
    // box rotated around Y: bring p into the box's local frame
    float3 q = float3(w.x * pr.cosR - w.z * pr.sinR, w.y, w.x * pr.sinR + w.z * pr.cosR);
    float3 d = abs(q) - pr.size;
    return length(max(d, 0)) + min(max(d.x, max(d.y, d.z)), 0);
}

#define MAX_PRIMS 64
groupshared SdfPrim gsPrims[MAX_PRIMS];
groupshared uint gsGlass[MAX_PRIMS];   // 1 = the primitive's material is glass
groupshared uint gsShellBits;          // thickest unsigned-grid shell (asuint), 0 without open meshes
#define NO_PRIM 0xFFFFFFFFu

struct MapResult { float d; uint mat; bool mesh; uint prim; };

MapResult Map(float3 p, bool withLights, bool withPlanes = false, bool excludeGlass = false)
{
    MapResult r;
    r.d = 1e9;
    r.mat = 0;
    r.mesh = false;
    r.prim = NO_PRIM;
    uint count = min(rc2, MAX_PRIMS);
    for (uint i = withPlanes ? 0 : rc4; i < count; i++)
    {
        if (excludeGlass && gsGlass[i] != 0) continue;
        bool isMesh = gsPrims[i].type == SDF_MESH;
        if (isMesh && MeshBoxDistance(gsPrims[i], p) >= r.d) continue;   // skip the texture lookup when it can't win
        float d = PrimDistance(gsPrims[i], p);
        if (d < r.d) { r.d = d; r.mat = gsPrims[i].material; r.mesh = isMesh; r.prim = i; }
    }
    if (withLights)
    {
        // only the main bulbs are marched (visible); 128 extra spheres per step would be far too slow
        for (uint li = 0; li < gShadowedLights; li++)
        {
            Light Lt = LoadLight(li);
            float d = length(p - Lt.pos) - Lt.radius;
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
bool March(float3 o, float3 d, out float t, out uint mat, out bool hitPlane, out float3 planeN, out bool hitMesh, out uint prim)
{
    hitMesh = false;
    prim = NO_PRIM;
    float tMax = IntersectPlanes(o, d, mat, planeN);
    hitPlane = false;
    t = 0.0;
    // hit threshold grows with distance ~ half a pixel footprint
    uint lastMat = 0;
    const float tFar = 200;   // open scenes: rays escape to the sky
    for (int i = 0; i < 192 && t < tMax && t < tFar; i++)
    {
        MapResult m = Map(o + d * t, true);
        if (m.d < 1e-4 + 6e-4 * t) { mat = m.mat; hitMesh = m.mesh; prim = m.prim; return true; }
        lastMat = m.mat;
        hitMesh = m.mesh;
        prim = m.prim;
        t += m.d;
    }
    // Out of steps while still in front of the wall = ray grazing an edge: it is right next to that
    // surface, so take it (falling through to the wall behind would draw streaks along edges).
    if (t < tMax && t < tFar) { mat = lastMat; return true; }
    hitMesh = false;
    prim = NO_PRIM;
    if (tMax < 1e9) { t = tMax; hitPlane = true; return true; }
    return false;
}

// Inside a glass primitive: march to the exit using its negated distance.
float MarchInside(uint prim, float3 o, float3 d)
{
    float t = 0.004;
    for (int i = 0; i < 128; i++)
    {
        float h = -PrimDistance(gsPrims[prim], o + d * t);
        if (h < 3e-4) break;
        t += max(h, 0.002);
    }
    return t;
}

float3 MeshNormal(SdfPrim pr, float3 p)
{
    // gradient over about half a voxel: smooths the piecewise-trilinear field
    const float2 k = float2(1, -1);
    float h = 0.5 * pr.pad.y;
    return normalize(k.xyy * MeshDistance(pr, p + k.xyy * h) + k.yyx * MeshDistance(pr, p + k.yyx * h) +
                     k.yxy * MeshDistance(pr, p + k.yxy * h) + k.xxx * MeshDistance(pr, p + k.xxx * h));
}

// Exact surface for a mesh hit: the nearest of the triangles stored in the 8 voxels around p.
bool MeshSurface(SdfPrim pr, float3 p, float3 d, out Surface s)
{
    s = (Surface)0;
    Texture3D<uint> triIds = ResourceDescriptorHeap[asuint(pr.sinR)];
    uint3 dims;
    triIds.GetDimensions(dims.x, dims.y, dims.z);
    float3 bmin = pr.center - pr.size;
    int3 base = (int3)floor((p - bmin) / (2 * pr.size) * float3(dims) - 0.5);
    StructuredBuffer<Vertex> vb = ResourceDescriptorHeap[gVbIndex];
    StructuredBuffer<uint> ib = ResourceDescriptorHeap[gIbIndex];
    float best = 1e30;
    uint bestTri = 0;
    float3 bestCp = p;
    [unroll] for (int k = 0; k < 8; k++)
    {
        int3 v = clamp(base + int3(k & 1, (k >> 1) & 1, k >> 2), 0, int3(dims) - 1);
        uint code = triIds.Load(int4(v, 0));
        if (code == 0) continue;
        uint tri = code - 1;
        float3 a = vb[ib[tri * 3]].pos, b = vb[ib[tri * 3 + 1]].pos, c = vb[ib[tri * 3 + 2]].pos;
        float3 cp = ClosestPointOnTriangle(p, a, b, c);
        float dist = length(cp - p);
        if (dist < best) { best = dist; bestTri = code; bestCp = cp; }
    }
    if (bestTri == 0) return false;
    uint tri = bestTri - 1;
    float3 w = Barycentrics(bestCp, vb[ib[tri * 3]].pos, vb[ib[tri * 3 + 1]].pos, vb[ib[tri * 3 + 2]].pos);
    s = TriangleSurface(tri, saturate(w), d);
    s.pos = bestCp;
    return true;
}

float3 PrimNormal(uint prim, float3 p)
{
    if (gsPrims[prim].type == SDF_MESH) return MeshNormal(gsPrims[prim], p);
    const float2 k = float2(1, -1);
    const float h = 5e-4;
    SdfPrim pr = gsPrims[prim];
    return normalize(k.xyy * PrimDistance(pr, p + k.xyy * h) + k.yyx * PrimDistance(pr, p + k.yyx * h) +
                     k.yxy * PrimDistance(pr, p + k.yxy * h) + k.xxx * PrimDistance(pr, p + k.xxx * h));
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

// ---- culling for the secondary queries: only primitives whose bounding sphere can matter ----
float PrimBoundRadius(SdfPrim pr) { return pr.type == SDF_SPHERE ? pr.size.x : length(pr.size); }

// Bitmask (prims 0..63) of the objects whose bounds come within `margin` of the segment o + L*[0, maxT].
uint2 PrimsNearSegment(float3 o, float3 L, float maxT, float margin, bool excludeGlass)
{
    uint2 mask = 0;
    uint count = min(rc2, MAX_PRIMS);
    for (uint i = rc4; i < count; i++)
    {
        if (excludeGlass && gsGlass[i] != 0) continue;
        SdfPrim pr = gsPrims[i];
        float3 oc = pr.center - o;
        float tc = clamp(dot(oc, L), 0, maxT);
        if (length(oc - L * tc) < PrimBoundRadius(pr) + margin)
            mask[i >> 5] |= 1u << (i & 31);
    }
    return mask;
}

uint2 PrimsNearPoint(float3 p, float radius)
{
    uint2 mask = 0;
    uint count = min(rc2, MAX_PRIMS);
    for (uint i = rc4; i < count; i++)
        if (length(gsPrims[i].center - p) < PrimBoundRadius(gsPrims[i]) + radius)
            mask[i >> 5] |= 1u << (i & 31);
    return mask;
}

float MapMasked(float3 p, uint2 mask)
{
    float d = 1e9;
    [unroll] for (uint w = 0; w < 2; w++)
    {
        uint m = mask[w];
        while (m != 0)
        {
            uint i = firstbitlow(m) + 32 * w;
            m &= m - 1;
            d = min(d, PrimDistance(gsPrims[i], p));
        }
    }
    return d;
}

// Soft shadow (Quilez, improved): the closest a shadow ray passes to any surface, relative to the
// distance travelled, approximates how much of the light's disk is covered. An object farther than maxT/k from the segment can't
// darken the penumbra (res = k*h/t >= 1), so only nearby objects are evaluated - usually none or one.
float SoftShadow(float3 o, float3 L, float maxT, float k, bool excludeGlass = false)
{
    uint2 mask = PrimsNearSegment(o, L, maxT, maxT / k + 0.02, excludeGlass);
    if (all(mask == 0)) return 1.0;
    float res = 1.0;
    float t = 0.02;   // start a bit away so a surface lit at grazing angles doesn't shadow itself
    float ph = 1e10;
    for (int i = 0; i < 64 && t < maxT; i++)
    {
        float h = MapMasked(o + L * t, mask);
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
    float shell = asfloat(gsShellBits);
    uint2 mask = PrimsNearPoint(p, 0.25 + shell);
    float occ = 0.0;
    float sca = 1.0;
    for (int i = 0; i < 5; i++)
    {
        float h = 0.02 + 0.15 * i / 4.0;
        float3 q = p + n * (h + shell);
        float d = MapMasked(q, mask) + shell;   // unsigned grids read 'distance - shell'
        for (uint j = 0; j < rc4; j++) d = min(d, PrimDistance(gsPrims[j], q));   // room half-spaces
        occ += (h - d) * sca;
        sca *= 0.9;
    }
    return saturate(1.0 - 2.5 * occ);
}

// Soft shadow towards a light; glass casts a tinted, partial shadow instead of a black one
// (same cheat as the Whitted ray tracer).
float3 LightShadow(float3 p, float3 N, uint li)
{
    // start outside the thin shell of unsigned (open mesh) grids, or the ray is 'inside' from the start
    float3 P = p + N * (0.01 + 1.5 * asfloat(gsShellBits));
    Light Lt = LoadLight(li);
    float3 Lv = Lt.pos - P;
    float dist = length(Lv);
    float3 L = Lv / dist;
    float maxT = dist - Lt.radius;
    // penumbra sharpness ~ 1 / angular size of the bulb
    float k = clamp(0.5 * dist / Lt.radius, 2.0, 64.0);
    float opaque = SoftShadow(P, L, maxT, k, true);
    if (opaque <= 0) return 0;

    // glass-only march: every glass object crossed lets 65% of its tint through.
    // Most shadow rays miss every glass object's bounding sphere - skip the march for those.
    uint count = min(rc2, MAX_PRIMS);
    bool nearGlass = false;
    for (uint g = rc4; g < count; g++)
    {
        if (gsGlass[g] == 0) continue;
        SdfPrim pr = gsPrims[g];
        float radius = pr.type == SDF_SPHERE ? pr.size.x : length(pr.size);
        float3 oc = pr.center - P;
        float tc = clamp(dot(oc, L), 0, maxT);
        if (length(oc - L * tc) < radius) { nearGlass = true; break; }
    }
    if (!nearGlass) return opaque;

    float3 T = 1;
    float t = 0.02;
    uint lastGlass = NO_PRIM;
    for (int i = 0; i < 64 && t < maxT; i++)
    {
        float h = 1e9;
        uint hp = NO_PRIM;
        for (uint j = rc4; j < count; j++)
        {
            if (gsGlass[j] == 0 || j == lastGlass) continue;
            float dj = PrimDistance(gsPrims[j], P + L * t);
            if (dj < h) { h = dj; hp = j; }
        }
        if (hp == NO_PRIM) break;
        if (h < 1e-3)
        {
            T *= LoadMaterial(gsPrims[hp].material).albedo * 0.65;
            lastGlass = hp;   // don't count the same object twice while inside it
            t += 0.05;
            continue;
        }
        if (lastGlass != NO_PRIM && PrimDistance(gsPrims[lastGlass], P + L * t) > 0) lastGlass = NO_PRIM;
        t += max(h, 0.01);
    }
    return opaque * T;
}

// Same Whitted ray tree as the ray tracing mode (explicit stack), with SDF marching instead of RayQuery.
struct RayTask
{
    float3 o;
    float3 d;
    float3 throughput;
    uint depth;
    uint inside;   // glass primitive the ray travels inside of, or NO_PRIM
    float dist;    // distance travelled so far (ray cone width for texture LOD)
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
    first.inside = NO_PRIM;
    first.dist = 0;
    stack[sp++] = first;
    float3 radiance = 0;

    for (int n = 0; n < MAX_RAYS && sp > 0; n++)
    {
        RayTask r = stack[--sp];
        float3 o = r.o, d = r.d;
        float t;
        uint mat;
        bool onMesh;
        uint prim = r.inside;
        bool front = r.inside == NO_PRIM;

        if (r.inside != NO_PRIM)
        {
            t = MarchInside(r.inside, o, d);
            mat = gsPrims[r.inside].material;
            onMesh = gsPrims[r.inside].type == SDF_MESH;
        }
        else
        {
            bool onPlane;
            float3 planeN;
            if (!March(o, d, t, mat, onPlane, planeN, onMesh, prim))
            {
                radiance += r.throughput * SkyRadiance(d);
                continue;
            }
            if (mat >= LIGHT_MAT)
            {
                radiance += r.throughput * LightEmission(mat - LIGHT_MAT);
                continue;
            }
            if (onPlane) prim = NO_PRIM;
        }

        // surface: exact triangle data for mesh hits, analytic otherwise
        float3 p = o + d * t;
        Surface s = (Surface)0;
        s.pos = p;
        s.material = mat;
        s.skyVis = 1;
        bool haveTri = onMesh && prim != NO_PRIM && MeshSurface(gsPrims[prim], p, d, s);
        if (!haveTri)
        {
            float3 N;
            if (prim == NO_PRIM) { uint pm; float3 pn; IntersectPlanes(o, d, pm, pn); N = pn; }
            else N = PrimNormal(prim, p);
            s.N = dot(N, d) > 0 ? -N : N;
        }
        if (!front) s.N = dot(s.N, d) > 0 ? -s.N : s.N;   // inside the glass: normal faces the ray too
        float dist = r.dist + t;
        Material m = LoadMaterial(s.material);
        float3 albedo = haveTri ? MaterialAlbedoLod(m, s.pos, s.uv, TextureLod(m, s, dist * gPixelSpread, d)) : MaterialAlbedo(m, s.pos);
        float3 N = s.N;
        p = s.pos;
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
            c.dist = dist;
            if (F < 1.0 && sp < MAX_STACK)
            {
                c.o = p - N * 2e-3;
                c.d = refract(d, N, eta);
                c.throughput = thr * (1.0 - F);
                c.inside = front ? prim : NO_PRIM;   // entering -> now inside, leaving -> outside
                stack[sp++] = c;
            }
            if (sp < MAX_STACK && max(thr.r, max(thr.g, thr.b)) * F > 0.01)
            {
                c.o = p + N * 2e-3;
                c.d = reflect(d, N);
                c.throughput = thr * F;
                c.inside = front ? NO_PRIM : prim;  // internal reflection stays inside
                stack[sp++] = c;
            }
            continue;
        }

        bool reflective = (m.roughness < 0.1 || m.metallic > 0.5) && canRecurse;
        float3 direct = 0;
        for (uint li = 0; li < gNumLights; li++)
        {
            float3 c = ShadePointLight(li, p, N, V, albedo, m, !reflective);
            if (any(c > 0) && li < gShadowedLights) c *= LightShadow(p, N, li);   // the many small lights: unshadowed
            direct += c;
        }
        // ambient x SDF ambient occlusion (the room's constant fill is boosted so the AO is visible)
        float3 ambient = 2.5 * gAmbient + SkyAmbient(N) * SkyOcclusion(s.skyVis);
        direct += ambient * albedo * (1.0 - m.metallic) * AmbientOcclusion(p, N);
        radiance += r.throughput * direct;

        if (reflective && sp < MAX_STACK)
        {
            float3 F = F_Schlick(SpecularF0(albedo, m.metallic), saturate(dot(N, V)));
            RayTask c;
            c.o = p + N * 2e-3;
            c.d = reflect(d, N);
            c.throughput = r.throughput * F;
            c.depth = r.depth + 1;
            c.inside = NO_PRIM;
            c.dist = dist;
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
    if (gi == 0) gsShellBits = 0;
    GroupMemoryBarrierWithGroupSync();
    if (gi < min(rc2, MAX_PRIMS))
    {
        SdfPrim pr = prims[gi];
        gsPrims[gi] = pr;
        gsGlass[gi] = LoadMaterial(pr.material).type == MAT_GLASS ? 1u : 0u;
        if (pr.type == SDF_MESH && (asuint(pr.pad.x) & 1) == 0)
            InterlockedMax(gsShellBits, asuint(0.6 * pr.pad.y));   // positive floats order like uints
    }
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
