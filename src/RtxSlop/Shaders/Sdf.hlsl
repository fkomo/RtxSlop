// Stage 5: signed distance field ray marching (sphere tracing) - no triangles, no rasterizer, no RT cores.
// The room is ~20 analytic primitives (half-spaces, rotated boxes, spheres) evaluated per step.
//  - sphere tracing for primary and reflection rays (mirrors, metals, polished floor)
//  - soft shadows from the SDF itself (penumbra estimated from the closest miss distance)
//  - ambient occlusion by sampling the distance field along the normal
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
#define LIGHT_MAT  0x10000u      // material ids >= this are light bulb (LIGHT_MAT + index)

float PrimDistance(SdfPrim pr, float3 p)
{
    if (pr.type == SDF_PLANE) return dot(p, pr.size) + pr.center.x;
    float3 w = p - pr.center;
    if (pr.type == SDF_SPHERE) return length(w) - pr.size.x;
    // box rotated around Y: bring p into the box's local frame
    float3 q = float3(w.x * pr.cosR - w.z * pr.sinR, w.y, w.x * pr.sinR + w.z * pr.cosR);
    float3 d = abs(q) - pr.size;
    return length(max(d, 0)) + min(max(d.x, max(d.y, d.z)), 0);
}

#define MAX_PRIMS 64
groupshared SdfPrim gsPrims[MAX_PRIMS];

struct MapResult { float d; uint mat; };

MapResult Map(float3 p, bool withLights, bool withPlanes = false)
{
    MapResult r;
    r.d = 1e9;
    r.mat = 0;
    uint count = min(rc2, MAX_PRIMS);
    for (uint i = withPlanes ? 0 : rc4; i < count; i++)
    {
        float d = PrimDistance(gsPrims[i], p);
        if (d < r.d) { r.d = d; r.mat = gsPrims[i].material; }
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
bool March(float3 o, float3 d, out float t, out uint mat, out bool hitPlane, out float3 planeN)
{
    float tMax = IntersectPlanes(o, d, mat, planeN);
    hitPlane = false;
    t = 0.0;
    // hit threshold grows with distance ~ half a pixel footprint
    uint lastMat = 0;
    for (int i = 0; i < 160 && t < tMax; i++)
    {
        MapResult m = Map(o + d * t, true);
        if (m.d < 1e-4 + 6e-4 * t) { mat = m.mat; return true; }
        lastMat = m.mat;
        t += m.d;
    }
    // Out of steps while still in front of the wall = ray grazing an edge: it is right next to that
    // surface, so take it (falling through to the wall behind would draw streaks along edges).
    if (t < tMax) { mat = lastMat; return true; }
    if (tMax < 1e9) { t = tMax; hitPlane = true; return true; }
    return false;
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
float SoftShadow(float3 o, float3 L, float maxT, float k)
{
    float res = 1.0;
    float t = 0.02;   // start a bit away so a surface lit at grazing angles doesn't shadow itself
    float ph = 1e10;
    for (int i = 0; i < 64 && t < maxT; i++)
    {
        float h = Map(o + L * t, false).d;
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

float3 TracePixel(float2 pixel)
{
    float3 o = gCamPos;
    float3 d = CameraRayDir(pixel);
    float3 radiance = 0;
    float3 throughput = 1;

    for (uint bounce = 0; bounce <= gMaxBounces; bounce++)
    {
        float t;
        uint mat;
        bool onPlane;
        float3 planeN;
        if (!March(o, d, t, mat, onPlane, planeN)) break;
        if (mat >= LIGHT_MAT)
        {
            radiance += throughput * LightEmission(mat - LIGHT_MAT);
            break;
        }

        float3 p = o + d * t;
        float3 N = onPlane ? planeN : SdfNormal(p);
        if (dot(N, d) > 0) N = -N;
        Material m = LoadMaterial(mat);
        float3 albedo = MaterialAlbedo(m, p);
        float3 V = -d;
        bool reflective = (m.roughness < 0.1 || m.metallic > 0.5) && bounce < gMaxBounces;
        float3 P = p + N * 2e-3;

        float3 direct = 0;
        for (uint li = 0; li < gNumLights; li++)
        {
            float3 c = ShadePointLight(li, p, N, V, albedo, m, !reflective);
            if (any(c > 0))
            {
                float3 Lv = gLights[li].pos - P;
                float dist = length(Lv);
                // penumbra sharpness ~ 1 / angular size of the bulb
                float k = clamp(0.5 * dist / gLights[li].radius, 2.0, 64.0);
                c *= SoftShadow(p + N * 0.01, Lv / dist, dist - gLights[li].radius, k);
            }
            direct += c;
        }
        // stronger ambient than the other modes so the SDF ambient occlusion is visible
        direct += 2.5 * gAmbient * albedo * (1.0 - m.metallic) * AmbientOcclusion(p, N);
        radiance += throughput * direct;

        if (!reflective) break;
        float3 F = F_Schlick(SpecularF0(albedo, m.metallic), saturate(dot(N, V)));
        throughput *= F;
        if (max(throughput.r, max(throughput.g, throughput.b)) < 0.01) break;
        o = P;
        d = reflect(d, N);
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
