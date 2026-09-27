// ReSTIR DI (reservoir-based spatio-temporal importance resampling, Bitterli et al. 2020) for the
// direct light of the path tracer's primary hits. Two passes before CS_PathTrace:
//  1) CS_RestirInitial: primary surface (traced with the same jitter as the path tracer's first sample,
//     or read from the hybrid G-buffer), 32 light candidates resampled into one (RIS), visibility test,
//     then merged with last frame's reservoir of the reprojected pixel (temporal reuse, M capped).
//  2) CS_RestirSpatial: merges the reservoirs of a few random neighbours on a similar surface, re-tests
//     the visibility of the winner and writes this frame's final reservoir - the path tracer shades
//     the primary hit with that one sample (f * Le * G * W) instead of NEE.
// A light sample is (light index, unit direction from the light's center to the point on its sphere),
// so it stays on the light when the light moves. Reuse is weighted with the generalized balance heuristic;
// the remaining bias comes from visibility (a neighbour's sample may be blocked here).
//
// Resource sets (consecutive bindless slots): set base + 0..3 = SRVs of reservoir A (dir.xyz, W),
// reservoir B (M, light index bits), surface A (pos.xyz, view distance; 0 = none), surface B
// (octahedral normal.xy, albedo rgb8 + roughness8 bits, metallic); base + 4..7 = the matching UAVs.
// Root constants: rc0 = this frame's set, rc1 = last frame's set, rc2 = temporary set (initial+temporal
//                 result), rc3/rc4 = hybrid G-buffer SRVs or NO_GBUFFER, rc5 = 1: no history

#include "PathTrace.hlsl"   // scene intersection, G-buffer decoding, sphere light sampling

static const uint RestirCandidates = 32;
static const float RestirHistoryCap = 20.0 * RestirCandidates;

struct RestirSurface
{
    float3 pos;       // offset from the surface (shadow ray origin)
    float3 N;
    float3 V;
    float3 albedo;
    float roughness;
    float metallic;
    float dist;       // camera distance, 0 = no ReSTIR here
};

struct Reservoir
{
    uint light;
    float3 dir;       // point on the light = light.pos + dir * radius
    float wSum;
    float M;
    float W;
    float pHat;
};

float2 OctEncode(float3 n)
{
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    float2 e = n.xy;
    if (n.z < 0) e = (1 - abs(n.yx)) * float2(e.x >= 0 ? 1 : -1, e.y >= 0 ? 1 : -1);
    return e;
}

float3 OctDecode(float2 e)
{
    float3 n = float3(e, 1 - abs(e.x) - abs(e.y));
    if (n.z < 0) n.xy = (1 - abs(n.yx)) * float2(n.x >= 0 ? 1 : -1, n.y >= 0 ? 1 : -1);
    return normalize(n);
}

uint PackUnorm4(float4 v)
{
    uint4 u = (uint4)round(saturate(v) * 255);
    return u.x | (u.y << 8) | (u.z << 16) | (u.w << 24);
}

float4 UnpackUnorm4(uint u) { return float4(u & 255, (u >> 8) & 255, (u >> 16) & 255, u >> 24) / 255.0; }

RestirSurface LoadSurface(uint set, uint2 p)
{
    Texture2D<float4> sa = ResourceDescriptorHeap[set + 2];
    Texture2D<float4> sb = ResourceDescriptorHeap[set + 3];
    float4 a = sa[p];
    float4 b = sb[p];
    RestirSurface s;
    s.pos = a.xyz;
    s.dist = a.w;
    s.N = OctDecode(b.xy);
    float4 c = UnpackUnorm4(asuint(b.z));
    s.albedo = c.rgb;
    s.roughness = c.a;
    s.metallic = b.w;
    s.V = normalize(gCamPos - s.pos);
    return s;
}

Reservoir LoadReservoir(uint set, uint2 p)
{
    Texture2D<float4> ra = ResourceDescriptorHeap[set + 0];
    Texture2D<float4> rb = ResourceDescriptorHeap[set + 1];
    float4 a = ra[p];
    float4 b = rb[p];
    Reservoir r = (Reservoir)0;
    r.dir = a.xyz;
    r.W = a.w;
    r.M = b.x;
    r.light = asuint(b.y);
    return r;
}

void StoreReservoir(uint set, uint2 p, Reservoir r)
{
    RWTexture2D<float4> ra = ResourceDescriptorHeap[set + 4];
    RWTexture2D<float4> rb = ResourceDescriptorHeap[set + 5];
    ra[p] = float4(r.dir, r.W);
    rb[p] = float4(r.M, asfloat(r.light), 0, 0);
}

// Target function: unshadowed contribution of the light point to the surface's outgoing radiance.
float TargetPdf(RestirSurface s, uint li, float3 dir)
{
    if (li >= gNumLights) return 0;
    Light Lt = LoadLight(li);
    float3 wi = Lt.pos + dir * Lt.radius - s.pos;
    float d2 = dot(wi, wi);
    wi *= rsqrt(d2);
    float cosX = dot(s.N, wi), cosY = -dot(dir, wi);
    if (cosX <= 0 || cosY <= 0) return 0;
    float3 f = EvalBRDF(s.N, s.V, wi, s.albedo, s.roughness, s.metallic, 0);
    return Luminance(f * LightEmission(li)) * cosX * cosY / d2;
}

bool SampleVisible(RestirSurface s, uint li, float3 dir)
{
    Light Lt = LoadLight(li);
    float3 wi = Lt.pos + dir * Lt.radius - s.pos;
    float d = length(wi);
    return TraceVisible(s.pos, wi / d, d - 2e-3);
}

// Merges the reservoirs of several pixels into one for surface s[0] with the generalized balance heuristic:
// sample i gets MIS weight M_i p_i(y) / sum_j M_j p_j(y), where p_j is the target function of pixel j.
// Unlike plain 1/M weighting this stays unbiased when the pixels' target functions differ (close lights,
// different normals / view vectors); only visibility differences between the pixels remain as bias.
#define MAX_MERGE 6

Reservoir CombineMis(RestirSurface s[MAX_MERGE], Reservoir c[MAX_MERGE], uint n, inout uint seed)
{
    // unrolled with constant indices so the arrays stay in registers
    Reservoir r = (Reservoir)0;
    [unroll] for (uint i = 0; i < MAX_MERGE; i++)
    {
        if (i >= n) break;
        r.M += c[i].M;
        if (c[i].W <= 0) continue;
        float pc = TargetPdf(s[0], c[i].light, c[i].dir);
        if (pc <= 0) continue;
        float num = 0, den = 0;
        [unroll] for (uint j = 0; j < MAX_MERGE; j++)
        {
            if (j >= n) break;
            float pj = c[j].M <= 0 ? 0 : c[j].M * (j == 0 ? pc : TargetPdf(s[j], c[i].light, c[i].dir));
            den += pj;
            if (j == i) num = pj;
        }
        float w = (num / den) * pc * c[i].W;
        r.wSum += w;
        if (Rand(seed) * r.wSum < w) { r.light = c[i].light; r.dir = c[i].dir; r.pHat = pc; }
    }
    r.W = r.pHat > 0 ? r.wSum / r.pHat : 0;
    return r;
}

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void CS_RestirInitial(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= (uint2)gScreenSize)) return;
    RWTexture2D<float4> surfA = ResourceDescriptorHeap[rc0 + 6];
    RWTexture2D<float4> surfB = ResourceDescriptorHeap[rc0 + 7];
    uint seed = PcgHash(id.x * 3u + PcgHash(id.y * 5u + PcgHash(gFrameIndex * 7u + 1u)));

    // the primary surface, exactly as the path tracer's first sample will see it
    float3 d = CameraRayDir(id + PrimaryJitter(id));
    SceneHit h;
    if (rc3 != NO_GBUFFER) h = FromGBuffer(id, d);
    else h = Intersect(gCamPos, d);
    Reservoir r = (Reservoir)0;
    Material m = (Material)0;
    if (h.hit && !h.isLight) m = LoadMaterial(h.s.material);
    if (!h.hit || h.isLight || !RestirEligible(m) || gNumLights == 0)
    {
        surfA[id] = 0;
        surfB[id] = 0;
        StoreReservoir(rc2, id, r);
        return;
    }

    RestirSurface s;
    s.pos = OffsetRay(h.s.pos, h.s.N);
    s.N = h.s.N;
    s.V = -d;
    s.albedo = MaterialAlbedoLod(m, h.s.pos, h.s.uv, TextureLod(m, h.s, h.t * gPixelSpread, d));
    s.roughness = m.roughness;
    s.metallic = m.metallic;
    s.dist = h.t;
    uint packedAlbedo = PackUnorm4(float4(s.albedo, s.roughness));
    surfA[id] = float4(s.pos, s.dist);
    surfB[id] = float4(OctEncode(s.N), asfloat(packedAlbedo), s.metallic);
    // the same quantized material the neighbours will see (keeps the MIS weights consistent)
    float4 q = UnpackUnorm4(packedAlbedo);
    s.albedo = q.rgb;
    s.roughness = q.a;
    s.N = OctDecode(OctEncode(s.N));

    // 1) RIS: candidates from "uniform light, then a uniform direction inside its cone", converted to
    //    area measure on the sphere (the sample must be a point on the light to be shareable)
    for (uint i = 0; i < RestirCandidates; i++)
    {
        uint li = min((uint)(Rand(seed) * gNumLights), gNumLights - 1);
        float3 wi; float pdfW; float tHit;
        float2 u = Rand2(seed);
        r.M += 1;
        if (!SampleSphereLight(li, s.pos, u, wi, pdfW, tHit)) continue;
        Light Lt = LoadLight(li);
        float3 y = s.pos + wi * tHit;
        float3 dir = normalize(y - Lt.pos);
        float cosY = saturate(-dot(dir, wi));
        float pdfA = pdfW * cosY / (tHit * tHit) / gNumLights;
        float p = TargetPdf(s, li, dir);
        float w = pdfA > 0 ? p / pdfA : 0;
        r.wSum += w;
        if (w > 0 && Rand(seed) * r.wSum < w) { r.light = li; r.dir = dir; r.pHat = p; }
    }
    r.W = r.pHat > 0 ? r.wSum / (r.M * r.pHat) : 0;
    // visibility reuse: an occluded winner contributes nothing (but keeps its M)
    if (r.W > 0 && !SampleVisible(s, r.light, r.dir)) r.W = 0;

    // 2) temporal: last frame's final reservoir at the reprojected position, if it is the same surface
    if (rc5 == 0)
    {
        float4 clip = mul(float4(h.s.pos, 1), gPrevViewProj);
        float2 uv = clip.xy / clip.w * float2(0.5, -0.5) + 0.5;
        int2 pp = (int2)floor(uv * gScreenSize);
        if (clip.w > 0 && all(pp >= 0) && all(pp < (int2)gScreenSize))
        {
            RestirSurface ps = LoadSurface(rc1, pp);
            ps.V = normalize(gPrevCamPos - ps.pos);
            bool similar = ps.dist > 0 && dot(ps.N, s.N) > 0.9 && abs(dot(ps.pos - s.pos, s.N)) < 0.02 * s.dist;
            if (similar)
            {
                RestirSurface ss[MAX_MERGE];
                Reservoir cc[MAX_MERGE];
                ss[0] = s; cc[0] = r;
                ss[1] = ps; cc[1] = LoadReservoir(rc1, pp);
                cc[1].M = min(cc[1].M, RestirHistoryCap);   // bounded history: old samples fade out
                r = CombineMis(ss, cc, 2, seed);
            }
        }
    }
    StoreReservoir(rc2, id, r);
}

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void CS_RestirSpatial(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= (uint2)gScreenSize)) return;
    RestirSurface s = LoadSurface(rc0, id);
    if (s.dist <= 0)
    {
        StoreReservoir(rc0, id, (Reservoir)0);
        return;
    }
    uint seed = PcgHash(id.x * 11u + PcgHash(id.y * 13u + PcgHash(gFrameIndex * 17u + 5u)));

    RestirSurface ss[MAX_MERGE];
    Reservoir cc[MAX_MERGE];
    ss[0] = s;
    cc[0] = LoadReservoir(rc2, id);
    [unroll] for (uint i = 1; i < MAX_MERGE; i++)
    {
        // a rejected neighbour stays in its slot with M = 0 (constant indices keep the arrays in registers)
        ss[i] = s;
        cc[i] = (Reservoir)0;
        float2 u = Rand2(seed);
        float radius = 30 * sqrt(u.x);
        int2 q = (int2)id + (int2)round(radius * float2(cos(2 * PI * u.y), sin(2 * PI * u.y)));
        if (any(q < 0) || any(q >= (int2)gScreenSize) || all(q == (int2)id)) continue;
        RestirSurface ns = LoadSurface(rc0, q);
        if (ns.dist <= 0 || dot(ns.N, s.N) < 0.9 || abs(ns.dist - s.dist) > 0.1 * s.dist) continue;
        ss[i] = ns;
        cc[i] = LoadReservoir(rc2, q);
    }
    Reservoir r = CombineMis(ss, cc, MAX_MERGE, seed);
    // the neighbours' winners were visible from *their* surfaces - check this one
    if (r.W > 0 && !SampleVisible(s, r.light, r.dir)) r.W = 0;
    StoreReservoir(rc0, id, r);
}
