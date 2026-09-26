// Stage 1: classic real-time rasterization.
//  - forward PBR shading (GGX)
//  - omnidirectional shadow maps (cube map array, hardware PCF + Poisson disk)
//  - planar reflections for mirrors (scene re-rendered through a mirrored camera)
//  - dynamic cube-map reflection probe for the metal spheres
//  - screen-space refraction for glass (PS_Glass, drawn after the opaque scene)
// Root constants: rc0 = first index in the index buffer, rc1 = light index (NO_LIGHT for static geometry),
//                 rc2 = (PS_Glass) SRV of the opaque scene color

#include "Common.hlsli"

struct VSOut
{
    float4 pos  : SV_Position;
    float3 wpos : WPOS;
    float3 nrm  : NORMAL;
    nointerpolation uint mat   : MATERIAL;
    nointerpolation uint light : LIGHT;
    float clip  : SV_ClipDistance0;
};

Vertex FetchVertex(uint vid, out float3 worldPos)
{
    StructuredBuffer<uint> ib = ResourceDescriptorHeap[gIbIndex];
    StructuredBuffer<Vertex> vb = ResourceDescriptorHeap[gVbIndex];
    Vertex v = vb[ib[rc0 + vid]];
    worldPos = v.pos;
    if (rc1 != NO_LIGHT)
    {
        Light L = gLights[rc1];
        worldPos = L.pos + v.pos * L.radius;
    }
    return v;
}

[RootSignature(RS)]
VSOut VS_Main(uint vid : SV_VertexID)
{
    float3 wp;
    Vertex v = FetchVertex(vid, wp);
    VSOut o;
    o.wpos = wp;
    o.nrm = v.normal;
    o.mat = v.material;
    o.light = rc1;
    o.pos = mul(float4(wp, 1), gPassViewProj);
    o.clip = dot(float4(wp, 1), gClipPlane);
    return o;
}

[RootSignature(RS)]
float4 VS_Shadow(uint vid : SV_VertexID) : SV_Position
{
    float3 wp;
    FetchVertex(vid, wp);
    return mul(float4(wp, 1), gPassViewProj);
}

static const float2 kPoisson[12] =
{
    float2(-0.326, -0.406), float2(-0.840, -0.074), float2(-0.696, 0.457),
    float2(-0.203, 0.621),  float2(0.962, -0.195),  float2(0.473, -0.480),
    float2(0.519, 0.767),   float2(0.185, -0.893),  float2(0.507, 0.064),
    float2(0.896, 0.412),   float2(-0.322, -0.933), float2(-0.792, -0.598)
};

float ShadowFactor(uint li, float3 P, float3 N, float2 screenPos)
{
    TextureCubeArray<float> sm = ResourceDescriptorHeap[gShadowIndex];
    Light Lt = gLights[li];
    float3 d = (P + N * 0.015) - Lt.pos;
    float3 ad = abs(d);
    float z = max(ad.x, max(ad.y, ad.z));
    float n = gShadowNear, f = gShadowFar;
    float depth = f / (f - n) - (f * n / (f - n)) / z;

    float3 dir = normalize(d);
    float3 t, b;
    BuildBasis(dir, t, b);
    // per-pixel rotation of the Poisson disk to turn banding into fine noise
    float ang = frac(sin(dot(screenPos, float2(12.9898, 78.233))) * 43758.5453) * 2 * PI;
    float2x2 rot = float2x2(cos(ang), -sin(ang), sin(ang), cos(ang));
    float radius = 0.004 + Lt.radius * 0.02;
    float sum = 0;
    [unroll] for (int i = 0; i < 12; i++)
    {
        float2 o = mul(kPoisson[i], rot) * radius;
        float3 sd = dir + t * o.x + b * o.y;
        sum += sm.SampleCmpLevelZero(sShadow, float4(sd, li), depth - 0.0002);
    }
    return sum / 12.0;
}

[RootSignature(RS)]
float4 PS_Main(VSOut i) : SV_Target
{
    if (i.light != NO_LIGHT)
        return float4(LightEmission(i.light), 1);

    Material m = LoadMaterial(i.mat);
    float3 N = normalize(i.nrm);
    float3 V = normalize(gPassCamPos - i.wpos);
    float3 albedo = MaterialAlbedo(m, i.wpos);
    if (m.type == MAT_GLASS) albedo *= 0.05;   // secondary passes (probe, mirrors): glass as dark glossy glass

    bool hasRefl = false;
    float3 refl = 0;
    if (m.type == MAT_MIRROR && (gPassFlags & PASS_NO_MIRROR) == 0)
    {
        Texture2D<float4> rt = ResourceDescriptorHeap[m.texIndex];
        refl = rt.SampleLevel(sLinear, i.pos.xy * gInvScreenSize, 0).rgb;
        hasRefl = true;
    }
    else if ((m.flags & MATF_PROBE) != 0 && (gPassFlags & PASS_NO_PROBE) == 0)
    {
        TextureCube<float4> probe = ResourceDescriptorHeap[m.texIndex];   // each material names its probe
        float3 R = reflect(-V, N);
        refl = probe.SampleLevel(sLinear, R, 0).rgb;
        hasRefl = true;
    }

    float3 color = 0;
    for (uint li = 0; li < gNumLights; li++)
    {
        float3 c = ShadePointLight(li, i.wpos, N, V, albedo, m, !hasRefl);
        if (any(c > 0))
            c *= ShadowFactor(li, i.wpos, N, i.pos.xy);
        color += c;
    }
    color += gAmbient * albedo * (1.0 - m.metallic);

    if (hasRefl)
    {
        float3 F = F_Schlick(SpecularF0(albedo, m.metallic), saturate(dot(N, V)));
        color += refl * F * (1.0 - 0.6 * m.roughness);
    }
    return float4(color, 1);
}

// Hybrid mode: G-buffer for the path tracer (primary visibility is rasterized, everything else is traced).
// RT0 = world position (w = 1 where covered), RT1 = normal + material code as float
// (material index, or GBUF_LIGHT_CODE + bulb index; small integers are exact in fp32).
#define GBUF_LIGHT_CODE 1024.0

struct GBufferOut
{
    float4 pos : SV_Target0;
    float4 nrmMat : SV_Target1;
};

[RootSignature(RS)]
GBufferOut PS_GBuffer(VSOut i)
{
    float3 N = normalize(i.nrm);
    if (dot(N, gPassCamPos - i.wpos) < 0) N = -N;
    GBufferOut o;
    o.pos = float4(i.wpos, 1);
    o.nrmMat = float4(N, i.light != NO_LIGHT ? GBUF_LIGHT_CODE + i.light : (float)i.mat);
    return o;
}

// Glass in the rasterizer: there is no second surface or real refraction, so fake it -
// bend the view ray, pretend it travels ~35 cm through the glass and look up the already rendered
// opaque scene there (screen-space refraction), mix with the probe reflection by Fresnel.
[RootSignature(RS)]
float4 PS_Glass(VSOut i) : SV_Target
{
    Material m = LoadMaterial(i.mat);
    float3 N = normalize(i.nrm);
    float3 V = normalize(gPassCamPos - i.wpos);
    if (dot(N, V) < 0) N = -N;
    float eta = 1.0 / m.ior;
    float F = FresnelDielectric(saturate(dot(N, V)), eta);

    const float thickness = 0.35;
    float3 T = refract(-V, N, eta);
    float4 clip = mul(float4(i.wpos + T * thickness, 1), gPassViewProj);
    float2 uv = saturate(clip.xy / clip.w * float2(0.5, -0.5) + 0.5);
    Texture2D<float4> sceneColor = ResourceDescriptorHeap[rc2];
    float3 refr = sceneColor.SampleLevel(sLinear, uv, 0).rgb * GlassAbsorption(m.albedo, 2.0 * thickness);

    TextureCube<float4> probe = ResourceDescriptorHeap[m.texIndex];
    float3 refl = probe.SampleLevel(sLinear, reflect(-V, N), 0).rgb;

    // sharp highlights of the bulbs (specular only)
    Material spec = m;
    spec.roughness = 0.05;
    spec.metallic = 0;
    float3 highlights = 0;
    for (uint li = 0; li < gNumLights; li++)
        highlights += ShadePointLight(li, i.wpos, N, V, float3(0, 0, 0), spec, true);

    return float4(lerp(refr, refl, F) + highlights, 1);
}
