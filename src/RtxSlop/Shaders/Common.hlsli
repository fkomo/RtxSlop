// Shared declarations for every shader in the demo.
// Binding model: SM 6.6 bindless (ResourceDescriptorHeap[]) + one root signature for everything.

#define RS "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED), " \
           "RootConstants(num32BitConstants=8, b0), " \
           "CBV(b1), " \
           "CBV(b2), " \
           "StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_LINEAR, addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP, addressW=TEXTURE_ADDRESS_CLAMP), " \
           "StaticSampler(s1, filter=FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT, comparisonFunc=COMPARISON_LESS_EQUAL, addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP, addressW=TEXTURE_ADDRESS_CLAMP), " \
           "StaticSampler(s2, filter=FILTER_MIN_MAG_MIP_POINT, addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP, addressW=TEXTURE_ADDRESS_CLAMP)"

static const float PI = 3.14159265359;
static const uint NO_LIGHT = 0xFFFFFFFF;

struct Light
{
    float3 pos;
    float radius;
    float3 color;
    float intensity;
};

// Per-draw / per-dispatch root constants. Meaning depends on the shader.
cbuffer RootConsts : register(b0)
{
    uint rc0, rc1, rc2, rc3, rc4, rc5, rc6, rc7;
};

cbuffer FrameCB : register(b1)
{
    row_major float4x4 gViewProj;
    row_major float4x4 gInvViewProj;
    row_major float4x4 gPrevViewProj;
    float3 gCamPos;        float gTime;
    float3 gPrevCamPos;    uint  gFrameIndex;
    float2 gScreenSize;    float2 gInvScreenSize;
    uint gNumLights;       uint gMaxBounces;   uint gSpp;   uint gFlags;
    float3 gAmbient;       float gExposure;
    Light gLights[4];
    uint gVbIndex;         uint gIbIndex;      uint gMatIndex;  uint gTlasIndex;
    uint gShadowIndex;     uint gProbeIndex;   uint gStaticIndexCount; uint gMeshFirstIndex;
    float3 gProbePos;      float gShadowNear;
    float gShadowFar;      uint gMeshMaterial;  uint gMeshSdfIndex; uint gPad1;   // gMeshMaterial = NO_MESH without a model
    float3 gMeshSdfMin;    float gPad2;
    float3 gMeshSdfMax;    float gPad3;
};

cbuffer PassCB : register(b2)
{
    row_major float4x4 gPassViewProj;
    float3 gPassCamPos;    uint gPassFlags;
    float4 gClipPlane;
};

#define PASS_NO_MIRROR 1u   // mirrors can't sample their planar reflection (reflection / probe passes)
#define PASS_NO_PROBE  2u   // probe-reflective objects can't sample the probe (probe pass)

#define FRAME_SOFT_SHADOWS 1u

SamplerState           sLinear : register(s0);
SamplerComparisonState sShadow : register(s1);
SamplerState           sPoint  : register(s2);

struct Vertex
{
    float3 pos;
    float3 normal;
    uint material;
    uint pad;
};

struct Material
{
    float3 albedo;   float roughness;
    float3 emission; float metallic;
    uint type;       uint texIndex;  uint flags;  float ior;
};

#define MAT_STANDARD 0u
#define MAT_CHECKER  1u
#define MAT_MIRROR   2u
#define MAT_GLASS    3u   // smooth dielectric: albedo = transmittance after 1 m (Beer-Lambert), ior
#define MATF_PROBE   1u

Material LoadMaterial(uint id)
{
    StructuredBuffer<Material> mats = ResourceDescriptorHeap[gMatIndex];
    return mats[id];
}

float3 MaterialAlbedo(Material m, float3 p)
{
    if (m.type == MAT_CHECKER)
    {
        int2 c = (int2)floor(p.xz * 1.0);
        bool odd = ((c.x + c.y) & 1) != 0;
        // thin grout lines between tiles
        float2 f = abs(frac(p.xz) - 0.5);
        float grout = step(0.485, max(f.x, f.y));
        float3 a = odd ? m.albedo : m.albedo * float3(0.12, 0.12, 0.13);
        return lerp(a, float3(0.3, 0.3, 0.3), grout);
    }
    return m.albedo;
}

float3 LightEmission(uint li)
{
    Light L = gLights[li];
    // Radiance of a sphere light that matches a point light of the given intensity: I = Le * pi * r^2
    return L.color * L.intensity / (PI * L.radius * L.radius);
}

float Luminance(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

// ---------------------------------------------------------------------------------------------
// Microfacet BRDF (GGX / Smith / Schlick) - shared by all three renderers
// ---------------------------------------------------------------------------------------------
float RoughnessToAlpha(float r) { return max(r * r, 0.001); }

float D_GGX(float NdotH, float a)
{
    float a2 = a * a;
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

float G1_Smith(float NdotX, float a)
{
    float a2 = a * a;
    return 2.0 * NdotX / (NdotX + sqrt(a2 + (1.0 - a2) * NdotX * NdotX));
}

float3 F_Schlick(float3 F0, float cosTheta)
{
    float f = pow(saturate(1.0 - cosTheta), 5.0);
    return F0 + (1.0 - F0) * f;
}

float3 SpecularF0(float3 albedo, float metallic) { return lerp(float3(0.04, 0.04, 0.04), albedo, metallic); }

// BRDF value (without the cosine term). specScale lets callers drop / widen the specular lobe.
float3 EvalBRDF(float3 N, float3 V, float3 L, float3 albedo, float roughness, float metallic, float alphaOverride)
{
    float NdotL = dot(N, L);
    float NdotV = dot(N, V);
    if (NdotL <= 0 || NdotV <= 0) return 0;
    float3 H = normalize(V + L);
    float NdotH = saturate(dot(N, H));
    float VdotH = saturate(dot(V, H));
    float a = alphaOverride > 0 ? alphaOverride : RoughnessToAlpha(roughness);
    float3 F = F_Schlick(SpecularF0(albedo, metallic), VdotH);
    float3 spec = D_GGX(NdotH, a) * G1_Smith(NdotL, a) * G1_Smith(NdotV, a) * F / (4.0 * NdotL * NdotV);
    float3 diff = (1.0 - F) * (1.0 - metallic) * albedo / PI;
    return diff + spec;
}

float3 EvalDiffuseOnly(float3 N, float3 V, float3 L, float3 albedo, float metallic)
{
    float NdotL = dot(N, L);
    if (NdotL <= 0) return 0;
    float3 H = normalize(V + L);
    float3 F = F_Schlick(SpecularF0(albedo, metallic), saturate(dot(V, H)));
    return (1.0 - F) * (1.0 - metallic) * albedo / PI;
}

// Direct light from a small sphere light treated as a point light (raster + whitted RT).
// Specular lobe is widened by the light's angular size so tiny/glossy highlights stay visible.
float3 ShadePointLight(uint li, float3 P, float3 N, float3 V, float3 albedo, Material m, bool includeSpec)
{
    Light Lt = gLights[li];
    float3 Lv = Lt.pos - P;
    float d2 = dot(Lv, Lv);
    float d = sqrt(d2);
    float3 L = Lv / d;
    float NdotL = dot(N, L);
    if (NdotL <= 0) return 0;
    float3 radiance = Lt.color * Lt.intensity / max(d2, Lt.radius * Lt.radius);
    float3 f;
    if (includeSpec)
    {
        float a = RoughnessToAlpha(m.roughness);
        float aw = saturate(a + Lt.radius / (2.0 * d));
        f = EvalBRDF(N, V, L, albedo, m.roughness, m.metallic, aw);
    }
    else
    {
        f = EvalDiffuseOnly(N, V, L, albedo, m.metallic);
    }
    return f * radiance * NdotL;
}

// ---------------------------------------------------------------------------------------------
// Random numbers
// ---------------------------------------------------------------------------------------------
uint PcgHash(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float Rand(inout uint s)
{
    s = PcgHash(s);
    return (s >> 8) * (1.0 / 16777216.0);
}

float2 Rand2(inout uint s) { return float2(Rand(s), Rand(s)); }

void BuildBasis(float3 n, out float3 t, out float3 b)
{
    // Frisvad / Duff et al. branchless ONB
    float sign = n.z >= 0 ? 1.0 : -1.0;
    float a = -1.0 / (sign + n.z);
    float c = n.x * n.y * a;
    t = float3(1.0 + sign * n.x * n.x * a, sign * c, -sign * n.x);
    b = float3(c, sign + n.y * n.y * a, -n.y);
}

// ---------------------------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------------------------
float3 CameraRayDir(float2 pixel)
{
    float2 ndc = pixel * gInvScreenSize * float2(2, -2) + float2(-1, 1);
    float4 w = mul(float4(ndc, 1, 1), gInvViewProj);
    return normalize(w.xyz / w.w - gCamPos);
}

// ---------------------------------------------------------------------------------------------
// Inline ray tracing helpers (DXR 1.1 RayQuery)
// ---------------------------------------------------------------------------------------------
#define MASK_GEOMETRY 0x01
#define MASK_LIGHTS   0x02
#define MASK_MESH     0x04

// TLAS instance ids: 0 = static room, 1..3 = light bulbs, INSTANCE_MESH = the loaded .obj model
#define INSTANCE_MESH 100u
#define NO_MESH 0xFFFFFFFFu

bool IsLightInstance(uint id) { return id >= 1 && id < INSTANCE_MESH; }

struct Hit
{
    bool hit;
    uint instance;   // 0 = static scene, 1+i = light sphere i
    uint prim;
    float2 bary;
    float t;
};

Hit TraceClosest(float3 o, float3 d, uint mask)
{
    RaytracingAccelerationStructure tlas = ResourceDescriptorHeap[gTlasIndex];
    RayDesc ray;
    ray.Origin = o; ray.Direction = d; ray.TMin = 0.0; ray.TMax = 1e30;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, mask, ray);
    q.Proceed();
    Hit h;
    h.hit = q.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
    h.instance = h.hit ? q.CommittedInstanceID() : 0;
    h.prim = h.hit ? q.CommittedPrimitiveIndex() : 0;
    h.bary = h.hit ? q.CommittedTriangleBarycentrics() : float2(0, 0);
    h.t = h.hit ? q.CommittedRayT() : 1e30;
    return h;
}

bool TraceVisible(float3 o, float3 d, float tMax, uint mask = MASK_GEOMETRY | MASK_MESH)
{
    RaytracingAccelerationStructure tlas = ResourceDescriptorHeap[gTlasIndex];
    RayDesc ray;
    ray.Origin = o; ray.Direction = d; ray.TMin = 0.0; ray.TMax = tMax;
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, mask, ray);
    q.Proceed();
    return q.CommittedStatus() == COMMITTED_NOTHING;
}

bool MeshIsGlass() { return gMeshMaterial != NO_MESH && LoadMaterial(gMeshMaterial).type == MAT_GLASS; }

// Shadow rays for the Whitted-style modes: a glass model lets (tinted, dimmed) light through instead of
// casting a black shadow - the usual "transparent shadow" cheat. The path tracer doesn't use this.
float3 ShadowTransmittance(float3 o, float3 d, float tMax)
{
    if (!MeshIsGlass()) return TraceVisible(o, d, tMax) ? 1 : 0;
    if (!TraceVisible(o, d, tMax, MASK_GEOMETRY)) return 0;
    if (TraceVisible(o, d, tMax, MASK_MESH)) return 1;
    return LoadMaterial(gMeshMaterial).albedo * 0.65;
}

struct Surface
{
    float3 pos;
    float3 N;     // shading normal, facing the incoming ray
    uint material;
    bool front;   // ray hit the outside of the surface (entering glass)
};

Surface GetSurface(Hit h, float3 o, float3 d)
{
    StructuredBuffer<Vertex> vb = ResourceDescriptorHeap[gVbIndex];
    StructuredBuffer<uint> ib = ResourceDescriptorHeap[gIbIndex];
    uint base = (h.instance == INSTANCE_MESH ? gMeshFirstIndex : 0) + h.prim * 3;
    Vertex v0 = vb[ib[base + 0]];
    Vertex v1 = vb[ib[base + 1]];
    Vertex v2 = vb[ib[base + 2]];
    float3 w = float3(1.0 - h.bary.x - h.bary.y, h.bary.x, h.bary.y);
    Surface s;
    s.pos = v0.pos * w.x + v1.pos * w.y + v2.pos * w.z;
    float3 n = normalize(v0.normal * w.x + v1.normal * w.y + v2.normal * w.z);
    s.front = dot(n, d) < 0;
    s.N = s.front ? n : -n;
    s.material = v0.material;
    return s;
}

float3 OffsetRay(float3 p, float3 n) { return p + n * 1e-3; }

// ---------------------------------------------------------------------------------------------
// Dielectrics (glass)
// ---------------------------------------------------------------------------------------------
// Exact unpolarized Fresnel reflectance. eta = n_incident / n_transmitted, cosI = dot(N, V) > 0.
float FresnelDielectric(float cosI, float eta)
{
    float sin2T = eta * eta * (1.0 - cosI * cosI);
    if (sin2T >= 1.0) return 1.0;   // total internal reflection
    float cosT = sqrt(1.0 - sin2T);
    float rs = (eta * cosI - cosT) / (eta * cosI + cosT);
    float rp = (cosI - eta * cosT) / (cosI + eta * cosT);
    return 0.5 * (rs * rs + rp * rp);
}

// Beer-Lambert absorption over a path of length t inside the glass (albedo = color after 1 m).
float3 GlassAbsorption(float3 albedo, float t) { return pow(max(albedo, 1e-3), t); }
