// Shared declarations for every shader in the demo.
// Binding model: SM 6.6 bindless (ResourceDescriptorHeap[]) + one root signature for everything.

#define RS "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED), " \
           "RootConstants(num32BitConstants=8, b0), " \
           "CBV(b1), " \
           "CBV(b2), " \
           "StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_LINEAR, addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP, addressW=TEXTURE_ADDRESS_CLAMP), " \
           "StaticSampler(s1, filter=FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT, comparisonFunc=COMPARISON_LESS_EQUAL, addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP, addressW=TEXTURE_ADDRESS_CLAMP), " \
           "StaticSampler(s2, filter=FILTER_MIN_MAG_MIP_POINT, addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP, addressW=TEXTURE_ADDRESS_CLAMP), " \
           "StaticSampler(s3, filter=FILTER_ANISOTROPIC, maxAnisotropy=8, addressU=TEXTURE_ADDRESS_WRAP, addressV=TEXTURE_ADDRESS_WRAP, addressW=TEXTURE_ADDRESS_WRAP), " \
           "StaticSampler(s4, filter=FILTER_MIN_MAG_MIP_LINEAR, addressU=TEXTURE_ADDRESS_WRAP, addressV=TEXTURE_ADDRESS_WRAP, addressW=TEXTURE_ADDRESS_WRAP)"

static const float PI = 3.14159265359;
static const uint NO_LIGHT = 0xFFFFFFFF;
static const uint NO_TEX = 0xFFFFFFFF;

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
    float gShadowFar;      uint gMeshMaterial;  uint gMeshSdfIndex; uint gGlassFirstIndex;   // gMeshMaterial = NO_MESH without a model
    float3 gMeshSdfMin;    float gSkyIntensity;       // gMeshSdf*: grid of the mesh being baked (SdfBake.hlsl)
    float3 gMeshSdfMax;    uint gAlphaFirstIndex;
    float gPixelSpread;    uint gBakeFlags;    float2 gPad4;   // pixel spread angle (ray cones), SDF bake: bit 0 = signed
};

cbuffer PassCB : register(b2)
{
    row_major float4x4 gPassViewProj;
    row_major float4x4 gPassInvViewProj;
    float3 gPassCamPos;    uint gPassFlags;
    float4 gClipPlane;
};

#define PASS_NO_MIRROR 1u   // mirrors can't sample their planar reflection (reflection / probe passes)
#define PASS_NO_PROBE  2u   // probe-reflective objects can't sample the probe (probe pass)

SamplerState           sLinear     : register(s0);
SamplerComparisonState sShadow     : register(s1);
SamplerState           sPoint      : register(s2);
SamplerState           sAniso      : register(s3);   // material textures, rasterizer (hardware derivatives)
SamplerState           sLinearWrap : register(s4);   // material textures, ray tracers (explicit LOD)

struct Vertex
{
    float3 pos;
    float3 normal;
    float2 uv;
    uint material;
    float skyVis;     // baked fraction of the hemisphere that sees the sky
};

struct Material
{
    float3 albedo;   float roughness;
    float3 emission; float metallic;
    uint type;       uint texIndex;  uint flags;  float ior;
    uint albedoTex;  uint alphaTex;  uint pad0;   uint pad1;
};

#define MAT_STANDARD 0u
#define MAT_CHECKER  1u
#define MAT_MIRROR   2u
#define MAT_GLASS    3u   // smooth dielectric: albedo = transmittance after 1 m (Beer-Lambert), ior
#define MATF_PROBE      1u
#define MATF_ALPHA_TEST 2u
#define MATF_ALPHA_IN_ALBEDO 4u   // alpha mask = the base color texture's alpha channel (else: red channel of alphaTex)

float MaskValue(Material m, float4 texel) { return (m.flags & MATF_ALPHA_IN_ALBEDO) != 0 ? texel.a : texel.r; }

Material LoadMaterial(uint id)
{
    StructuredBuffer<Material> mats = ResourceDescriptorHeap[gMatIndex];
    return mats[id];
}

// Base color without textures (+ the procedural checker floor of the room).
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

// Textured base color at an explicit mip level (ray tracers).
float3 MaterialAlbedoLod(Material m, float3 p, float2 uv, float lod)
{
    float3 a = MaterialAlbedo(m, p);
    if (m.albedoTex != NO_TEX)
    {
        Texture2D<float4> t = ResourceDescriptorHeap[m.albedoTex];
        a *= t.SampleLevel(sLinearWrap, uv, lod).rgb;
    }
    return a;
}

float3 LightEmission(uint li)
{
    Light L = gLights[li];
    // Radiance of a sphere light that matches a point light of the given intensity: I = Le * pi * r^2
    return L.color * L.intensity / (PI * L.radius * L.radius);
}

float Luminance(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

// ---------------------------------------------------------------------------------------------
// Sky: a simple overcast-ish dome (horizon -> zenith gradient), only in scenes that are open to it
// ---------------------------------------------------------------------------------------------
static const float3 kSkyZenith = float3(0.30, 0.50, 0.95);
static const float3 kSkyHorizon = float3(0.85, 0.90, 1.00);
static const float3 kGround = float3(0.30, 0.27, 0.24);

float3 SkyRadiance(float3 d)
{
    if (gSkyIntensity <= 0) return 0;
    if (d.y < 0) return kGround * 0.5 * gSkyIntensity;
    return lerp(kSkyHorizon, kSkyZenith, pow(saturate(d.y), 0.6)) * gSkyIntensity;
}

// Irradiance / pi from the sky dome for a normal, i.e. what a white diffuse surface reflects when it sees
// the whole sky. Multiplied by the baked sky visibility this is the "ambient" of the non path traced modes.
float3 SkyAmbient(float3 N)
{
    float up = N.y * 0.5 + 0.5;
    float3 sky = lerp(kSkyHorizon, kSkyZenith, 0.5);
    return lerp(kGround * 0.5, sky, up) * gSkyIntensity;
}

// The direct sky seen through the openings (skyVis) plus a crude constant for sky light that bounced
// around the building first - the non path traced modes have no other indirect light.
float SkyOcclusion(float skyVis) { return lerp(0.08, 1.0, skyVis); }
float3 AmbientLight(float3 N, float skyVis) { return gAmbient + SkyAmbient(N) * SkyOcclusion(skyVis); }

// ---------------------------------------------------------------------------------------------
// Microfacet BRDF (GGX / Smith / Schlick) - shared by all renderers
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

// BRDF value (without the cosine term). alphaOverride > 0 replaces the GGX alpha (light-size widening).
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

// Direct light from a small sphere light treated as a point light (raster + whitted RT + SDF).
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
// Triangle data
// ---------------------------------------------------------------------------------------------
struct Surface
{
    float3 pos;
    float3 N;         // shading normal, facing the incoming ray
    uint material;
    bool front;       // ray hit the outside of the surface (entering glass)
    float2 uv;
    float skyVis;
    float lodBias;    // 0.5 * log2(uv area / world area) of the triangle, for ray-cone texture LOD
};

// Barycentric interpolation of a triangle (global triangle index) as seen along direction d.
Surface TriangleSurface(uint tri, float3 w, float3 d)
{
    StructuredBuffer<Vertex> vb = ResourceDescriptorHeap[gVbIndex];
    StructuredBuffer<uint> ib = ResourceDescriptorHeap[gIbIndex];
    Vertex v0 = vb[ib[tri * 3 + 0]];
    Vertex v1 = vb[ib[tri * 3 + 1]];
    Vertex v2 = vb[ib[tri * 3 + 2]];
    Surface s;
    s.pos = v0.pos * w.x + v1.pos * w.y + v2.pos * w.z;
    float3 n = normalize(v0.normal * w.x + v1.normal * w.y + v2.normal * w.z);
    s.front = dot(n, d) < 0;
    s.N = s.front ? n : -n;
    s.material = v0.material;
    s.uv = v0.uv * w.x + v1.uv * w.y + v2.uv * w.z;
    s.skyVis = v0.skyVis * w.x + v1.skyVis * w.y + v2.skyVis * w.z;
    float worldArea = length(cross(v1.pos - v0.pos, v2.pos - v0.pos));
    float2 e1 = v1.uv - v0.uv, e2 = v2.uv - v0.uv;
    float uvArea = abs(e1.x * e2.y - e1.y * e2.x);
    s.lodBias = 0.5 * log2(max(uvArea, 1e-12) / max(worldArea, 1e-12));
    return s;
}

// Ericson, Real-Time Collision Detection 5.1.5
float3 ClosestPointOnTriangle(float3 p, float3 a, float3 b, float3 c)
{
    float3 ab = b - a, ac = c - a, ap = p - a;
    float d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0 && d2 <= 0) return a;
    float3 bp = p - b;
    float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) return b;
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) return a + ab * (d1 / (d1 - d3));
    float3 cp = p - c;
    float d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0 && d5 <= d6) return c;
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) return a + ac * (d2 / (d2 - d6));
    float va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    float denom = 1.0 / (va + vb + vc);
    return a + ab * (vb * denom) + ac * (vc * denom);
}

float3 Barycentrics(float3 p, float3 a, float3 b, float3 c)
{
    float3 v0 = b - a, v1 = c - a, v2 = p - a;
    float d00 = dot(v0, v0), d01 = dot(v0, v1), d11 = dot(v1, v1), d20 = dot(v2, v0), d21 = dot(v2, v1);
    float denom = max(d00 * d11 - d01 * d01, 1e-20);
    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    return float3(1.0 - v - w, v, w);
}

// Ray cones (Akenine-Moeller et al. 2019): texture LOD from the cone width at the hit, the triangle's
// texel density and the angle of incidence.
float TextureLod(Material m, Surface s, float coneWidth, float3 d)
{
    if (m.albedoTex == NO_TEX) return 0;
    Texture2D<float4> t = ResourceDescriptorHeap[m.albedoTex];
    uint w, h;
    t.GetDimensions(w, h);
    float cosI = max(abs(dot(d, s.N)), 0.1);
    return max(0, log2(max(coneWidth, 1e-6) / cosI) + s.lodBias + 0.5 * log2(float(w * h)));
}

// ---------------------------------------------------------------------------------------------
// Inline ray tracing helpers (DXR 1.1 RayQuery)
// ---------------------------------------------------------------------------------------------
#define MASK_GEOMETRY 0x01
#define MASK_LIGHTS   0x02
#define MASK_MESH     0x04
#define MASK_GLASS    0x08   // static glass objects
#define MASK_SOLID    (MASK_GEOMETRY | MASK_MESH | MASK_GLASS)

// TLAS instance ids: 0 = static scene, 1..3 = light bulbs, INSTANCE_MESH = the switchable .obj model,
// INSTANCE_GLASS = static glass objects (own BLAS so shadow rays can treat them separately)
#define INSTANCE_MESH 100u
#define INSTANCE_GLASS 101u
#define NO_MESH 0xFFFFFFFFu

bool IsLightInstance(uint id) { return id >= 1 && id < INSTANCE_MESH; }

// Global triangle index of a hit. The static BLAS has two geometries: 0 = opaque, 1 = alpha-tested.
uint GlobalTriangle(uint instance, uint geometry, uint prim)
{
    uint first = instance == INSTANCE_MESH ? gMeshFirstIndex
               : instance == INSTANCE_GLASS ? gGlassFirstIndex
               : (geometry == 1 ? gAlphaFirstIndex : 0);
    return first / 3 + prim;
}

// Alpha test of a non-opaque candidate (foliage, chains): mask texture at the finest mip.
bool AlphaTestPasses(uint instance, uint geometry, uint prim, float2 bary)
{
    uint tri = GlobalTriangle(instance, geometry, prim);
    StructuredBuffer<Vertex> vb = ResourceDescriptorHeap[gVbIndex];
    StructuredBuffer<uint> ib = ResourceDescriptorHeap[gIbIndex];
    Vertex v0 = vb[ib[tri * 3 + 0]];
    Material m = LoadMaterial(v0.material);
    if ((m.flags & MATF_ALPHA_TEST) == 0 || m.alphaTex == NO_TEX) return true;
    float2 uv = v0.uv * (1.0 - bary.x - bary.y) + vb[ib[tri * 3 + 1]].uv * bary.x + vb[ib[tri * 3 + 2]].uv * bary.y;
    Texture2D<float4> t = ResourceDescriptorHeap[m.alphaTex];
    return MaskValue(m, t.SampleLevel(sLinearWrap, uv, 0)) >= 0.5;
}

struct Hit
{
    bool hit;
    uint instance;   // 0 = static scene, 1+i = light sphere i, INSTANCE_MESH, INSTANCE_GLASS
    uint geometry;
    uint prim;
    float2 bary;
    float t;
};

Hit TraceClosest(float3 o, float3 d, uint mask, float tMax = 1e30)
{
    RaytracingAccelerationStructure tlas = ResourceDescriptorHeap[gTlasIndex];
    RayDesc ray;
    ray.Origin = o; ray.Direction = d; ray.TMin = 0.0; ray.TMax = tMax;
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, mask, ray);
    // only alpha-tested geometry is non-opaque, everything else never enters this loop
    while (q.Proceed())
    {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE &&
            AlphaTestPasses(q.CandidateInstanceID(), q.CandidateGeometryIndex(), q.CandidatePrimitiveIndex(), q.CandidateTriangleBarycentrics()))
            q.CommitNonOpaqueTriangleHit();
    }
    Hit h;
    h.hit = q.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
    h.instance = h.hit ? q.CommittedInstanceID() : 0;
    h.geometry = h.hit ? q.CommittedGeometryIndex() : 0;
    h.prim = h.hit ? q.CommittedPrimitiveIndex() : 0;
    h.bary = h.hit ? q.CommittedTriangleBarycentrics() : float2(0, 0);
    h.t = h.hit ? q.CommittedRayT() : 1e30;
    return h;
}

bool TraceVisible(float3 o, float3 d, float tMax, uint mask = MASK_SOLID)
{
    RaytracingAccelerationStructure tlas = ResourceDescriptorHeap[gTlasIndex];
    RayDesc ray;
    ray.Origin = o; ray.Direction = d; ray.TMin = 0.0; ray.TMax = tMax;
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, mask, ray);
    while (q.Proceed())
    {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE &&
            AlphaTestPasses(q.CandidateInstanceID(), q.CandidateGeometryIndex(), q.CandidatePrimitiveIndex(), q.CandidateTriangleBarycentrics()))
            q.CommitNonOpaqueTriangleHit();
    }
    return q.CommittedStatus() == COMMITTED_NOTHING;
}

Surface GetSurface(Hit h, float3 o, float3 d)
{
    float3 w = float3(1.0 - h.bary.x - h.bary.y, h.bary.x, h.bary.y);
    return TriangleSurface(GlobalTriangle(h.instance, h.geometry, h.prim), w, d);
}

float3 OffsetRay(float3 p, float3 n) { return p + n * 1e-3; }

// Shadow rays for the Whitted-style ray tracer: glass lets (tinted, dimmed) light through instead of
// casting a black shadow - the usual "transparent shadow" cheat. Glass lives in separate instances
// (static glass: MASK_GLASS, the model when its material is glass: MASK_MESH), so this is 1-3 cheap
// queries instead of walking through every glass surface.
// The path tracer doesn't use this (glass is a delta BSDF there; light through it arrives as caustics).
bool MeshIsGlass() { return gMeshMaterial != NO_MESH && LoadMaterial(gMeshMaterial).type == MAT_GLASS; }

float3 ShadowTransmittance(float3 o, float3 d, float tMax)
{
    bool meshGlass = MeshIsGlass();
    if (!TraceVisible(o, d, tMax, MASK_GEOMETRY | (meshGlass ? 0 : MASK_MESH))) return 0;
    float3 T = 1;
    Hit g = TraceClosest(o, d, MASK_GLASS, tMax);
    if (g.hit) T *= LoadMaterial(GetSurface(g, o, d).material).albedo * 0.65;
    if (meshGlass && !TraceVisible(o, d, tMax, MASK_MESH)) T *= LoadMaterial(gMeshMaterial).albedo * 0.65;
    return T;
}

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
