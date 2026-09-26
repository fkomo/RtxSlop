// Stage 3: unidirectional path tracer (inline ray queries).
//  - GGX microfacet BRDF with visible-normal (VNDF) importance sampling
//  - next event estimation towards the sphere lights (uniform cone sampling -> soft shadows)
//  - multiple importance sampling (power heuristic) between light and BSDF sampling
//  - Russian roulette
//  - smooth glass (Fresnel reflect/refract, Beer-Lambert absorption) - caustics emerge naturally
//  - writes albedo-demodulated radiance + "primary surface replacement" guides for the denoiser
//    (for mirror pixels the guides describe the surface seen *in* the mirror, like in Portal RTX)
// Root constants: rc0 = radiance UAV, rc1 = guide UAV (normal.xyz, virtual distance), rc2 = albedo UAV,
//                 rc3/rc4 = G-buffer SRVs (position / normal+material) for hybrid mode, or NO_GBUFFER
//                 to trace primary rays too.

#include "Common.hlsli"

// ---------------------------------------------------------------------------------------------
// BSDF sampling
// ---------------------------------------------------------------------------------------------
float3 SampleGGXVNDF(float3 Ve, float alpha, float2 u)
{
    // Heitz 2018, "Sampling the GGX Distribution of Visible Normals" (isotropic)
    float3 Vh = normalize(float3(alpha * Ve.x, alpha * Ve.y, Ve.z));
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 0 ? float3(-Vh.y, Vh.x, 0) * rsqrt(lensq) : float3(1, 0, 0);
    float3 T2 = cross(Vh, T1);
    float r = sqrt(u.x);
    float phi = 2.0 * PI * u.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(max(0, 1.0 - t1 * t1)) + s * t2;
    float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
    return normalize(float3(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z)));
}

float SpecularProbability(float3 N, float3 V, float3 albedo, Material m)
{
    float3 F = F_Schlick(SpecularF0(albedo, m.metallic), saturate(dot(N, V)));
    float ws = Luminance(F);
    float wd = Luminance(albedo) * (1.0 - m.metallic) * (1.0 - ws);
    return clamp(ws / max(ws + wd, 1e-6), 0.05, 1.0);
}

float BSDFPdf(float3 N, float3 V, float3 L, float pSpec, float alpha)
{
    float NdotL = dot(N, L);
    float NdotV = dot(N, V);
    if (NdotL <= 0 || NdotV <= 0) return 0;
    float3 H = normalize(V + L);
    float pdfSpec = G1_Smith(NdotV, alpha) * D_GGX(saturate(dot(N, H)), alpha) / (4.0 * NdotV);
    float pdfDiff = NdotL / PI;
    return pSpec * pdfSpec + (1.0 - pSpec) * pdfDiff;
}

bool SampleBSDF(float3 N, float3 V, float3 albedo, Material m, float3 u, out float3 L, out float3 weight, out float pdf)
{
    float alpha = RoughnessToAlpha(m.roughness);
    float pSpec = SpecularProbability(N, V, albedo, m);
    float3 t, b;
    BuildBasis(N, t, b);
    if (u.z < pSpec)
    {
        float3 Vl = float3(dot(V, t), dot(V, b), dot(V, N));
        float3 Hl = SampleGGXVNDF(Vl, alpha, u.xy);
        float3 H = t * Hl.x + b * Hl.y + N * Hl.z;
        L = reflect(-V, H);
    }
    else
    {
        float r = sqrt(u.x);
        float phi = 2.0 * PI * u.y;
        L = t * (r * cos(phi)) + b * (r * sin(phi)) + N * sqrt(max(0, 1.0 - u.x));
    }
    float NdotL = dot(N, L);
    if (NdotL <= 0) { weight = 0; pdf = 0; return false; }
    pdf = BSDFPdf(N, V, L, pSpec, alpha);
    if (pdf <= 0) { weight = 0; return false; }
    weight = EvalBRDF(N, V, L, albedo, m.roughness, m.metallic, 0) * NdotL / pdf;
    return true;
}

// ---------------------------------------------------------------------------------------------
// Sphere light sampling (uniform in the subtended cone)
// ---------------------------------------------------------------------------------------------
float OneMinusCosMax(float r2, float d2)
{
    float s2 = r2 / d2;
    return s2 / (1.0 + sqrt(max(0, 1.0 - s2)));   // numerically stable 1 - sqrt(1 - s2)
}

float SphereLightPdf(uint li, float3 P)
{
    Light Lt = gLights[li];
    float3 wc = Lt.pos - P;
    float d2 = dot(wc, wc);
    float r2 = Lt.radius * Lt.radius;
    if (d2 <= r2) return 0;
    return 1.0 / (2.0 * PI * OneMinusCosMax(r2, d2));
}

bool SampleSphereLight(uint li, float3 P, float2 u, out float3 dir, out float pdf, out float tHit)
{
    Light Lt = gLights[li];
    float3 wc = Lt.pos - P;
    float d2 = dot(wc, wc);
    float r2 = Lt.radius * Lt.radius;
    dir = 0; pdf = 0; tHit = 0;
    if (d2 <= r2) return false;
    float d = sqrt(d2);
    wc /= d;
    float omc = OneMinusCosMax(r2, d2);
    float cosT = 1.0 - u.x * omc;
    float sinT = sqrt(max(0, 1.0 - cosT * cosT));
    float phi = 2.0 * PI * u.y;
    float3 t, b;
    BuildBasis(wc, t, b);
    dir = normalize(t * (cos(phi) * sinT) + b * (sin(phi) * sinT) + wc * cosT);
    pdf = 1.0 / (2.0 * PI * omc);
    float tc = d * cosT;
    tHit = tc - sqrt(max(0, r2 - (d2 - tc * tc)));
    return true;
}

// ---------------------------------------------------------------------------------------------
// Scene intersection: traced ray, or (hybrid mode, depth 0) the rasterized G-buffer
// ---------------------------------------------------------------------------------------------
#define NO_GBUFFER 0xFFFFFFFFu
#define GBUF_LIGHT_CODE 1024u

struct SceneHit
{
    bool hit;
    bool isLight;
    uint light;
    float t;
    Surface s;
};

SceneHit Intersect(float3 o, float3 d)
{
    Hit h = TraceClosest(o, d, 0xFF);
    SceneHit r = (SceneHit)0;
    r.hit = h.hit;
    r.t = h.t;
    r.isLight = h.hit && IsLightInstance(h.instance);
    r.light = h.instance - 1;
    if (h.hit && !r.isLight) r.s = GetSurface(h, o, d);
    return r;
}

// Primary visibility from the raster pass: no ray is traced, dir is the camera -> surface direction.
SceneHit FromGBuffer(uint2 id, out float3 dir)
{
    Texture2D<float4> gPos = ResourceDescriptorHeap[rc3];
    Texture2D<float4> gNrm = ResourceDescriptorHeap[rc4];
    float4 p = gPos[id];
    float4 n = gNrm[id];
    uint code = (uint)(n.w + 0.5);
    SceneHit r = (SceneHit)0;
    r.hit = p.w > 0;
    float3 v = p.xyz - gCamPos;
    r.t = length(v);
    dir = r.hit ? v / r.t : CameraRayDir(id + 0.5);
    r.isLight = code >= GBUF_LIGHT_CODE;
    r.light = code - GBUF_LIGHT_CODE;
    r.s.pos = p.xyz;
    r.s.N = n.xyz;
    r.s.material = code;
    r.s.front = true;   // the camera is outside
    return r;
}

bool IsGuideMirror(Material m) { return m.metallic > 0.5 && m.roughness < 0.05; }

float3 ClampLum(float3 c, float maxLum)
{
    float l = Luminance(c);
    return l > maxLum ? c * (maxLum / l) : c;
}

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void CS_PathTrace(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= (uint2)gScreenSize)) return;
    RWTexture2D<float4> outRadiance = ResourceDescriptorHeap[rc0];
    RWTexture2D<float4> outGuide = ResourceDescriptorHeap[rc1];
    RWTexture2D<float4> outAlbedo = ResourceDescriptorHeap[rc2];

    uint seed = PcgHash(id.x + PcgHash(id.y + PcgHash(gFrameIndex)));
    bool hybrid = rc3 != NO_GBUFFER;

    // Denoiser guides: deterministic walk from the pixel center through perfect mirrors
    // ("primary surface replacement"). Using the jittered sample instead would make edge pixels
    // flip between surfaces every frame and constantly reject their temporal history.
    float3 guideN, guideAlbedo = 1;
    float guideDist = 0;
    {
        float3 o = gCamPos;
        float3 d = CameraRayDir(id + 0.5);
        float3 tint = 1;
        guideN = -d;
        for (uint k = 0; k < 4; k++)
        {
            SceneHit h;
            if (k == 0 && hybrid) h = FromGBuffer(id, d);
            else h = Intersect(o, d);
            if (!h.hit) { guideDist += 1000; break; }
            guideDist += h.t;
            if (h.isLight) { guideN = -d; guideAlbedo = tint; break; }
            Surface s = h.s;
            Material m = LoadMaterial(s.material);
            float3 albedo = MaterialAlbedo(m, s.pos);
            guideN = s.N;
            guideAlbedo = tint * albedo;
            bool glass = m.type == MAT_GLASS;
            if (!(IsGuideMirror(m) || glass) || k == 3) break;
            if (glass)
            {
                // see through glass along the dominant (refracted) direction
                float eta = s.front ? 1.0 / m.ior : m.ior;
                float F = FresnelDielectric(saturate(dot(s.N, -d)), eta);
                float3 T = refract(d, s.N, eta);
                if (F < 1.0) { tint *= (1.0 - F) * albedo; o = s.pos - s.N * 1e-3; d = T; }
                else { o = OffsetRay(s.pos, s.N); d = reflect(d, s.N); }
                continue;
            }
            tint *= F_Schlick(SpecularF0(albedo, m.metallic), saturate(dot(s.N, -d)));
            o = OffsetRay(s.pos, s.N);
            d = reflect(d, s.N);
        }
    }

    float3 sum = 0;

    for (uint sampleIdx = 0; sampleIdx < gSpp; sampleIdx++)
    {
        float2 jitter = Rand2(seed);   // sub-pixel jitter = free anti-aliasing after accumulation
        float3 o = gCamPos;
        float3 d = CameraRayDir(id + jitter);
        float3 throughput = 1;
        float3 L = 0;
        float prevPdf = 0;
        float3 prevPos = o;
        bool afterRough = false;


        for (uint depth = 0; depth <= gMaxBounces; depth++)
        {
            // hybrid: the first hit comes from the G-buffer (no AA jitter - it's the rasterized pixel)
            SceneHit h;
            if (depth == 0 && hybrid) h = FromGBuffer(id, d);
            else h = Intersect(o, d);
            if (!h.hit) break;

            if (h.isLight)
            {
                uint li = h.light;
                float3 Le = LightEmission(li);
                float w = 1;
                if (depth > 0)
                {
                    float pl = SphereLightPdf(li, prevPos);
                    w = (prevPdf * prevPdf) / (prevPdf * prevPdf + pl * pl);
                }
                float3 c = throughput * Le * w;
                L += afterRough ? ClampLum(c, 10) : c;
                break;
            }

            Surface s = h.s;
            Material m = LoadMaterial(s.material);
            float3 albedo = MaterialAlbedo(m, s.pos);
            float3 V = -d;
            float3 P = OffsetRay(s.pos, s.N);

            if (m.type == MAT_GLASS)
            {
                // smooth dielectric (delta BSDF): pick reflection or refraction with probability F,
                // which makes the Fresnel weights cancel. No NEE through delta lobes.
                if (!s.front) throughput *= GlassAbsorption(albedo, h.t);
                if (depth == gMaxBounces) break;
                float eta = s.front ? 1.0 / m.ior : m.ior;
                float F = FresnelDielectric(saturate(dot(s.N, V)), eta);
                if (Rand(seed) < F) { o = P; d = reflect(d, s.N); }
                else { o = s.pos - s.N * 1e-3; d = refract(d, s.N, eta); }
                prevPdf = 1e8;   // delta: light hits after this get full MIS weight
                prevPos = o;
                continue;
            }

            // Next event estimation with MIS
            float alpha = RoughnessToAlpha(m.roughness);
            float pSpec = SpecularProbability(s.N, V, albedo, m);
            for (uint li = 0; li < gNumLights; li++)
            {
                float3 ldir; float lpdf; float tHit;
                if (!SampleSphereLight(li, P, Rand2(seed), ldir, lpdf, tHit)) continue;
                float NdotL = dot(s.N, ldir);
                if (NdotL <= 0) continue;
                float3 f = EvalBRDF(s.N, V, ldir, albedo, m.roughness, m.metallic, 0);
                if (all(f <= 0)) continue;
                if (!TraceVisible(P, ldir, tHit - 2e-3)) continue;
                float bpdf = BSDFPdf(s.N, V, ldir, pSpec, alpha);
                float w = (lpdf * lpdf) / (lpdf * lpdf + bpdf * bpdf);
                float3 c = throughput * f * NdotL * LightEmission(li) * (w / lpdf);
                L += afterRough ? ClampLum(c, 10) : c;
            }

            if (depth == gMaxBounces) break;

            // Continue the path
            float3 newDir, weight;
            float pdf;
            float3 u = float3(Rand2(seed), Rand(seed));
            if (!SampleBSDF(s.N, V, albedo, m, u, newDir, weight, pdf)) break;
            throughput *= weight;
            afterRough = afterRough || m.roughness > 0.2;
            prevPdf = pdf;
            prevPos = P;
            o = P;
            d = newDir;

            if (depth >= 2)
            {
                float p = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.05, 0.95);
                if (Rand(seed) > p) break;
                throughput /= p;
            }
        }
        sum += L;
    }

    float3 radiance = sum / gSpp;
    float3 demodAlbedo = max(guideAlbedo, 0.01);
    outRadiance[id] = float4(radiance / demodAlbedo, 1);
    outGuide[id] = float4(guideN, guideDist);
    outAlbedo[id] = float4(demodAlbedo, 1);
}
