// Stage 2: Whitted-style ray tracing with DXR 1.1 inline ray queries (RayQuery in a compute shader).
//  - primary visibility is ray traced
//  - exact hard shadows: one shadow ray per light
//  - recursive mirror reflections (mirror <-> mirror "infinite" corridor, chrome sphere, polished floor)
//  - glass: Fresnel-weighted reflection + refraction rays, Beer-Lambert absorption, tinted shadows
//  - textures filtered with ray cones, alpha-tested foliage, sky + baked sky visibility as ambient
// Root constants: rc0 = output UAV (HDR), rc1 = rays per pixel (1 or 4, rotated grid SSAA)

#include "Common.hlsli"

// Mirrors, the polished floor and all metals get a (single, sharp) reflection ray.
bool IsTracedReflective(Material m) { return m.type != MAT_GLASS && (m.roughness < 0.1 || m.metallic > 0.5); }

// Classic Whitted ray tree, flattened into a small explicit stack: glass spawns a reflection *and* a
// refraction ray, mirrors/metals a reflection ray, weighted by Fresnel.
struct RayTask
{
    float3 o;
    float3 d;
    float3 throughput;
    uint depth;
    float dist;   // distance travelled so far (ray cone width for texture LOD)
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
    first.dist = 0;
    stack[sp++] = first;
    float3 radiance = 0;

    for (int n = 0; n < MAX_RAYS && sp > 0; n++)
    {
        RayTask r = stack[--sp];
        Hit h = TraceClosest(r.o, r.d, 0xFF);
        if (!h.hit)
        {
            radiance += r.throughput * SkyRadiance(r.d);
            continue;
        }
        if (IsLightInstance(h.instance))
        {
            radiance += r.throughput * LightEmission(h.instance - 1);
            continue;
        }

        Surface s = GetSurface(h, r.o, r.d);
        Material m = LoadMaterial(s.material);
        float dist = r.dist + h.t;
        float3 albedo = MaterialAlbedoLod(m, s.pos, s.uv, TextureLod(m, s, dist * gPixelSpread, r.d));
        float3 V = -r.d;
        // glass gets extra depth: thin parts bounce light around inside a few times
        bool canRecurse = r.depth < gMaxBounces + (m.type == MAT_GLASS ? 4u : 0u);

        if (m.type == MAT_GLASS)
        {
            float3 thr = r.throughput;
            if (!s.front) thr *= GlassAbsorption(albedo, h.t);   // this segment ran inside the glass
            if (!canRecurse) continue;
            float eta = s.front ? 1.0 / m.ior : m.ior;
            float F = FresnelDielectric(saturate(dot(s.N, V)), eta);
            RayTask t;
            t.depth = r.depth + 1;
            t.dist = dist;
            if (F < 1.0 && sp < MAX_STACK)
            {
                t.o = s.pos - s.N * 1e-3;
                t.d = refract(r.d, s.N, eta);
                t.throughput = thr * (1.0 - F);
                stack[sp++] = t;
            }
            if (sp < MAX_STACK && max(thr.r, max(thr.g, thr.b)) * F > 0.01)
            {
                t.o = OffsetRay(s.pos, s.N);
                t.d = reflect(r.d, s.N);
                t.throughput = thr * F;
                stack[sp++] = t;
            }
            continue;
        }

        bool reflective = IsTracedReflective(m) && canRecurse;
        float3 direct = 0;
        float3 P = OffsetRay(s.pos, s.N);
        for (uint li = 0; li < gNumLights; li++)
        {
            float3 c = ShadePointLight(li, s.pos, s.N, V, albedo, m, !reflective);
            if (any(c > 0))
            {
                float3 Lv = gLights[li].pos - P;
                float dist = length(Lv);
                direct += c * ShadowTransmittance(P, Lv / dist, dist - gLights[li].radius);
            }
        }
        direct += AmbientLight(s.N, s.skyVis) * albedo * (1.0 - m.metallic);
        radiance += r.throughput * direct;

        if (reflective && sp < MAX_STACK)
        {
            float3 F = F_Schlick(SpecularF0(albedo, m.metallic), saturate(dot(s.N, V)));
            RayTask t;
            t.o = P;
            t.d = reflect(r.d, s.N);
            t.throughput = r.throughput * F;
            t.depth = r.depth + 1;
            t.dist = dist;
            if (max(t.throughput.r, max(t.throughput.g, t.throughput.b)) > 0.01)
                stack[sp++] = t;
        }
    }
    return radiance;
}

static const float2 kRotatedGrid[4] = { float2(0.375, 0.125), float2(0.875, 0.375), float2(0.125, 0.625), float2(0.625, 0.875) };

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void CS_RayTrace(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= (uint2)gScreenSize)) return;
    RWTexture2D<float4> output = ResourceDescriptorHeap[rc0];
    float3 c = 0;
    if (rc1 >= 4)
    {
        // weight samples by 1/(1+luminance) so very bright lights don't alias the edges
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
