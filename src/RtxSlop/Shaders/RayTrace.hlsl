// Stage 2: Whitted-style ray tracing with DXR 1.1 inline ray queries (RayQuery in a compute shader).
//  - primary visibility is ray traced
//  - exact hard shadows: one shadow ray per light
//  - recursive mirror reflections (mirror <-> mirror "infinite" corridor, chrome sphere, polished floor)
// Root constants: rc0 = output UAV (HDR), rc1 = rays per pixel (1 or 4, rotated grid SSAA)

#include "Common.hlsli"

// Mirrors, the polished floor and all metals get a (single, sharp) reflection ray.
bool IsTracedReflective(Material m) { return m.roughness < 0.1 || m.metallic > 0.5; }

float3 TracePixel(float2 pixel)
{
    float3 o = gCamPos;
    float3 d = CameraRayDir(pixel);
    float3 radiance = 0;
    float3 throughput = 1;

    for (uint bounce = 0; bounce <= gMaxBounces; bounce++)
    {
        Hit h = TraceClosest(o, d, 0xFF);
        if (!h.hit) break;
        if (h.instance > 0)
        {
            radiance += throughput * LightEmission(h.instance - 1);
            break;
        }

        Surface s = GetSurface(h, o, d);
        Material m = LoadMaterial(s.material);
        float3 albedo = MaterialAlbedo(m, s.pos);
        float3 V = -d;
        bool reflective = IsTracedReflective(m) && bounce < gMaxBounces;

        float3 direct = 0;
        float3 P = OffsetRay(s.pos, s.N);
        for (uint li = 0; li < gNumLights; li++)
        {
            float3 c = ShadePointLight(li, s.pos, s.N, V, albedo, m, !reflective);
            if (any(c > 0))
            {
                float3 Lv = gLights[li].pos - P;
                float dist = length(Lv);
                if (TraceVisible(P, Lv / dist, dist - gLights[li].radius))
                    direct += c;
            }
        }
        direct += gAmbient * albedo * (1.0 - m.metallic);
        radiance += throughput * direct;

        if (!reflective) break;

        float3 F = F_Schlick(SpecularF0(albedo, m.metallic), saturate(dot(s.N, V)));
        throughput *= F;
        if (max(throughput.r, max(throughput.g, throughput.b)) < 0.01) break;
        o = P;
        d = reflect(d, s.N);
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
