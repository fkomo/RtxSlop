// Real-time denoiser for the path tracer (simplified SVGF):
//  1. temporal accumulation with reprojection of the *virtual* (mirror-replaced) position,
//     history rejection via normal/depth, per-pixel luminance moments -> variance
//  2. edge-avoiding a-trous wavelet filter (5x5 B3 spline, 4 iterations), guided by
//     normal, depth and variance-scaled luminance; final pass re-applies albedo.

#include "Common.hlsli"

float3 PixelRayDir(float2 pixel) { return CameraRayDir(pixel); }

// rc0 noisy radiance SRV, rc1 guide SRV (cur), rc2 guide SRV (prev), rc3 history SRV (prev),
// rc4 history UAV (out), rc5 moments SRV (prev), rc6 moments UAV (out), rc7 = reset | (maxHistory << 1)
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void CS_Temporal(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= (uint2)gScreenSize)) return;
    Texture2D<float4> noisy = ResourceDescriptorHeap[rc0];
    Texture2D<float4> guideCur = ResourceDescriptorHeap[rc1];
    Texture2D<float4> guidePrev = ResourceDescriptorHeap[rc2];
    Texture2D<float4> histPrev = ResourceDescriptorHeap[rc3];
    RWTexture2D<float4> histOut = ResourceDescriptorHeap[rc4];
    Texture2D<float4> momPrev = ResourceDescriptorHeap[rc5];
    RWTexture2D<float4> momOut = ResourceDescriptorHeap[rc6];
    bool reset = (rc7 & 1) != 0;
    float maxHistory = (float)(rc7 >> 1);

    float3 cur = noisy[id].rgb;
    float4 g = guideCur[id];
    float lum = Luminance(cur);

    // virtual position: for mirror pixels this is the reflected image, which reprojects correctly
    float3 vpos = gCamPos + PixelRayDir(id + 0.5) * g.w;
    float4 pc = mul(float4(vpos, 1), gPrevViewProj);
    float4 prevColor = 0;
    float2 prevMom = 0;
    float sumW = 0;
    if (!reset && pc.w > 0)
    {
        float2 puv = pc.xy / pc.w * float2(0.5, -0.5) + 0.5;
        float2 ppix = puv * gScreenSize - 0.5;
        int2 base = (int2)floor(ppix);
        float2 f = ppix - base;
        float expectedDist = length(vpos - gPrevCamPos);
        [unroll] for (int k = 0; k < 4; k++)
        {
            int2 off = int2(k & 1, k >> 1);
            int2 p = base + off;
            if (any(p < 0) || any(p >= (int2)gScreenSize)) continue;
            float4 gp = guidePrev[p];
            bool valid = dot(gp.xyz, g.xyz) > 0.9 && abs(gp.w - expectedDist) < 0.03 * expectedDist + 0.02;
            if (!valid) continue;
            float w = (off.x ? f.x : 1 - f.x) * (off.y ? f.y : 1 - f.y);
            prevColor += histPrev[p] * w;
            prevMom += momPrev[p].xy * w;
            sumW += w;
        }
    }

    float hist = 1;
    float3 color = cur;
    float2 mom = float2(lum, lum * lum);
    if (sumW > 0.01)
    {
        prevColor /= sumW;
        prevMom /= sumW;
        hist = min(prevColor.a + 1, maxHistory);
        float a = 1.0 / hist;
        color = lerp(prevColor.rgb, cur, a);
        mom = lerp(prevMom, mom, a);
    }
    float variance = max(0, mom.y - mom.x * mom.x);
    if (hist < 4) variance = max(variance, 4.0 / hist);   // fresh pixels: allow aggressive blur
    histOut[id] = float4(color, hist);
    momOut[id] = float4(mom, variance, 0);
}

// rc0 src SRV, rc1 dst UAV, rc2 guide SRV, rc3 moments SRV, rc4 albedo SRV, rc5 step size (0 = copy),
// rc6 = 1 on the final pass (re-apply albedo)
static const float kKernel[3] = { 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0 };

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void CS_Atrous(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= (uint2)gScreenSize)) return;
    Texture2D<float4> src = ResourceDescriptorHeap[rc0];
    RWTexture2D<float4> dst = ResourceDescriptorHeap[rc1];
    Texture2D<float4> guide = ResourceDescriptorHeap[rc2];
    Texture2D<float4> moments = ResourceDescriptorHeap[rc3];
    Texture2D<float4> albedoTex = ResourceDescriptorHeap[rc4];
    int step = (int)rc5;
    bool final = rc6 != 0;

    float4 center = src[id];
    float3 result = center.rgb;
    if (step > 0)
    {
        float4 gc = guide[id];
        float lc = Luminance(center.rgb);
        float sigmaL = 4.0 * sqrt(moments[id].z) + 1e-4;
        float3 sum = 0;
        float sumW = 0;
        [unroll] for (int y = -2; y <= 2; y++)
        {
            [unroll] for (int x = -2; x <= 2; x++)
            {
                int2 p = (int2)id + int2(x, y) * step;
                if (any(p < 0) || any(p >= (int2)gScreenSize)) continue;
                float4 c = src[p];
                float4 gq = guide[p];
                float wn = pow(saturate(dot(gc.xyz, gq.xyz)), 64.0);
                float wz = exp(-abs(gc.w - gq.w) / (gc.w * 0.01 * step + 1e-3));
                float wl = exp(-abs(lc - Luminance(c.rgb)) / sigmaL);
                float w = kKernel[abs(x)] * kKernel[abs(y)] * wn * wz * wl;
                sum += c.rgb * w;
                sumW += w;
            }
        }
        result = sum / max(sumW, 1e-6);
    }

    if (final)
        dst[id] = float4(result * albedoTex[id].rgb, 1);
    else
        dst[id] = float4(result, center.a);
}
