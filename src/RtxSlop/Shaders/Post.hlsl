// Final pass: exposure + ACES filmic tonemap into the sRGB back buffer, HUD overlay.
// rc0 = HDR SRV, rc1 = HUD SRV, rc2 = HUD width, rc3 = HUD height

#include "Common.hlsli"

struct FSOut
{
    float4 pos : SV_Position;
};

[RootSignature(RS)]
FSOut VS_Fullscreen(uint id : SV_VertexID)
{
    float2 uv = float2((id << 1) & 2, id & 2);
    FSOut o;
    o.pos = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
    return o;
}

float3 ACESFitted(float3 x)
{
    // Krzysztof Narkowicz's fit of the ACES RRT+ODT
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

[RootSignature(RS)]
float4 PS_Tonemap(FSOut i) : SV_Target
{
    Texture2D<float4> hdr = ResourceDescriptorHeap[rc0];
    int2 p = (int2)i.pos.xy;
    float3 c = hdr[p].rgb;
    if (any(isnan(c)) || any(isinf(c))) c = float3(1, 0, 1);   // make NaNs obvious
    c = ACESFitted(c * gExposure);

    Texture2D<float4> hud = ResourceDescriptorHeap[rc1];
    int2 hp = p - int2(10, 10);
    if (all(hp >= 0) && hp.x < (int)rc2 && hp.y < (int)rc3)
    {
        float4 h = hud[hp];
        c = lerp(c, pow(h.rgb, 2.2), h.a);
    }
    return float4(c, 1);   // RTV is *_SRGB, hardware applies the gamma
}
