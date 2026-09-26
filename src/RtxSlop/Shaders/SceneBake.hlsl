// One-time bake at scene load: per-vertex sky visibility.
// For every vertex, cosine-distributed rays over its hemisphere count how much of the sky it sees.
// The rasterizer, the Whitted ray tracer and the SDF marcher multiply the sky light with it - a cheap,
// noise-free stand-in for the sky occlusion the path tracer gets for free.
// Root constants: rc0 = vertex buffer UAV, rc1 = vertex count, rc2 = rays per vertex, rc3 = threads per row

#include "Common.hlsli"

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void CS_SkyVisibility(uint3 id : SV_DispatchThreadID)
{
    uint vi = id.x + id.y * rc3;
    if (vi >= rc1) return;
    RWStructuredBuffer<Vertex> vb = ResourceDescriptorHeap[rc0];
    Vertex v = vb[vi];
    float3 N = normalize(v.normal);
    float3 t, b;
    BuildBasis(N, t, b);
    uint seed = PcgHash(vi * 9781u + 17u);
    float rotation = Rand(seed) * 2 * PI;
    float3 origin = v.pos + N * 0.01;

    uint open = 0;
    for (uint i = 0; i < rc2; i++)
    {
        // stratified cosine-weighted directions (golden-angle spiral), randomly rotated per vertex
        float u = (i + 0.5) / rc2;
        float r = sqrt(u);
        float phi = i * 2.39996323 + rotation;
        float3 d = normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + N * sqrt(1.0 - u));
        if (TraceVisible(origin, d, 1e30)) open++;
    }
    vb[vi].skyVis = (float)open / rc2;
}
