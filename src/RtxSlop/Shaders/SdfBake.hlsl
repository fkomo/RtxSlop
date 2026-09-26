// One-time GPU bake of the loaded triangle mesh into a 3D signed distance texture for the SDF mode.
//  1. CS_SdfClear     reset the seed grid
//  2. CS_SdfSeed x2   per triangle: voxels near the triangle find their exact closest surface point
//                     (pass 0: InterlockedMin of the distance, pass 1: the winner writes its point)
//  3. CS_SdfJfa       3D jump flooding spreads the closest seed points to every voxel (log2 N passes)
//  4. CS_SdfFinalize  distance = |voxel - seed|, sign from inline ray queries against the mesh BLAS
//                     (hitting a back face first means the voxel is inside)
// FrameCB supplies the mesh buffers (gVbIndex, gIbIndex, gMeshFirstIndex, gTlasIndex) and the grid bounds
// (gMeshSdfMin/Max). Root constants: rc0 = seed UAV (src), rc1 = seed UAV (dst), rc2 = distance-key UAV,
// rc3 = SDF UAV, rc4 = JFA step / seed pass, rc5 = triangle count, rc6 = threads per dispatch row.

#include "Common.hlsli"

uint3 GridDims()
{
    RWTexture3D<float4> seeds = ResourceDescriptorHeap[rc0];
    uint3 d;
    seeds.GetDimensions(d.x, d.y, d.z);
    return d;
}

float3 VoxelCenter(uint3 v, uint3 dims)
{
    return gMeshSdfMin + (float3(v) + 0.5) * (gMeshSdfMax - gMeshSdfMin) / float3(dims);
}

[RootSignature(RS)]
[numthreads(4, 4, 4)]
void CS_SdfClear(uint3 v : SV_DispatchThreadID)
{
    uint3 dims = GridDims();
    if (any(v >= dims)) return;
    RWTexture3D<float4> seeds = ResourceDescriptorHeap[rc0];
    RWTexture3D<uint> keys = ResourceDescriptorHeap[rc2];
    seeds[v] = 0;
    keys[v] = 0xFFFFFFFFu;
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

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void CS_SdfSeed(uint3 id : SV_DispatchThreadID)
{
    uint tri = id.x + id.y * rc6;
    if (tri >= rc5) return;
    StructuredBuffer<Vertex> vb = ResourceDescriptorHeap[gVbIndex];
    StructuredBuffer<uint> ib = ResourceDescriptorHeap[gIbIndex];
    uint base = gMeshFirstIndex + tri * 3;
    float3 a = vb[ib[base]].pos, b = vb[ib[base + 1]].pos, c = vb[ib[base + 2]].pos;

    uint3 dims = GridDims();
    float3 voxel = (gMeshSdfMax - gMeshSdfMin) / float3(dims);
    int3 lo = max(int3(floor((min(a, min(b, c)) - gMeshSdfMin) / voxel - 1.0)), 0);
    int3 hi = min(int3(ceil((max(a, max(b, c)) - gMeshSdfMin) / voxel + 1.0)), int3(dims) - 1);
    hi = min(hi, lo + 15);   // cap huge triangles (not an issue for scanned meshes)
    float band = max(voxel.x, max(voxel.y, voxel.z));

    RWTexture3D<uint> keys = ResourceDescriptorHeap[rc2];
    RWTexture3D<float4> seeds = ResourceDescriptorHeap[rc0];
    for (int z = lo.z; z <= hi.z; z++)
    for (int y = lo.y; y <= hi.y; y++)
    for (int x = lo.x; x <= hi.x; x++)
    {
        uint3 v = uint3(x, y, z);
        float3 center = VoxelCenter(v, dims);
        float3 cp = ClosestPointOnTriangle(center, a, b, c);
        if (any(isnan(cp))) continue;
        float d = length(cp - center);
        if (d > band) continue;
        uint key = asuint(d);   // positive floats order like uints
        if (rc4 == 0)
            InterlockedMin(keys[v], key);
        else if (keys[v] == key)
            seeds[v] = float4(cp, 1);
    }
}

[RootSignature(RS)]
[numthreads(4, 4, 4)]
void CS_SdfJfa(uint3 v : SV_DispatchThreadID)
{
    uint3 dims = GridDims();
    if (any(v >= dims)) return;
    RWTexture3D<float4> src = ResourceDescriptorHeap[rc0];
    RWTexture3D<float4> dst = ResourceDescriptorHeap[rc1];
    int step = (int)rc4;
    float3 center = VoxelCenter(v, dims);
    float4 best = src[v];
    float bestD = best.w > 0 ? length(best.xyz - center) : 1e30;
    for (int z = -1; z <= 1; z++)
    for (int y = -1; y <= 1; y++)
    for (int x = -1; x <= 1; x++)
    {
        int3 n = int3(v) + int3(x, y, z) * step;
        if (any(n < 0) || any(n >= int3(dims))) continue;
        float4 s = src[n];
        if (s.w <= 0) continue;
        float d = length(s.xyz - center);
        if (d < bestD) { bestD = d; best = s; }
    }
    dst[v] = best;
}

[RootSignature(RS)]
[numthreads(4, 4, 4)]
void CS_SdfFinalize(uint3 v : SV_DispatchThreadID)
{
    uint3 dims = GridDims();
    if (any(v >= dims)) return;
    RWTexture3D<float4> seeds = ResourceDescriptorHeap[rc0];
    RWTexture3D<float> sdf = ResourceDescriptorHeap[rc3];
    float3 center = VoxelCenter(v, dims);
    float4 s = seeds[v];
    float d = s.w > 0 ? length(s.xyz - center) : length(gMeshSdfMax - gMeshSdfMin);

    // inside/outside: majority vote of 3 rays (robust against the odd hole or grazing hit)
    static const float3 dirs[3] = { float3(0.577, 0.577, 0.577), float3(-0.707, 0.1, 0.7), float3(0.2, -0.95, -0.24) };
    RaytracingAccelerationStructure tlas = ResourceDescriptorHeap[gTlasIndex];
    int inside = 0;
    [unroll] for (int i = 0; i < 3; i++)
    {
        RayDesc ray;
        ray.Origin = center; ray.Direction = normalize(dirs[i]); ray.TMin = 0; ray.TMax = 1e30;
        RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
        q.TraceRayInline(tlas, RAY_FLAG_NONE, MASK_MESH, ray);
        q.Proceed();
        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT && !q.CommittedTriangleFrontFace()) inside++;
    }
    sdf[v] = inside >= 2 ? -d : d;
}
