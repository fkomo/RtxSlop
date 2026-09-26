using System.Numerics;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;

namespace RtxSlop;

public enum RenderMode { Raster = 0, RayTrace = 1, PathTrace = 2, Hybrid = 3, Sdf = 4 }

public sealed class RenderSettings
{
    public RenderMode Mode = RenderMode.Raster;
    public int RtBounces = 6;
    public int PtBounces = 5;
    public int Spp = 1;
    public bool Denoise = true;
    public float Exposure = 1.0f;
    public bool VSync = false;
    public bool LightsAnimating = true;
    public bool LightsOn = true;           // off: only the sky (Sponza) / ambient (room) lights the scene
    public bool AntiAliasing = true;   // 4x MSAA (raster) / 4x SSAA (ray tracing)
    public bool SdfAntiAliasing = false;   // 4x SSAA for the SDF marcher (~5x the cost, so off by default)
}

/// <summary>Layout must match FrameCB in Common.hlsli.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct FrameData
{
    public Matrix4x4 ViewProj;
    public Matrix4x4 InvViewProj;
    public Matrix4x4 PrevViewProj;
    public Vector3 CamPos; public float Time;
    public Vector3 PrevCamPos; public uint FrameIndex;
    public Vector2 ScreenSize; public Vector2 InvScreenSize;
    public uint NumLights, MaxBounces, Spp, Flags;
    public Vector3 Ambient; public float Exposure;
    public LightData Light0, Light1, Light2, Light3;
    public uint VbIndex, IbIndex, MatIndex, TlasIndex;
    public uint ShadowIndex, ProbeIndex, StaticIndexCount, MeshFirstIndex;
    public Vector3 ProbePos; public float ShadowNear;
    public float ShadowFar; public uint MeshMaterial, MeshSdfIndex, GlassFirstIndex;
    public Vector3 MeshSdfMin; public float SkyIntensity;
    public Vector3 MeshSdfMax; public uint AlphaFirstIndex;
    public float PixelSpread; public uint BakeFlags; public Vector2 Pad4;
}

/// <summary>Layout must match PassCB in Common.hlsli.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct PassData
{
    public Matrix4x4 ViewProj;
    public Matrix4x4 InvViewProj;
    public Vector3 CamPos; public uint Flags;
    public Vector4 ClipPlane;

    public const uint NoMirror = 1, NoProbe = 2;
}

/// <summary>D3D12_RAYTRACING_INSTANCE_DESC (64 bytes), written straight into upload memory.</summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct InstanceDesc
{
    public fixed float Transform[12];      // 3x4 row-major
    public uint InstanceIdAndMask;         // id:24 | mask:8
    public uint HitGroupAndFlags;          // contribution:24 | flags:8
    public ulong AccelerationStructure;

    public static InstanceDesc Create(Matrix4x4 m, uint id, byte mask, ulong blas, RaytracingInstanceFlags flags = RaytracingInstanceFlags.None)
    {
        var d = new InstanceDesc();
        // System.Numerics is row-vector (translation in M41..M43); DXR wants a 3x4 column-vector matrix.
        d.Transform[0] = m.M11; d.Transform[1] = m.M21; d.Transform[2] = m.M31; d.Transform[3] = m.M41;
        d.Transform[4] = m.M12; d.Transform[5] = m.M22; d.Transform[6] = m.M32; d.Transform[7] = m.M42;
        d.Transform[8] = m.M13; d.Transform[9] = m.M23; d.Transform[10] = m.M33; d.Transform[11] = m.M43;
        d.InstanceIdAndMask = (id & 0xFFFFFF) | ((uint)mask << 24);
        d.HitGroupAndFlags = (uint)flags << 24;
        d.AccelerationStructure = blas;
        return d;
    }
}

/// <summary>A GPU texture plus its tracked state and descriptor slots.</summary>
public sealed class GpuTexture : IDisposable
{
    public ID3D12Resource Resource;
    public ResourceStates State;
    public int Srv = -1, Uav = -1;
    public CpuDescriptorHandle Rtv, Dsv;
    public int Width, Height, Depth = 1;

    public void Dispose()
    {
        Resource?.Dispose();
        Resource = null;
    }
}

/// <summary>Linear per-frame allocator in a persistently mapped upload heap.</summary>
public sealed unsafe class UploadRing : IDisposable
{
    public readonly ID3D12Resource Buffer;
    readonly byte* cpu;
    readonly ulong gpu, size;
    ulong offset;

    public UploadRing(ID3D12Device device, ulong size)
    {
        this.size = size;
        Buffer = device.CreateCommittedResource(HeapType.Upload, ResourceDescription.Buffer(size), ResourceStates.GenericRead);
        void* p;
        Buffer.Map(0, null, &p).CheckError();
        cpu = (byte*)p;
        gpu = Buffer.GPUVirtualAddress;
    }

    public void Reset() => offset = 0;

    public (ulong Gpu, ulong Offset, IntPtr Cpu) Alloc(ulong bytes, ulong align = 256)
    {
        offset = (offset + align - 1) & ~(align - 1);
        if (offset + bytes > size) throw new InvalidOperationException("Upload ring overflow");
        var r = (gpu + offset, offset, (IntPtr)(cpu + offset));
        offset += bytes;
        return r;
    }

    public ulong Push<T>(in T data, ulong align = 256) where T : unmanaged
    {
        var a = Alloc((ulong)sizeof(T), align);
        *(T*)a.Cpu = data;
        return a.Gpu;
    }

    public void Dispose()
    {
        Buffer.Unmap(0);
        Buffer.Dispose();
    }
}
