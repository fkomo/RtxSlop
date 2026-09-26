using System.Drawing;
using System.Drawing.Imaging;
using System.Numerics;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;
using Vortice.DXGI;
using ShaderResourceViewDimension = Vortice.Direct3D12.ShaderResourceViewDimension;

namespace RtxSlop;

/// <summary>Everything the GPU holds for one scene. Created on first use and kept, so switching is instant.</summary>
sealed class SceneGpu : IDisposable
{
    public Scene Scene;
    public ID3D12Resource Vb, Ib, MatBuf, SdfBuf;
    public ID3D12Resource BlasStatic, BlasGlass, BlasSphere, BlasMesh, Tlas, TlasScratch;
    public BuildRaytracingAccelerationStructureInputs TlasInputs;
    public int TlasInstanceCount;
    public int VbSlot, IbSlot, MatSlot, SdfSlot, TlasSlot;
    public readonly List<GpuTexture> Textures = new();
    public uint[] TextureSlots = Array.Empty<uint>();
    public readonly List<GpuTexture> SdfGrids = new();

    /// <summary>Scene materials reference textures by id + 1; the GPU copy needs their SRV slots.</summary>
    public Material Remap(Material m)
    {
        m.AlbedoTex = m.AlbedoTex == 0 ? 0xFFFFFFFFu : TextureSlots[m.AlbedoTex - 1];
        m.AlphaTex = m.AlphaTex == 0 ? 0xFFFFFFFFu : TextureSlots[m.AlphaTex - 1];
        return m;
    }

    public void Dispose()
    {
        foreach (var d in new IDisposable[] { Vb, Ib, MatBuf, SdfBuf, BlasStatic, BlasGlass, BlasSphere, BlasMesh, Tlas, TlasScratch })
            d?.Dispose();
        foreach (var t in Textures) t.Dispose();
        foreach (var t in SdfGrids) t.Dispose();
    }
}

public sealed unsafe partial class Renderer
{
    const int SkyVisRays = 64;
    readonly Dictionary<Scene, SceneGpu> sceneGpus = new();
    SceneGpu gpu;
    Scene scene => gpu.Scene;
    int nextSrvSlot = FirstDynamicSlot;

    int AllocSrvSlot()
    {
        if (nextSrvSlot >= SrvHeapSize) throw new InvalidOperationException("Descriptor heap full");
        return nextSrvSlot++;
    }

    /// <summary>Makes <paramref name="s"/> the rendered scene, uploading / baking it the first time.</summary>
    public void SetScene(Scene s)
    {
        WaitIdle();
        if (!sceneGpus.TryGetValue(s, out var g))
        {
            g = CreateSceneGpu(s);
            sceneGpus.Add(s, g);
        }
        gpu = g;
        pendingMaterials.Clear();
        historyValid = false;
    }

    SceneGpu CreateSceneGpu(Scene s)
    {
        var g = new SceneGpu { Scene = s };
        var temps = new List<IDisposable>();
        allocators[0].Reset();
        cl.Reset(allocators[0], null);
        cl.SetDescriptorHeaps(srvHeap);
        cl.SetComputeRootSignature(rootSig);

        ID3D12Resource Upload<T>(string name, ReadOnlySpan<T> data, ResourceFlags flags = ResourceFlags.None) where T : unmanaged
        {
            ulong size = (ulong)Math.Max(1, data.Length * sizeof(T));
            var dst = CreateBuffer(name, size, HeapType.Default, flags, ResourceStates.Common);
            var src = CreateBuffer(name + " upload", size, HeapType.Upload, ResourceFlags.None, ResourceStates.GenericRead);
            if (data.Length > 0) src.SetData(data);
            temps.Add(src);
            cl.CopyBufferRegion(dst, 0, src, 0, size);
            cl.ResourceBarrierTransition(dst, ResourceStates.CopyDest, ResourceStates.AllShaderResource, D3D12.ResourceBarrierAllSubResources, ResourceBarrierFlags.None);
            return dst;
        }

        // ---- textures (decoded + mipmapped on the CPU in parallel) ----
        LoadTextures(g, temps);

        // ---- geometry + materials ----
        g.Vb = Upload(s.Name + " vertices", CollectionsMarshal.AsSpan(s.Vertices), ResourceFlags.AllowUnorderedAccess);
        g.Ib = Upload(s.Name + " indices", CollectionsMarshal.AsSpan(s.Indices));
        var gpuMaterials = s.Materials.Select(g.Remap).ToArray();
        g.MatBuf = Upload(s.Name + " materials", gpuMaterials.AsSpan());
        g.VbSlot = AllocSrvSlot(); CreateStructuredSrv(g.Vb, g.VbSlot, s.Vertices.Count, sizeof(Vertex));
        g.IbSlot = AllocSrvSlot(); CreateStructuredSrv(g.Ib, g.IbSlot, s.Indices.Count, 4);
        g.MatSlot = AllocSrvSlot(); CreateStructuredSrv(g.MatBuf, g.MatSlot, s.Materials.Count, sizeof(Material));

        // ---- bottom level acceleration structures ----
        ID3D12Resource BuildBlas(string name, params (int First, int Count, bool Opaque)[] ranges)
        {
            var geos = ranges.Where(r => r.Count > 0).Select(r => new RaytracingGeometryDescription(new RaytracingGeometryTrianglesDescription
            {
                VertexBuffer = new GpuVirtualAddressAndStride(g.Vb.GPUVirtualAddress, (ulong)sizeof(Vertex)),
                VertexFormat = Format.R32G32B32_Float,
                VertexCount = (uint)s.Vertices.Count,
                IndexBuffer = g.Ib.GPUVirtualAddress + (ulong)r.First * 4,
                IndexFormat = Format.R32_UInt,
                IndexCount = (uint)r.Count,
                Transform3x4 = 0,
            }, r.Opaque ? RaytracingGeometryFlags.Opaque : RaytracingGeometryFlags.None)).ToArray();
            if (geos.Length == 0) return null;
            var inputs = new BuildRaytracingAccelerationStructureInputs
            {
                Type = RaytracingAccelerationStructureType.BottomLevel,
                Flags = RaytracingAccelerationStructureBuildFlags.PreferFastTrace,
                Layout = ElementsLayout.Array,
                DescriptorsCount = (uint)geos.Length,
                GeometryDescriptions = geos,
            };
            var info = device.GetRaytracingAccelerationStructurePrebuildInfo(inputs);
            var result = CreateBuffer(name, info.ResultDataMaxSizeInBytes, HeapType.Default, ResourceFlags.AllowUnorderedAccess, ResourceStates.RaytracingAccelerationStructure);
            var scratch = CreateBuffer(name + " scratch", info.ScratchDataSizeInBytes, HeapType.Default, ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);
            temps.Add(scratch);
            cl.BuildRaytracingAccelerationStructure(new BuildRaytracingAccelerationStructureDescription
            {
                DestinationAccelerationStructureData = result.GPUVirtualAddress,
                Inputs = inputs,
                ScratchAccelerationStructureData = scratch.GPUVirtualAddress,
            });
            cl.ResourceBarrierUnorderedAccessView(result);
            return result;
        }

        cl.ResourceBarrierUnorderedAccessView(null);
        // static: geometry 0 = opaque, geometry 1 = alpha tested (non-opaque -> candidate loop in the shaders)
        g.BlasStatic = BuildBlas(s.Name + " BLAS static", (0, s.AlphaFirstIndex, true), (s.AlphaFirstIndex, s.GlassFirstIndex - s.AlphaFirstIndex, false));
        g.BlasGlass = BuildBlas(s.Name + " BLAS static glass", (s.GlassFirstIndex, s.StaticIndexCount - s.GlassFirstIndex, true));
        g.BlasSphere = BuildBlas(s.Name + " BLAS light sphere", (s.SphereFirstIndex, s.SphereIndexCount, true));
        if (s.HasMesh) g.BlasMesh = BuildBlas(s.Name + " BLAS mesh", (s.MeshFirstIndex, s.MeshIndexCount, true));

        // ---- top level: static + 1 per light (+ mesh, + static glass), rebuilt every frame ----
        g.TlasInstanceCount = 1 + s.Lights.Length + (g.BlasMesh != null ? 1 : 0) + (g.BlasGlass != null ? 1 : 0);
        g.TlasInputs = new BuildRaytracingAccelerationStructureInputs
        {
            Type = RaytracingAccelerationStructureType.TopLevel,
            Flags = RaytracingAccelerationStructureBuildFlags.PreferFastBuild,
            Layout = ElementsLayout.Array,
            DescriptorsCount = (uint)g.TlasInstanceCount,
        };
        var tinfo = device.GetRaytracingAccelerationStructurePrebuildInfo(g.TlasInputs);
        g.Tlas = CreateBuffer(s.Name + " TLAS", tinfo.ResultDataMaxSizeInBytes, HeapType.Default, ResourceFlags.AllowUnorderedAccess, ResourceStates.RaytracingAccelerationStructure);
        g.TlasScratch = CreateBuffer(s.Name + " TLAS scratch", Math.Max(tinfo.ScratchDataSizeInBytes, tinfo.UpdateScratchDataSizeInBytes), HeapType.Default, ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);
        g.TlasSlot = AllocSrvSlot();
        device.CreateShaderResourceView(null, new ShaderResourceViewDescription
        {
            Format = Format.Unknown,
            ViewDimension = ShaderResourceViewDimension.RaytracingAccelerationStructure,
            Shader4ComponentMapping = ShaderComponentMapping.Default,
            RaytracingAccelerationStructure = new RaytracingAccelerationStructureShaderResourceView { Location = g.Tlas.GPUVirtualAddress },
        }, SrvCpu(g.TlasSlot));

        // the bakes below ray-query the scene
        var prevGpu = gpu;
        gpu = g;
        var ring = rings[0];
        ring.Reset();
        BuildTlas(ring, lightsOn: false);

        // ---- SDF grids: decide resolutions + fill in the prims, upload them, then bake ----
        var grids = new List<(SdfMesh Mesh, int Dx, int Dy, int Dz, GpuTexture Sdf, GpuTexture Tris)>();
        foreach (var sm in s.SdfMeshes)
        {
            var prim = s.SdfPrims[sm.PrimIndex];
            var extent = prim.Size * 2;
            float voxel = MathF.Max(extent.X, MathF.Max(extent.Y, extent.Z)) / sm.Resolution;
            int dx = Math.Max(4, (int)MathF.Ceiling(extent.X / voxel));
            int dy = Math.Max(4, (int)MathF.Ceiling(extent.Y / voxel));
            int dz = Math.Max(4, (int)MathF.Ceiling(extent.Z / voxel));
            var min = prim.Center - prim.Size;
            var max = min + new Vector3(dx, dy, dz) * voxel;
            var sdf = CreateTexture3D($"{s.Name} SDF {sm.PrimIndex}", Format.R32_Float, dx, dy, dz, ResourceFlags.AllowUnorderedAccess, ResourceStates.UnorderedAccess);
            var tris = CreateTexture3D($"{s.Name} SDF triangles {sm.PrimIndex}", Format.R32_UInt, dx, dy, dz, ResourceFlags.AllowUnorderedAccess, ResourceStates.UnorderedAccess);
            sdf.Srv = AllocSrvSlot(); CreateSrv3D(sdf, sdf.Srv, Format.R32_Float);
            tris.Srv = AllocSrvSlot(); CreateSrv3D(tris, tris.Srv, Format.R32_UInt);
            g.SdfGrids.Add(sdf); g.SdfGrids.Add(tris);
            s.SdfPrims[sm.PrimIndex] = prim with
            {
                Center = (min + max) * 0.5f,
                Size = (max - min) * 0.5f,
                CosR = BitConverter.UInt32BitsToSingle((uint)sdf.Srv),
                SinR = BitConverter.UInt32BitsToSingle((uint)tris.Srv),
                Pad = new Vector2(BitConverter.UInt32BitsToSingle(sm.Signed ? 1u : 0u), voxel),
            };
            grids.Add((sm, dx, dy, dz, sdf, tris));
        }
        g.SdfBuf = Upload(s.Name + " SDF primitives", CollectionsMarshal.AsSpan(s.SdfPrims));
        g.SdfSlot = AllocSrvSlot(); CreateStructuredSrv(g.SdfBuf, g.SdfSlot, Math.Max(1, s.SdfPrims.Count), sizeof(SdfPrim));
        foreach (var gr in grids)
            BakeMeshSdf(ring, gr.Mesh, s.SdfPrims[gr.Mesh.PrimIndex], gr.Dx, gr.Dy, gr.Dz, gr.Sdf, gr.Tris, temps);

        // ---- per-vertex sky visibility (only scenes open to the sky need it) ----
        if (s.SkyIntensity > 0)
            BakeSkyVisibility(ring, g);

        cl.Close();
        queue.ExecuteCommandList(cl);
        WaitIdle();
        foreach (var t in temps) t.Dispose();
        gpu = prevGpu;
        return g;
    }

    FrameData BakeFrameData() => new()
    {
        VbIndex = (uint)gpu.VbSlot, IbIndex = (uint)gpu.IbSlot, MatIndex = (uint)gpu.MatSlot, TlasIndex = (uint)gpu.TlasSlot,
        MeshFirstIndex = (uint)scene.MeshFirstIndex, GlassFirstIndex = (uint)scene.GlassFirstIndex, AlphaFirstIndex = (uint)scene.AlphaFirstIndex,
        MeshMaterial = scene.HasMesh ? (uint)scene.MeshMaterialIndex : 0xFFFFFFFFu,
    };

    // ---------------------------------------------------------------------------------------------
    // Textures
    // ---------------------------------------------------------------------------------------------
    void LoadTextures(SceneGpu g, List<IDisposable> temps)
    {
        var list = g.Scene.Textures;
        var images = new (int W, int H, byte[][] Mips)[list.Count];
        Parallel.For(0, list.Count, i => images[i] = DecodeWithMips(list[i].Path));
        g.TextureSlots = new uint[list.Count];
        for (int i = 0; i < list.Count; i++)
        {
            var (w, h, mips) = images[i];
            var fmt = list[i].Srgb ? Format.B8G8R8A8_UNorm_SRgb : Format.B8G8R8A8_UNorm;
            var desc = ResourceDescription.Texture2D(fmt, (uint)w, (uint)h, 1, (ushort)mips.Length, 1, 0, ResourceFlags.None);
            var tex = new GpuTexture
            {
                Resource = device.CreateCommittedResource(HeapType.Default, HeapFlags.None, desc, ResourceStates.CopyDest, null),
                State = ResourceStates.CopyDest, Width = w, Height = h,
            };
            tex.Resource.Name = Path.GetFileName(list[i].Path);

            var layouts = new PlacedSubresourceFootPrint[mips.Length];
            var rows = new uint[mips.Length];
            var rowSizes = new ulong[mips.Length];
            device.GetCopyableFootprints(desc, 0, (uint)mips.Length, 0, layouts, rows, rowSizes, out ulong total);
            var upload = CreateBuffer("texture upload", total, HeapType.Upload, ResourceFlags.None, ResourceStates.GenericRead);
            temps.Add(upload);
            void* mapped;
            upload.Map(0, null, &mapped).CheckError();
            for (int m = 0; m < mips.Length; m++)
            {
                int rowBytes = (int)rowSizes[m];
                fixed (byte* src = mips[m])
                    for (int y = 0; y < rows[m]; y++)
                        Buffer.MemoryCopy(src + y * rowBytes, (byte*)mapped + layouts[m].Offset + (ulong)y * layouts[m].Footprint.RowPitch, rowBytes, rowBytes);
                cl.CopyTextureRegion(new TextureCopyLocation(tex.Resource, (uint)m), 0, 0, 0, new TextureCopyLocation(upload, layouts[m]), null);
            }
            upload.Unmap(0);
            Transition(tex, ResourceStates.AllShaderResource);

            tex.Srv = AllocSrvSlot();
            device.CreateShaderResourceView(tex.Resource, new ShaderResourceViewDescription
            {
                Format = fmt,
                ViewDimension = ShaderResourceViewDimension.Texture2D,
                Shader4ComponentMapping = ShaderComponentMapping.Default,
                Texture2D = new Texture2DShaderResourceView { MipLevels = (uint)mips.Length },
            }, SrvCpu(tex.Srv));
            g.Textures.Add(tex);
            g.TextureSlots[i] = (uint)tex.Srv;
        }
    }

    /// <summary>PNG/JPG/... -> BGRA8 + a box-filtered mip chain.</summary>
    static (int W, int H, byte[][] Mips) DecodeWithMips(string path)
    {
        using var bmp = new Bitmap(path);
        int w = bmp.Width, h = bmp.Height;
        var data = bmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
        var level0 = new byte[w * h * 4];
        for (int y = 0; y < h; y++)
            Marshal.Copy(data.Scan0 + y * data.Stride, level0, y * w * 4, w * 4);
        bmp.UnlockBits(data);

        var mips = new List<byte[]> { level0 };
        int cw = w, ch = h;
        while (cw > 1 || ch > 1)
        {
            int nw = Math.Max(1, cw / 2), nh = Math.Max(1, ch / 2);
            var src = mips[^1];
            var dst = new byte[nw * nh * 4];
            for (int y = 0; y < nh; y++)
            for (int x = 0; x < nw; x++)
            {
                int x0 = Math.Min(2 * x, cw - 1), x1 = Math.Min(2 * x + 1, cw - 1);
                int y0 = Math.Min(2 * y, ch - 1), y1 = Math.Min(2 * y + 1, ch - 1);
                for (int c = 0; c < 4; c++)
                    dst[(y * nw + x) * 4 + c] = (byte)((src[(y0 * cw + x0) * 4 + c] + src[(y0 * cw + x1) * 4 + c] +
                                                         src[(y1 * cw + x0) * 4 + c] + src[(y1 * cw + x1) * 4 + c] + 2) >> 2);
            }
            mips.Add(dst);
            cw = nw; ch = nh;
        }
        return (w, h, mips.ToArray());
    }

    // ---------------------------------------------------------------------------------------------
    // SDF mode: bake a triangle range into a 3D distance grid + nearest-triangle grid (SdfBake.hlsl)
    // ---------------------------------------------------------------------------------------------
    GpuTexture CreateTexture3D(string name, Format fmt, int w, int h, int d, ResourceFlags flags, ResourceStates state)
    {
        var desc = ResourceDescription.Texture3D(fmt, (uint)w, (uint)h, (ushort)d, 1, flags);
        var res = device.CreateCommittedResource(HeapType.Default, HeapFlags.None, desc, state, null);
        res.Name = name;
        return new GpuTexture { Resource = res, State = state, Width = w, Height = h, Depth = d };
    }

    void CreateUav3D(GpuTexture t, int slot, Format fmt)
    {
        t.Uav = slot;
        device.CreateUnorderedAccessView(t.Resource, null, new UnorderedAccessViewDescription
        {
            Format = fmt,
            ViewDimension = UnorderedAccessViewDimension.Texture3D,
            Texture3D = new Texture3DUnorderedAccessView { MipSlice = 0, FirstWSlice = 0, WSize = (uint)t.Depth },
        }, SrvCpu(slot));
    }

    void CreateSrv3D(GpuTexture t, int slot, Format fmt)
    {
        device.CreateShaderResourceView(t.Resource, new ShaderResourceViewDescription
        {
            Format = fmt,
            ViewDimension = ShaderResourceViewDimension.Texture3D,
            Shader4ComponentMapping = ShaderComponentMapping.Default,
            Texture3D = new Texture3DShaderResourceView { MipLevels = 1 },
        }, SrvCpu(slot));
    }

    void BakeMeshSdf(UploadRing ring, SdfMesh mesh, SdfPrim prim, int dx, int dy, int dz, GpuTexture sdf, GpuTexture tris, List<IDisposable> temps)
    {
        var seeds = new[]
        {
            CreateTexture3D("SDF bake seeds 0", Format.R32G32B32A32_Float, dx, dy, dz, ResourceFlags.AllowUnorderedAccess, ResourceStates.UnorderedAccess),
            CreateTexture3D("SDF bake seeds 1", Format.R32G32B32A32_Float, dx, dy, dz, ResourceFlags.AllowUnorderedAccess, ResourceStates.UnorderedAccess),
        };
        var keys = CreateTexture3D("SDF bake keys", Format.R32_UInt, dx, dy, dz, ResourceFlags.AllowUnorderedAccess, ResourceStates.UnorderedAccess);
        CreateUav3D(seeds[0], AllocSrvSlot(), Format.R32G32B32A32_Float);
        CreateUav3D(seeds[1], AllocSrvSlot(), Format.R32G32B32A32_Float);
        CreateUav3D(keys, AllocSrvSlot(), Format.R32_UInt);
        CreateUav3D(sdf, AllocSrvSlot(), Format.R32_Float);
        CreateUav3D(tris, AllocSrvSlot(), Format.R32_UInt);
        temps.AddRange(seeds);
        temps.Add(keys);

        var fd = BakeFrameData();
        fd.MeshFirstIndex = (uint)mesh.FirstIndex;
        fd.MeshSdfMin = prim.Center - prim.Size;
        fd.MeshSdfMax = prim.Center + prim.Size;
        fd.BakeFlags = mesh.Signed ? 1u : 0u;
        cl.SetComputeRootConstantBufferView(1, ring.Push(fd));
        cl.SetComputeRootConstantBufferView(2, ring.Push(new PassData()));

        void Barrier() => cl.ResourceBarrierUnorderedAccessView(null);
        void DispatchGrid() => cl.Dispatch((uint)(dx + 3) / 4, (uint)(dy + 3) / 4, (uint)(dz + 3) / 4);
        uint K = (uint)keys.Uav, S = (uint)sdf.Uav, T = (uint)tris.Uav;

        cl.SetPipelineState(psoBakeClear);
        RootConstants(true, (uint)seeds[0].Uav, (uint)seeds[1].Uav, K, S, 0, 0, 0, T);
        DispatchGrid();
        Barrier();

        int triCount = mesh.IndexCount / 3;
        int groups = (triCount + 63) / 64;
        int gx = Math.Min(groups, 65535), gy = (groups + gx - 1) / gx;
        cl.SetPipelineState(psoBakeSeed);
        for (uint pass = 0; pass < 2; pass++)
        {
            RootConstants(true, (uint)seeds[0].Uav, (uint)seeds[1].Uav, K, S, pass, (uint)triCount, (uint)(gx * 64), T);
            cl.Dispatch((uint)gx, (uint)gy, 1);
            Barrier();
        }

        // jump flooding: N/2, N/4, ..., 1, then one more 1 ("JFA+1") to fix the rare wrong pick
        cl.SetPipelineState(psoBakeJfa);
        int maxDim = Math.Max(dx, Math.Max(dy, dz));
        int step = 1;
        while (step * 2 < maxDim) step *= 2;
        var steps = new List<int>();
        for (; step >= 1; step /= 2) steps.Add(step);
        steps.Add(1);
        int src = 0;
        foreach (int st in steps)
        {
            RootConstants(true, (uint)seeds[src].Uav, (uint)seeds[1 - src].Uav, K, S, (uint)st, 0, 0, T);
            DispatchGrid();
            Barrier();
            src = 1 - src;
        }

        cl.SetPipelineState(psoBakeFinalize);
        RootConstants(true, (uint)seeds[src].Uav, 0, K, S, 0, 0, 0, T);
        DispatchGrid();
        Barrier();
        Transition(sdf, ResourceStates.AllShaderResource);
        Transition(tris, ResourceStates.AllShaderResource);
    }

    // ---------------------------------------------------------------------------------------------
    // Sky visibility per vertex (SceneBake.hlsl)
    // ---------------------------------------------------------------------------------------------
    void BakeSkyVisibility(UploadRing ring, SceneGpu g)
    {
        int uav = AllocSrvSlot();
        device.CreateUnorderedAccessView(g.Vb, null, new UnorderedAccessViewDescription
        {
            Format = Format.Unknown,
            ViewDimension = UnorderedAccessViewDimension.Buffer,
            Buffer = new BufferUnorderedAccessView { FirstElement = 0, NumElements = (uint)g.Scene.Vertices.Count, StructureByteStride = (uint)sizeof(Vertex) },
        }, SrvCpu(uav));

        cl.SetComputeRootConstantBufferView(1, ring.Push(BakeFrameData()));
        cl.SetComputeRootConstantBufferView(2, ring.Push(new PassData()));
        cl.ResourceBarrierTransition(g.Vb, ResourceStates.AllShaderResource, ResourceStates.UnorderedAccess, D3D12.ResourceBarrierAllSubResources, ResourceBarrierFlags.None);
        int count = g.Scene.Vertices.Count;
        int groups = (count + 63) / 64;
        int gx = Math.Min(groups, 65535), gy = (groups + gx - 1) / gx;
        // ~64 rays per vertex, fewer for huge scenes so the one-time bake stays around a second
        uint rays = (uint)Math.Clamp(SkyVisRays * 4_000_000L / Math.Max(count, 1), 16, SkyVisRays);
        cl.SetPipelineState(psoSkyVis);
        RootConstants(true, (uint)uav, (uint)count, rays, (uint)(gx * 64));
        cl.Dispatch((uint)gx, (uint)gy, 1);
        cl.ResourceBarrierTransition(g.Vb, ResourceStates.UnorderedAccess, ResourceStates.AllShaderResource, D3D12.ResourceBarrierAllSubResources, ResourceBarrierFlags.None);
    }

    // ---------------------------------------------------------------------------------------------
    // Top level acceleration structure (every frame: the lights move)
    // ---------------------------------------------------------------------------------------------
    void BuildTlas(UploadRing ring, bool lightsOn)
    {
        var g = gpu;
        var a = ring.Alloc((ulong)(g.TlasInstanceCount * sizeof(InstanceDesc)), 256);
        var inst = (InstanceDesc*)a.Cpu;
        int n = 0;
        inst[n++] = InstanceDesc.Create(Matrix4x4.Identity, 0, 0x01, g.BlasStatic.GPUVirtualAddress);
        for (int i = 0; i < scene.Lights.Length; i++)
        {
            var L = scene.Lights[i];
            var m = Matrix4x4.CreateScale(L.Radius) * Matrix4x4.CreateTranslation(L.Pos);
            // switched-off bulbs get mask 0: no ray can hit them
            inst[n++] = InstanceDesc.Create(m, (uint)(1 + i), lightsOn ? (byte)0x02 : (byte)0, g.BlasSphere.GPUVirtualAddress);
        }
        if (g.BlasMesh != null)
            inst[n++] = InstanceDesc.Create(Matrix4x4.Identity, 100, 0x04, g.BlasMesh.GPUVirtualAddress);    // INSTANCE_MESH, MASK_MESH
        if (g.BlasGlass != null)
            inst[n++] = InstanceDesc.Create(Matrix4x4.Identity, 101, 0x08, g.BlasGlass.GPUVirtualAddress);   // INSTANCE_GLASS, MASK_GLASS
        g.TlasInputs.InstanceDescriptions = a.Gpu;
        cl.BuildRaytracingAccelerationStructure(new BuildRaytracingAccelerationStructureDescription
        {
            DestinationAccelerationStructureData = g.Tlas.GPUVirtualAddress,
            Inputs = g.TlasInputs,
            ScratchAccelerationStructureData = g.TlasScratch.GPUVirtualAddress,
        });
        cl.ResourceBarrierUnorderedAccessView(g.Tlas);
    }
}
