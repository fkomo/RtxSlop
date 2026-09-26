using System.Numerics;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using Vortice.Direct3D;
using Vortice.Direct3D12;
using Vortice.Direct3D12.Debug;
using Vortice.Dxc;
using Vortice.DXGI;
using Vortice.Mathematics;
using ShaderResourceViewDimension = Vortice.Direct3D12.ShaderResourceViewDimension;

namespace RtxSlop;

public sealed unsafe class Renderer : IDisposable
{
    const int FrameCount = 2;          // frames in flight
    const int BackBufferCount = 3;
    const int ShadowSize = 1024;
    const int ProbeSize = 256;
    const float ShadowNear = 0.05f, ShadowFar = 30f;
    const int HudWidth = 560, HudHeight = 216;
    const Format HdrFormat = Format.R16G16B16A16_Float;
    const int MsaaSamples = 4;

    // Fixed bindless descriptor slots
    const int SlotVB = 0, SlotIB = 1, SlotMat = 2, SlotTlas = 3, SlotShadow = 4, SlotProbe = 5, SlotHud = 6;
    const int SlotHdrSrv = 10, SlotHdrUav = 11;
    public const int SlotRefl0 = 12;
    const int SlotPtBase = 20;   // PT textures: srv = base + 2k, uav = base + 2k + 1 (slots 20..39)
    const int SlotGBufPos = 40, SlotGBufNrm = 41, SlotSdf = 42, SlotSceneColor = 43;
    const int SlotProbe1 = 7;
    const int SlotBakeSeed0 = 44, SlotBakeSeed1 = 45, SlotBakeKeys = 46, SlotBakeSdfUav = 47, SlotMeshSdf = 48;
    const int SdfGridResolution = 128;   // voxels along the longest axis of the mesh bounds
    public static uint ProbeSlot(int i) => (uint)(i == 0 ? SlotProbe : SlotProbe1);
    const Format GBufPosFormat = Format.R32G32B32A32_Float, GBufNrmFormat = Format.R32G32B32A32_Float;

    public string AdapterName { get; }
    public int Width { get; private set; }
    public int Height { get; private set; }
    public double GpuTimeMs { get; private set; }

    readonly Scene scene;
    readonly bool debug;
    readonly IDXGIFactory6 factory;
    readonly ID3D12Device5 device;
    readonly ID3D12CommandQueue queue;
    readonly IDXGISwapChain3 swapChain;
    readonly bool tearingSupported;
    readonly ID3D12InfoQueue infoQueue;
    readonly ID3D12CommandAllocator[] allocators = new ID3D12CommandAllocator[FrameCount];
    readonly ID3D12GraphicsCommandList4 cl;
    readonly ID3D12Fence fence;
    readonly AutoResetEvent fenceEvent = new(false);
    ulong fenceValue;
    readonly ulong[] frameFence = new ulong[FrameCount];
    readonly UploadRing[] rings = new UploadRing[FrameCount];
    ulong frameCounter;

    readonly ID3D12DescriptorHeap srvHeap, rtvHeap, dsvHeap;
    readonly uint srvInc, rtvInc, dsvInc;
    int nextRtv, nextDsv;

    ID3D12RootSignature rootSig;
    ID3D12PipelineState psoMain, psoMainMsaa, psoMainMirrored, psoShadow, psoTonemap, psoRayTrace, psoPathTrace, psoTemporal, psoAtrous, psoGBuffer, psoSdf,
        psoGlass, psoGlassMsaa, psoBakeClear, psoBakeSeed, psoBakeJfa, psoBakeFinalize;

    // static scene data
    ID3D12Resource vb, ib, matBuf, sdfBuf;
    ID3D12Resource blasStatic, blasGlass, blasSphere, blasMesh, tlas, tlasScratch;
    GpuTexture meshSdf;
    Vector3 meshSdfMin, meshSdfMax;
    readonly List<(int Index, Material Material)> pendingMaterials = new();
    BuildRaytracingAccelerationStructureInputs tlasInputs;

    // fixed size render targets
    GpuTexture shadowMap, probeDepth, hud;
    GpuTexture[] probes = new GpuTexture[2];   // 0 = chrome sphere, 1 = mesh
    CpuDescriptorHandle[] shadowDsv = new CpuDescriptorHandle[18];
    CpuDescriptorHandle[,] probeRtv = new CpuDescriptorHandle[2, 6];

    // size dependent
    GpuTexture[] backBuffers = new GpuTexture[BackBufferCount];
    GpuTexture depth, hdr, msaaColor, msaaDepth, sceneColor;
    GpuTexture[] refl = new GpuTexture[2];
    GpuTexture ptNoisy, ptAlbedo, gbufPos, gbufNrm;
    GpuTexture[] ptGuide = new GpuTexture[2], ptHist = new GpuTexture[2], ptMom = new GpuTexture[2], ptFilt = new GpuTexture[2];
    CpuDescriptorHandle[] backBufferRtvs = new CpuDescriptorHandle[BackBufferCount];

    // timing
    ID3D12QueryHeap queryHeap;
    ID3D12Resource queryReadback;
    double timestampPeriodMs;

    // per-frame state
    Matrix4x4 prevViewProj;
    Vector3 prevCamPos;
    bool historyValid;
    RenderMode lastMode = (RenderMode)(-1);
    byte[] pendingHud;
    string pendingCapture;

    public Renderer(IntPtr hwnd, int width, int height, Scene scene, string shaderDir, bool debug)
    {
        this.scene = scene;
        this.debug = debug;
        Width = Math.Max(1, width);
        Height = Math.Max(1, height);

        if (debug && D3D12.D3D12GetDebugInterface(out ID3D12Debug dbg).Success)
        {
            dbg.EnableDebugLayer();
            dbg.Dispose();
        }

        try { factory = DXGI.CreateDXGIFactory2<IDXGIFactory6>(debug); }
        catch when (debug) { factory = DXGI.CreateDXGIFactory2<IDXGIFactory6>(false); }   // Graphics Tools not installed
        using (var adapter = factory.EnumAdapterByGpuPreference<IDXGIAdapter1>(0, GpuPreference.HighPerformance))
        {
            AdapterName = adapter.Description1.Description;
            device = D3D12.D3D12CreateDevice<ID3D12Device5>(adapter, FeatureLevel.Level_12_1);
        }

        var opt5 = device.Options5;
        if (opt5.RaytracingTier < RaytracingTier.Tier1_1)
            throw new NotSupportedException($"{AdapterName}: DXR 1.1 (inline ray tracing) is required.");
        if (device.CheckHighestShaderModel(ShaderModel.Model6_6) < ShaderModel.Model6_6)
            throw new NotSupportedException($"{AdapterName}: Shader Model 6.6 (bindless) is required.");

        if (debug)
            infoQueue = device.QueryInterfaceOrNull<ID3D12InfoQueue>();

        queue = device.CreateCommandQueue(new CommandQueueDescription(CommandListType.Direct));
        queue.Name = "Main queue";

        using (var f5 = factory.QueryInterfaceOrNull<IDXGIFactory5>())
            tearingSupported = f5 != null && f5.PresentAllowTearing;

        var scDesc = new SwapChainDescription1
        {
            Width = (uint)Width,
            Height = (uint)Height,
            Format = Format.R8G8B8A8_UNorm,
            BufferCount = BackBufferCount,
            BufferUsage = Usage.RenderTargetOutput,
            SampleDescription = new SampleDescription(1, 0),
            Scaling = Scaling.Stretch,
            SwapEffect = SwapEffect.FlipDiscard,
            AlphaMode = AlphaMode.Ignore,
            Flags = tearingSupported ? SwapChainFlags.AllowTearing : SwapChainFlags.None,
        };
        using (var sc1 = factory.CreateSwapChainForHwnd(queue, hwnd, scDesc))
            swapChain = sc1.QueryInterface<IDXGISwapChain3>();
        factory.MakeWindowAssociation(hwnd, WindowAssociationFlags.IgnoreAltEnter);

        srvHeap = device.CreateDescriptorHeap(new DescriptorHeapDescription(DescriptorHeapType.ConstantBufferViewShaderResourceViewUnorderedAccessView, 128, DescriptorHeapFlags.ShaderVisible));
        rtvHeap = device.CreateDescriptorHeap(new DescriptorHeapDescription(DescriptorHeapType.RenderTargetView, 32, DescriptorHeapFlags.None));
        dsvHeap = device.CreateDescriptorHeap(new DescriptorHeapDescription(DescriptorHeapType.DepthStencilView, 32, DescriptorHeapFlags.None));
        srvInc = device.GetDescriptorHandleIncrementSize(DescriptorHeapType.ConstantBufferViewShaderResourceViewUnorderedAccessView);
        rtvInc = device.GetDescriptorHandleIncrementSize(DescriptorHeapType.RenderTargetView);
        dsvInc = device.GetDescriptorHandleIncrementSize(DescriptorHeapType.DepthStencilView);

        for (int i = 0; i < FrameCount; i++)
        {
            allocators[i] = device.CreateCommandAllocator(CommandListType.Direct);
            rings[i] = new UploadRing(device, 16 * 1024 * 1024);
        }
        cl = device.CreateCommandList<ID3D12GraphicsCommandList4>(0, CommandListType.Direct, allocators[0], null);
        cl.Close();
        fence = device.CreateFence(0);

        queryHeap = device.CreateQueryHeap<ID3D12QueryHeap>(new QueryHeapDescription(QueryHeapType.Timestamp, 2 * FrameCount, 0));
        queryReadback = device.CreateCommittedResource(HeapType.Readback, ResourceDescription.Buffer(16 * FrameCount), ResourceStates.CopyDest);
        queue.GetTimestampFrequency(out ulong freq).CheckError();
        timestampPeriodMs = 1000.0 / freq;

        CreatePipelines(shaderDir);
        CreateSceneResources();
        CreateFixedTargets();
        CreateSizeDependent();
    }

    // =============================================================================================
    // Descriptors
    // =============================================================================================
    CpuDescriptorHandle SrvCpu(int slot) => srvHeap.GetCPUDescriptorHandleForHeapStart() + (int)(slot * srvInc);
    CpuDescriptorHandle AllocRtv() => rtvHeap.GetCPUDescriptorHandleForHeapStart() + (int)(nextRtv++ * rtvInc);
    CpuDescriptorHandle AllocDsv() => dsvHeap.GetCPUDescriptorHandleForHeapStart() + (int)(nextDsv++ * dsvInc);

    void CreateStructuredSrv(ID3D12Resource res, int slot, int count, int stride)
    {
        device.CreateShaderResourceView(res, new ShaderResourceViewDescription
        {
            Format = Format.Unknown,
            ViewDimension = ShaderResourceViewDimension.Buffer,
            Shader4ComponentMapping = ShaderComponentMapping.Default,
            Buffer = new BufferShaderResourceView { FirstElement = 0, NumElements = (uint)count, StructureByteStride = (uint)stride },
        }, SrvCpu(slot));
    }

    void CreateTexSrv(GpuTexture t, int slot, Format fmt)
    {
        t.Srv = slot;
        device.CreateShaderResourceView(t.Resource, new ShaderResourceViewDescription
        {
            Format = fmt,
            ViewDimension = ShaderResourceViewDimension.Texture2D,
            Shader4ComponentMapping = ShaderComponentMapping.Default,
            Texture2D = new Texture2DShaderResourceView { MipLevels = 1 },
        }, SrvCpu(slot));
    }

    void CreateTexUav(GpuTexture t, int slot, Format fmt)
    {
        t.Uav = slot;
        device.CreateUnorderedAccessView(t.Resource, null, new UnorderedAccessViewDescription
        {
            Format = fmt,
            ViewDimension = UnorderedAccessViewDimension.Texture2D,
        }, SrvCpu(slot));
    }

    // =============================================================================================
    // Resources
    // =============================================================================================
    GpuTexture CreateTexture(string name, Format fmt, int w, int h, int arraySize, ResourceFlags flags, ResourceStates state, ClearValue? clear = null, int samples = 1)
    {
        var desc = ResourceDescription.Texture2D(fmt, (uint)w, (uint)h, (ushort)arraySize, 1, (uint)samples, 0, flags);
        var res = device.CreateCommittedResource(HeapType.Default, HeapFlags.None, desc, state, clear);
        res.Name = name;
        return new GpuTexture { Resource = res, State = state, Width = w, Height = h };
    }

    ID3D12Resource CreateBuffer(string name, ulong size, HeapType heap, ResourceFlags flags, ResourceStates state)
    {
        var res = device.CreateCommittedResource(heap, HeapFlags.None, ResourceDescription.Buffer(size, flags), state, null);
        res.Name = name;
        return res;
    }

    void Transition(GpuTexture t, ResourceStates to)
    {
        if (t.State == to) return;
        cl.ResourceBarrierTransition(t.Resource, t.State, to, D3D12.ResourceBarrierAllSubResources, ResourceBarrierFlags.None);
        t.State = to;
    }

    void CreateSceneResources()
    {
        var uploads = new List<IDisposable>();
        ID3D12Resource Upload<T>(string name, List<T> data, ResourceFlags flags = ResourceFlags.None) where T : unmanaged
        {
            ulong size = (ulong)(data.Count * sizeof(T));
            var dst = CreateBuffer(name, size, HeapType.Default, flags, ResourceStates.Common);
            var src = CreateBuffer(name + " upload", size, HeapType.Upload, ResourceFlags.None, ResourceStates.GenericRead);
            var span = CollectionsMarshal.AsSpan(data);
            src.SetData<T>(span);
            uploads.Add(src);
            cl.CopyBufferRegion(dst, 0, src, 0, size);
            cl.ResourceBarrierTransition(dst, ResourceStates.CopyDest, ResourceStates.AllShaderResource, D3D12.ResourceBarrierAllSubResources, ResourceBarrierFlags.None);
            return dst;
        }

        allocators[0].Reset();
        cl.Reset(allocators[0], null);

        vb = Upload("Vertices", scene.Vertices);
        ib = Upload("Indices", scene.Indices);
        matBuf = Upload("Materials", scene.Materials);
        CreateStructuredSrv(vb, SlotVB, scene.Vertices.Count, sizeof(Vertex));
        CreateStructuredSrv(ib, SlotIB, scene.Indices.Count, 4);
        CreateStructuredSrv(matBuf, SlotMat, scene.Materials.Count, sizeof(Material));
        sdfBuf = Upload("SDF primitives", scene.SdfPrims);
        CreateStructuredSrv(sdfBuf, SlotSdf, scene.SdfPrims.Count, sizeof(SdfPrim));

        // ---- bottom level acceleration structures ----
        ID3D12Resource BuildBlas(string name, int firstIndex, int indexCount)
        {
            var geo = new RaytracingGeometryDescription(new RaytracingGeometryTrianglesDescription
            {
                VertexBuffer = new GpuVirtualAddressAndStride(vb.GPUVirtualAddress, (ulong)sizeof(Vertex)),
                VertexFormat = Format.R32G32B32_Float,
                VertexCount = (uint)scene.Vertices.Count,
                IndexBuffer = ib.GPUVirtualAddress + (ulong)firstIndex * 4,
                IndexFormat = Format.R32_UInt,
                IndexCount = (uint)indexCount,
                Transform3x4 = 0,
            }, RaytracingGeometryFlags.Opaque);
            var inputs = new BuildRaytracingAccelerationStructureInputs
            {
                Type = RaytracingAccelerationStructureType.BottomLevel,
                Flags = RaytracingAccelerationStructureBuildFlags.PreferFastTrace,
                Layout = ElementsLayout.Array,
                DescriptorsCount = 1,
                GeometryDescriptions = new[] { geo },
            };
            var info = device.GetRaytracingAccelerationStructurePrebuildInfo(inputs);
            var result = CreateBuffer(name, info.ResultDataMaxSizeInBytes, HeapType.Default, ResourceFlags.AllowUnorderedAccess, ResourceStates.RaytracingAccelerationStructure);
            var scratch = CreateBuffer(name + " scratch", info.ScratchDataSizeInBytes, HeapType.Default, ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);
            uploads.Add(scratch);
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
        blasStatic = BuildBlas("BLAS static", 0, scene.GlassFirstIndex);
        if (scene.StaticIndexCount > scene.GlassFirstIndex)
            blasGlass = BuildBlas("BLAS static glass", scene.GlassFirstIndex, scene.StaticIndexCount - scene.GlassFirstIndex);
        blasSphere = BuildBlas("BLAS light sphere", scene.SphereFirstIndex, scene.SphereIndexCount);
        if (scene.HasMesh)
            blasMesh = BuildBlas("BLAS mesh", scene.MeshFirstIndex, scene.MeshIndexCount);

        // ---- top level: static room + 1 instance per light (+ mesh), rebuilt every frame ----
        tlasInputs = new BuildRaytracingAccelerationStructureInputs
        {
            Type = RaytracingAccelerationStructureType.TopLevel,
            Flags = RaytracingAccelerationStructureBuildFlags.PreferFastBuild,
            Layout = ElementsLayout.Array,
            DescriptorsCount = (uint)TlasInstanceCount,
        };
        var tinfo = device.GetRaytracingAccelerationStructurePrebuildInfo(tlasInputs);
        tlas = CreateBuffer("TLAS", tinfo.ResultDataMaxSizeInBytes, HeapType.Default, ResourceFlags.AllowUnorderedAccess, ResourceStates.RaytracingAccelerationStructure);
        tlasScratch = CreateBuffer("TLAS scratch", Math.Max(tinfo.ScratchDataSizeInBytes, tinfo.UpdateScratchDataSizeInBytes), HeapType.Default, ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);
        device.CreateShaderResourceView(null, new ShaderResourceViewDescription
        {
            Format = Format.Unknown,
            ViewDimension = ShaderResourceViewDimension.RaytracingAccelerationStructure,
            Shader4ComponentMapping = ShaderComponentMapping.Default,
            RaytracingAccelerationStructure = new RaytracingAccelerationStructureShaderResourceView { Location = tlas.GPUVirtualAddress },
        }, SrvCpu(SlotTlas));

        if (scene.HasMesh)
            BakeMeshSdf(uploads);

        cl.Close();
        queue.ExecuteCommandList(cl);
        WaitIdle();
        foreach (var u in uploads) u.Dispose();
    }

    int TlasInstanceCount => 1 + scene.Lights.Length + (scene.HasMesh ? 1 : 0) + (blasGlass != null ? 1 : 0);

    // ---------------------------------------------------------------------------------------------
    // SDF mode: bake the mesh into a 3D signed distance texture (SdfBake.hlsl), once at startup
    // ---------------------------------------------------------------------------------------------
    GpuTexture CreateTexture3D(string name, Format fmt, int w, int h, int d, ResourceFlags flags, ResourceStates state)
    {
        var desc = ResourceDescription.Texture3D(fmt, (uint)w, (uint)h, (ushort)d, 1, flags);
        var res = device.CreateCommittedResource(HeapType.Default, HeapFlags.None, desc, state, null);
        res.Name = name;
        return new GpuTexture { Resource = res, State = state, Width = w, Height = h };
    }

    void CreateUav3D(GpuTexture t, int slot, Format fmt, int depth3D)
    {
        t.Uav = slot;
        device.CreateUnorderedAccessView(t.Resource, null, new UnorderedAccessViewDescription
        {
            Format = fmt,
            ViewDimension = UnorderedAccessViewDimension.Texture3D,
            Texture3D = new Texture3DUnorderedAccessView { MipSlice = 0, FirstWSlice = 0, WSize = (uint)depth3D },
        }, SrvCpu(slot));
    }

    void BakeMeshSdf(List<IDisposable> temporaries)
    {
        var prim = scene.SdfPrims.Last(p => p.Type == SdfPrim.Mesh);
        var extent = prim.Size * 2;
        float voxel = MathF.Max(extent.X, MathF.Max(extent.Y, extent.Z)) / SdfGridResolution;
        int dx = Math.Max(4, (int)MathF.Ceiling(extent.X / voxel));
        int dy = Math.Max(4, (int)MathF.Ceiling(extent.Y / voxel));
        int dz = Math.Max(4, (int)MathF.Ceiling(extent.Z / voxel));
        meshSdfMin = prim.Center - prim.Size;
        meshSdfMax = meshSdfMin + new Vector3(dx, dy, dz) * voxel;

        var seeds = new[]
        {
            CreateTexture3D("SDF bake seeds 0", Format.R32G32B32A32_Float, dx, dy, dz, ResourceFlags.AllowUnorderedAccess, ResourceStates.UnorderedAccess),
            CreateTexture3D("SDF bake seeds 1", Format.R32G32B32A32_Float, dx, dy, dz, ResourceFlags.AllowUnorderedAccess, ResourceStates.UnorderedAccess),
        };
        var keys = CreateTexture3D("SDF bake keys", Format.R32_UInt, dx, dy, dz, ResourceFlags.AllowUnorderedAccess, ResourceStates.UnorderedAccess);
        meshSdf = CreateTexture3D("Mesh SDF", Format.R32_Float, dx, dy, dz, ResourceFlags.AllowUnorderedAccess, ResourceStates.UnorderedAccess);
        CreateUav3D(seeds[0], SlotBakeSeed0, Format.R32G32B32A32_Float, dz);
        CreateUav3D(seeds[1], SlotBakeSeed1, Format.R32G32B32A32_Float, dz);
        CreateUav3D(keys, SlotBakeKeys, Format.R32_UInt, dz);
        CreateUav3D(meshSdf, SlotBakeSdfUav, Format.R32_Float, dz);
        meshSdf.Srv = SlotMeshSdf;
        device.CreateShaderResourceView(meshSdf.Resource, new ShaderResourceViewDescription
        {
            Format = Format.R32_Float,
            ViewDimension = ShaderResourceViewDimension.Texture3D,
            Shader4ComponentMapping = ShaderComponentMapping.Default,
            Texture3D = new Texture3DShaderResourceView { MipLevels = 1 },
        }, SrvCpu(SlotMeshSdf));
        temporaries.AddRange(seeds);
        temporaries.Add(keys);

        // the sign test ray-queries the mesh, so the TLAS must exist
        var ring = rings[0];
        ring.Reset();
        cl.SetDescriptorHeaps(srvHeap);
        cl.SetComputeRootSignature(rootSig);
        BuildTlas(ring);
        var fd = new FrameData
        {
            VbIndex = SlotVB, IbIndex = SlotIB, MatIndex = SlotMat, TlasIndex = SlotTlas,
            MeshFirstIndex = (uint)scene.MeshFirstIndex, MeshSdfMin = meshSdfMin, MeshSdfMax = meshSdfMax,
        };
        cl.SetComputeRootConstantBufferView(1, ring.Push(fd));
        cl.SetComputeRootConstantBufferView(2, ring.Push(new PassData()));

        void Barrier() => cl.ResourceBarrierUnorderedAccessView(null);
        void DispatchGrid() => cl.Dispatch((uint)(dx + 3) / 4, (uint)(dy + 3) / 4, (uint)(dz + 3) / 4);

        cl.SetPipelineState(psoBakeClear);
        RootConstants(true, SlotBakeSeed0, SlotBakeSeed1, SlotBakeKeys, SlotBakeSdfUav, 0, 0, 0);
        DispatchGrid();
        Barrier();

        int tris = scene.MeshIndexCount / 3;
        int groups = (tris + 63) / 64;
        int gx = Math.Min(groups, 65535), gy = (groups + gx - 1) / gx;
        cl.SetPipelineState(psoBakeSeed);
        for (uint pass = 0; pass < 2; pass++)
        {
            RootConstants(true, SlotBakeSeed0, SlotBakeSeed1, SlotBakeKeys, SlotBakeSdfUav, pass, (uint)tris, (uint)(gx * 64));
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
            RootConstants(true, (uint)seeds[src].Uav, (uint)seeds[1 - src].Uav, SlotBakeKeys, SlotBakeSdfUav, (uint)st, 0, 0);
            DispatchGrid();
            Barrier();
            src = 1 - src;
        }

        cl.SetPipelineState(psoBakeFinalize);
        RootConstants(true, (uint)seeds[src].Uav, 0, SlotBakeKeys, SlotBakeSdfUav, 0, 0, 0);
        DispatchGrid();
        Transition(meshSdf, ResourceStates.AllShaderResource);
    }

    void CreateFixedTargets()
    {
        // omnidirectional shadow maps: 3 lights x 6 faces in one cube array
        int lights = scene.Lights.Length;
        shadowMap = CreateTexture("Shadow cube array", Format.R32_Typeless, ShadowSize, ShadowSize, 6 * lights,
            ResourceFlags.AllowDepthStencil, ResourceStates.AllShaderResource, new ClearValue(Format.D32_Float, 1.0f, 0));
        for (int i = 0; i < 6 * lights; i++)
        {
            shadowDsv[i] = AllocDsv();
            device.CreateDepthStencilView(shadowMap.Resource, new DepthStencilViewDescription
            {
                Format = Format.D32_Float,
                ViewDimension = DepthStencilViewDimension.Texture2DArray,
                Texture2DArray = new Texture2DArrayDepthStencilView { MipSlice = 0, FirstArraySlice = (uint)i, ArraySize = 1 },
            }, shadowDsv[i]);
        }
        shadowMap.Srv = SlotShadow;
        device.CreateShaderResourceView(shadowMap.Resource, new ShaderResourceViewDescription
        {
            Format = Format.R32_Float,
            ViewDimension = ShaderResourceViewDimension.TextureCubeArray,
            Shader4ComponentMapping = ShaderComponentMapping.Default,
            TextureCubeArray = new TextureCubeArrayShaderResourceView { MipLevels = 1, First2DArrayFace = 0, NumCubes = (uint)lights },
        }, SrvCpu(SlotShadow));

        // dynamic reflection probes (cube maps rendered from the chrome sphere's / the mesh's center)
        for (int p = 0; p < 2; p++)
        {
            var probe = probes[p] = CreateTexture($"Reflection probe {p}", HdrFormat, ProbeSize, ProbeSize, 6, ResourceFlags.AllowRenderTarget,
                ResourceStates.AllShaderResource, new ClearValue(HdrFormat, new Color4(0, 0, 0, 1)));
            for (int f = 0; f < 6; f++)
            {
                probeRtv[p, f] = AllocRtv();
                device.CreateRenderTargetView(probe.Resource, new RenderTargetViewDescription
                {
                    Format = HdrFormat,
                    ViewDimension = RenderTargetViewDimension.Texture2DArray,
                    Texture2DArray = new Texture2DArrayRenderTargetView { MipSlice = 0, FirstArraySlice = (uint)f, ArraySize = 1 },
                }, probeRtv[p, f]);
            }
            probe.Srv = (int)ProbeSlot(p);
            device.CreateShaderResourceView(probe.Resource, new ShaderResourceViewDescription
            {
                Format = HdrFormat,
                ViewDimension = ShaderResourceViewDimension.TextureCube,
                Shader4ComponentMapping = ShaderComponentMapping.Default,
                TextureCube = new TextureCubeShaderResourceView { MipLevels = 1 },
            }, SrvCpu(probe.Srv));
        }
        probeDepth = CreateTexture("Probe depth", Format.D32_Float, ProbeSize, ProbeSize, 1, ResourceFlags.AllowDepthStencil,
            ResourceStates.DepthWrite, new ClearValue(Format.D32_Float, 1.0f, 0));
        probeDepth.Dsv = AllocDsv();
        device.CreateDepthStencilView(probeDepth.Resource, null, probeDepth.Dsv);

        hud = CreateTexture("HUD", Format.B8G8R8A8_UNorm, HudWidth, HudHeight, 1, ResourceFlags.None, ResourceStates.CopyDest);
        CreateTexSrv(hud, SlotHud, Format.B8G8R8A8_UNorm);
        pendingHud = new byte[HudWidth * HudHeight * 4];
    }

    void CreateSizeDependent()
    {
        nextRtv = 12;   // first 12 RTVs belong to the two probes
        nextDsv = 18 + 1;

        for (int i = 0; i < BackBufferCount; i++)
        {
            var bb = swapChain.GetBuffer<ID3D12Resource>((uint)i);
            backBuffers[i] = new GpuTexture { Resource = bb, State = ResourceStates.Present, Width = Width, Height = Height };
            backBufferRtvs[i] = AllocRtv();
            device.CreateRenderTargetView(bb, new RenderTargetViewDescription
            {
                Format = Format.R8G8B8A8_UNorm_SRgb,
                ViewDimension = RenderTargetViewDimension.Texture2D,
            }, backBufferRtvs[i]);
        }

        depth = CreateTexture("Depth", Format.D32_Float, Width, Height, 1, ResourceFlags.AllowDepthStencil, ResourceStates.DepthWrite, new ClearValue(Format.D32_Float, 1.0f, 0));
        depth.Dsv = AllocDsv();
        device.CreateDepthStencilView(depth.Resource, null, depth.Dsv);

        var black = new ClearValue(HdrFormat, new Color4(0, 0, 0, 1));
        msaaColor = CreateTexture("MSAA color", HdrFormat, Width, Height, 1, ResourceFlags.AllowRenderTarget, ResourceStates.ResolveSource, black, MsaaSamples);
        msaaColor.Rtv = AllocRtv();
        device.CreateRenderTargetView(msaaColor.Resource, null, msaaColor.Rtv);
        msaaDepth = CreateTexture("MSAA depth", Format.D32_Float, Width, Height, 1, ResourceFlags.AllowDepthStencil, ResourceStates.DepthWrite, new ClearValue(Format.D32_Float, 1.0f, 0), MsaaSamples);
        msaaDepth.Dsv = AllocDsv();
        device.CreateDepthStencilView(msaaDepth.Resource, null, msaaDepth.Dsv);

        sceneColor = CreateTexture("Scene color copy", HdrFormat, Width, Height, 1, ResourceFlags.None, ResourceStates.AllShaderResource);
        CreateTexSrv(sceneColor, SlotSceneColor, HdrFormat);

        hdr = CreateTexture("HDR", HdrFormat, Width, Height, 1, ResourceFlags.AllowRenderTarget | ResourceFlags.AllowUnorderedAccess, ResourceStates.AllShaderResource, black);
        hdr.Rtv = AllocRtv();
        device.CreateRenderTargetView(hdr.Resource, null, hdr.Rtv);
        CreateTexSrv(hdr, SlotHdrSrv, HdrFormat);
        CreateTexUav(hdr, SlotHdrUav, HdrFormat);

        for (int i = 0; i < 2; i++)
        {
            refl[i] = CreateTexture($"Planar reflection {i}", HdrFormat, Width, Height, 1, ResourceFlags.AllowRenderTarget, ResourceStates.AllShaderResource, black);
            refl[i].Rtv = AllocRtv();
            device.CreateRenderTargetView(refl[i].Resource, null, refl[i].Rtv);
            CreateTexSrv(refl[i], SlotRefl0 + i, HdrFormat);
        }

        int k = 0;
        GpuTexture Pt(string name, Format fmt)
        {
            var t = CreateTexture(name, fmt, Width, Height, 1, ResourceFlags.AllowUnorderedAccess, ResourceStates.AllShaderResource);
            CreateTexSrv(t, SlotPtBase + 2 * k, fmt);
            CreateTexUav(t, SlotPtBase + 2 * k + 1, fmt);
            k++;
            return t;
        }
        ptNoisy = Pt("PT radiance", Format.R16G16B16A16_Float);
        ptAlbedo = Pt("PT albedo", Format.R16G16B16A16_Float);
        for (int i = 0; i < 2; i++)
        {
            ptGuide[i] = Pt($"PT guide {i}", Format.R32G32B32A32_Float);
            ptHist[i] = Pt($"PT history {i}", Format.R16G16B16A16_Float);
            ptMom[i] = Pt($"PT moments {i}", Format.R32G32B32A32_Float);
            ptFilt[i] = Pt($"PT filter {i}", Format.R16G16B16A16_Float);
        }

        // hybrid mode G-buffer (rasterized primary visibility)
        var zero = new Color4(0, 0, 0, 0);
        gbufPos = CreateTexture("GBuffer position", GBufPosFormat, Width, Height, 1, ResourceFlags.AllowRenderTarget, ResourceStates.AllShaderResource, new ClearValue(GBufPosFormat, zero));
        gbufPos.Rtv = AllocRtv();
        device.CreateRenderTargetView(gbufPos.Resource, null, gbufPos.Rtv);
        CreateTexSrv(gbufPos, SlotGBufPos, GBufPosFormat);
        gbufNrm = CreateTexture("GBuffer normal+material", GBufNrmFormat, Width, Height, 1, ResourceFlags.AllowRenderTarget, ResourceStates.AllShaderResource, new ClearValue(GBufNrmFormat, zero));
        gbufNrm.Rtv = AllocRtv();
        device.CreateRenderTargetView(gbufNrm.Resource, null, gbufNrm.Rtv);
        CreateTexSrv(gbufNrm, SlotGBufNrm, GBufNrmFormat);

        historyValid = false;
    }

    void ReleaseSizeDependent()
    {
        foreach (var t in backBuffers) t?.Dispose();
        depth?.Dispose(); hdr?.Dispose(); msaaColor?.Dispose(); msaaDepth?.Dispose(); sceneColor?.Dispose();
        foreach (var t in refl) t?.Dispose();
        ptNoisy?.Dispose(); ptAlbedo?.Dispose(); gbufPos?.Dispose(); gbufNrm?.Dispose();
        foreach (var arr in new[] { ptGuide, ptHist, ptMom, ptFilt })
            foreach (var t in arr) t?.Dispose();
    }

    public void Resize(int width, int height)
    {
        width = Math.Max(1, width);
        height = Math.Max(1, height);
        if (width == Width && height == Height) return;
        WaitIdle();
        ReleaseSizeDependent();
        Width = width;
        Height = height;
        swapChain.ResizeBuffers(BackBufferCount, (uint)Width, (uint)Height, Format.R8G8B8A8_UNorm,
            tearingSupported ? SwapChainFlags.AllowTearing : SwapChainFlags.None).CheckError();
        CreateSizeDependent();
    }

    // =============================================================================================
    // Shaders / pipelines
    // =============================================================================================
    static string LoadSource(string dir, string file)
    {
        var src = File.ReadAllText(Path.Combine(dir, file));
        return Regex.Replace(src, "^#include \"(.+)\"", m => LoadSource(dir, m.Groups[1].Value), RegexOptions.Multiline);
    }

    static byte[] Compile(string dir, string file, string entry, string profile)
    {
        var src = LoadSource(dir, file);
        var args = new[] { "-E", entry, "-T", profile, "-HV", "2021", "-O3" };
        using var result = DxcCompiler.Compile(src, args, null);
        if (result.GetStatus().Failure)
            throw new InvalidOperationException($"Shader compilation failed: {file} ({entry})\n{result.GetErrors()}");
        return result.GetObjectBytecodeArray();
    }

    public void ReloadShaders(string shaderDir)
    {
        WaitIdle();
        CreatePipelines(shaderDir);
        historyValid = false;
    }

    void CreatePipelines(string dir)
    {
        // Compile everything first so a broken shader during hot reload keeps the old pipelines alive.
        var vsMain = Compile(dir, "Raster.hlsl", "VS_Main", "vs_6_6");
        var psMain = Compile(dir, "Raster.hlsl", "PS_Main", "ps_6_6");
        var vsShadow = Compile(dir, "Raster.hlsl", "VS_Shadow", "vs_6_6");
        var psGBuffer = Compile(dir, "Raster.hlsl", "PS_GBuffer", "ps_6_6");
        var psGlass = Compile(dir, "Raster.hlsl", "PS_Glass", "ps_6_6");
        var csBakeClear = Compile(dir, "SdfBake.hlsl", "CS_SdfClear", "cs_6_6");
        var csBakeSeed = Compile(dir, "SdfBake.hlsl", "CS_SdfSeed", "cs_6_6");
        var csBakeJfa = Compile(dir, "SdfBake.hlsl", "CS_SdfJfa", "cs_6_6");
        var csBakeFinalize = Compile(dir, "SdfBake.hlsl", "CS_SdfFinalize", "cs_6_6");
        var vsFs = Compile(dir, "Post.hlsl", "VS_Fullscreen", "vs_6_6");
        var psTone = Compile(dir, "Post.hlsl", "PS_Tonemap", "ps_6_6");
        var csRt = Compile(dir, "RayTrace.hlsl", "CS_RayTrace", "cs_6_6");
        var csPt = Compile(dir, "PathTrace.hlsl", "CS_PathTrace", "cs_6_6");
        var csTemporal = Compile(dir, "Denoise.hlsl", "CS_Temporal", "cs_6_6");
        var csAtrous = Compile(dir, "Denoise.hlsl", "CS_Atrous", "cs_6_6");
        var csSdf = Compile(dir, "Sdf.hlsl", "CS_Sdf", "cs_6_6");

        DisposePipelines();
        rootSig = device.CreateRootSignature(0, vsMain);

        GraphicsPipelineStateDescription Gfx(byte[] vs, byte[] ps, RasterizerDescription rs, Format[] rtvs, Format dsv, DepthStencilDescription ds, int samples = 1) => new()
        {
            RootSignature = rootSig,
            VertexShader = vs,
            PixelShader = ps ?? ReadOnlyMemory<byte>.Empty,
            PrimitiveTopologyType = PrimitiveTopologyType.Triangle,
            RasterizerState = rs,
            BlendState = OpaqueBlend(),
            DepthStencilState = ds,
            SampleMask = uint.MaxValue,
            RenderTargetFormats = rtvs,
            DepthStencilFormat = dsv,
            SampleDescription = new SampleDescription((uint)samples, 0),
        };

        var cullBack = new RasterizerDescription(CullMode.Back, FillMode.Solid);
        var cullBackMirrored = new RasterizerDescription(CullMode.Back, FillMode.Solid) { FrontCounterClockwise = true };
        // shadow casters: render back faces (cull front) + slope bias -> almost no acne, no peter-panning
        var shadowRs = new RasterizerDescription(CullMode.Front, FillMode.Solid) { SlopeScaledDepthBias = 1.0f, DepthBias = 0, DepthBiasClamp = 0.0f };

        psoMain = device.CreateGraphicsPipelineState(Gfx(vsMain, psMain, cullBack, new[] { HdrFormat }, Format.D32_Float, DepthStencilDescription.Default));
        psoMainMsaa = device.CreateGraphicsPipelineState(Gfx(vsMain, psMain, new RasterizerDescription(CullMode.Back, FillMode.Solid) { MultisampleEnable = true }, new[] { HdrFormat }, Format.D32_Float, DepthStencilDescription.Default, MsaaSamples));
        psoGBuffer = device.CreateGraphicsPipelineState(Gfx(vsMain, psGBuffer, cullBack, new[] { GBufPosFormat, GBufNrmFormat }, Format.D32_Float, DepthStencilDescription.Default));
        psoGlass = device.CreateGraphicsPipelineState(Gfx(vsMain, psGlass, cullBack, new[] { HdrFormat }, Format.D32_Float, DepthStencilDescription.Default));
        psoGlassMsaa = device.CreateGraphicsPipelineState(Gfx(vsMain, psGlass, new RasterizerDescription(CullMode.Back, FillMode.Solid) { MultisampleEnable = true }, new[] { HdrFormat }, Format.D32_Float, DepthStencilDescription.Default, MsaaSamples));
        psoMainMirrored = device.CreateGraphicsPipelineState(Gfx(vsMain, psMain, cullBackMirrored, new[] { HdrFormat }, Format.D32_Float, DepthStencilDescription.Default));
        psoShadow = device.CreateGraphicsPipelineState(Gfx(vsShadow, null, shadowRs, Array.Empty<Format>(), Format.D32_Float, DepthStencilDescription.Default));
        psoTonemap = device.CreateGraphicsPipelineState(Gfx(vsFs, psTone, new RasterizerDescription(CullMode.None, FillMode.Solid),
            new[] { Format.R8G8B8A8_UNorm_SRgb }, Format.Unknown, DepthStencilDescription.None));

        ID3D12PipelineState Cs(byte[] cs) => device.CreateComputePipelineState(new ComputePipelineStateDescription { RootSignature = rootSig, ComputeShader = cs });
        psoRayTrace = Cs(csRt);
        psoPathTrace = Cs(csPt);
        psoTemporal = Cs(csTemporal);
        psoAtrous = Cs(csAtrous);
        psoSdf = Cs(csSdf);
        psoBakeClear = Cs(csBakeClear);
        psoBakeSeed = Cs(csBakeSeed);
        psoBakeJfa = Cs(csBakeJfa);
        psoBakeFinalize = Cs(csBakeFinalize);
    }

    // BlendDescription.Opaque only fills RenderTarget[0]; MRT pipelines need every slot valid.
    static BlendDescription OpaqueBlend()
    {
        var b = BlendDescription.Opaque;
        for (int i = 1; i < 8; i++) b.RenderTarget[i] = b.RenderTarget[0];
        return b;
    }

    void DisposePipelines()
    {
        foreach (var p in new IDisposable[] { psoMain, psoMainMsaa, psoMainMirrored, psoShadow, psoTonemap, psoRayTrace, psoPathTrace, psoTemporal, psoAtrous, psoGBuffer, psoSdf,
            psoGlass, psoGlassMsaa, psoBakeClear, psoBakeSeed, psoBakeJfa, psoBakeFinalize, rootSig })
            p?.Dispose();
    }

    // =============================================================================================
    // Frame
    // =============================================================================================
    public void SetHud(byte[] bgra) => pendingHud = bgra;

    /// <summary>Replaces one entry of the GPU material buffer at the start of the next frame.</summary>
    public void SetMaterial(int index, Material material)
    {
        pendingMaterials.Add((index, material));
        historyValid = false;   // the path tracer's history is now wrong
    }

    void UploadMaterials(UploadRing ring)
    {
        if (pendingMaterials.Count == 0) return;
        cl.ResourceBarrierTransition(matBuf, ResourceStates.AllShaderResource, ResourceStates.CopyDest, D3D12.ResourceBarrierAllSubResources, ResourceBarrierFlags.None);
        foreach (var (index, material) in pendingMaterials)
        {
            var a = ring.Alloc((ulong)sizeof(Material), 16);
            *(Material*)a.Cpu = material;
            cl.CopyBufferRegion(matBuf, (ulong)(index * sizeof(Material)), ring.Buffer, a.Offset, (ulong)sizeof(Material));
        }
        cl.ResourceBarrierTransition(matBuf, ResourceStates.CopyDest, ResourceStates.AllShaderResource, D3D12.ResourceBarrierAllSubResources, ResourceBarrierFlags.None);
        pendingMaterials.Clear();
    }
    public void RequestCapture(string path) => pendingCapture = path;
    public static (int W, int H) HudSize => (HudWidth, HudHeight);

    void RootConstants(bool compute, params uint[] v)
    {
        fixed (uint* p = v)
        {
            if (compute) cl.SetComputeRoot32BitConstants(0, (uint)v.Length, p, 0);
            else cl.SetGraphicsRoot32BitConstants(0, (uint)v.Length, p, 0);
        }
    }

    void Dispatch2D() => cl.Dispatch((uint)(Width + 7) / 8, (uint)(Height + 7) / 8, 1);

    public void Render(Camera camera, RenderSettings s, float time)
    {
        int fi = (int)(frameCounter % FrameCount);
        WaitForFence(frameFence[fi]);
        ReadGpuTime(fi);

        var ring = rings[fi];
        ring.Reset();
        allocators[fi].Reset();
        cl.Reset(allocators[fi], null);
        cl.EndQuery(queryHeap, QueryType.Timestamp, (uint)(fi * 2));

        cl.SetDescriptorHeaps(srvHeap);
        cl.SetGraphicsRootSignature(rootSig);
        cl.SetComputeRootSignature(rootSig);
        cl.IASetPrimitiveTopology(PrimitiveTopology.TriangleList);

        // ---- frame constants ----
        float aspect = (float)Width / Height;
        var view = camera.View;
        var proj = Matrix4x4.CreatePerspectiveFieldOfViewLeftHanded(camera.FovY, aspect, 0.05f, 100f);
        var viewProj = view * proj;
        Matrix4x4.Invert(viewProj, out var invViewProj);
        if (lastMode != s.Mode) historyValid = false;
        if (!historyValid) { prevViewProj = viewProj; prevCamPos = camera.Position; }

        var fd = new FrameData
        {
            ViewProj = viewProj,
            InvViewProj = invViewProj,
            PrevViewProj = prevViewProj,
            CamPos = camera.Position,
            PrevCamPos = prevCamPos,
            Time = time,
            FrameIndex = (uint)frameCounter,
            ScreenSize = new Vector2(Width, Height),
            InvScreenSize = new Vector2(1f / Width, 1f / Height),
            NumLights = (uint)scene.Lights.Length,
            MaxBounces = (uint)(s.Mode is RenderMode.PathTrace or RenderMode.Hybrid ? s.PtBounces : s.RtBounces),
            Spp = (uint)s.Spp,
            Ambient = s.Mode is RenderMode.PathTrace or RenderMode.Hybrid ? Vector3.Zero : new Vector3(0.025f, 0.025f, 0.03f),
            Exposure = s.Exposure,
            Light0 = scene.Lights[0], Light1 = scene.Lights[1], Light2 = scene.Lights[2],
            VbIndex = SlotVB, IbIndex = SlotIB, MatIndex = SlotMat, TlasIndex = SlotTlas,
            ShadowIndex = SlotShadow, ProbeIndex = SlotProbe, StaticIndexCount = (uint)scene.StaticIndexCount,
            ProbePos = scene.ProbePositions[0], ShadowNear = ShadowNear, ShadowFar = ShadowFar,
            MeshFirstIndex = (uint)scene.MeshFirstIndex,
            GlassFirstIndex = (uint)scene.GlassFirstIndex,
            MeshMaterial = scene.HasMesh ? (uint)scene.MeshMaterialIndex : 0xFFFFFFFFu,
            MeshSdfIndex = SlotMeshSdf, MeshSdfMin = meshSdfMin, MeshSdfMax = meshSdfMax,
        };
        ulong frameCb = ring.Push(fd);
        cl.SetGraphicsRootConstantBufferView(1, frameCb);
        cl.SetComputeRootConstantBufferView(1, frameCb);
        ulong mainPass = ring.Push(new PassData { ViewProj = viewProj, CamPos = camera.Position, ClipPlane = new Vector4(0, 0, 0, 1) });
        cl.SetComputeRootConstantBufferView(2, mainPass);

        UploadHud(ring);
        UploadMaterials(ring);

        switch (s.Mode)
        {
            case RenderMode.Raster:
                RenderRaster(ring, camera, view, proj, mainPass, s.AntiAliasing);
                break;
            case RenderMode.RayTrace:
                BuildTlas(ring);
                Transition(hdr, ResourceStates.UnorderedAccess);
                cl.SetPipelineState(psoRayTrace);
                RootConstants(true, SlotHdrUav, s.AntiAliasing ? 4u : 1u);
                Dispatch2D();
                break;
            case RenderMode.PathTrace:
                BuildTlas(ring);
                RenderPathTrace(s, fi, false);
                break;
            case RenderMode.Sdf:
                Transition(hdr, ResourceStates.UnorderedAccess);
                cl.SetPipelineState(psoSdf);
                RootConstants(true, SlotHdrUav, SlotSdf, (uint)scene.SdfPrims.Count, s.SdfAntiAliasing ? 4u : 1u, (uint)scene.SdfPlaneCount);
                Dispatch2D();
                break;
            case RenderMode.Hybrid:
                BuildTlas(ring);
                RenderGBuffer(mainPass);
                RenderPathTrace(s, fi, true);
                break;
        }

        // ---- tonemap + HUD into the back buffer ----
        uint bbIndex = swapChain.CurrentBackBufferIndex;
        var bb = backBuffers[bbIndex];
        Transition(hdr, ResourceStates.AllShaderResource);
        Transition(hud, ResourceStates.AllShaderResource);
        Transition(bb, ResourceStates.RenderTarget);
        cl.OMSetRenderTargets(backBufferRtvs[bbIndex], null);
        SetViewport(Width, Height);
        cl.SetPipelineState(psoTonemap);
        RootConstants(false, SlotHdrSrv, SlotHud, HudWidth, HudHeight);
        cl.DrawInstanced(3, 1, 0, 0);

        string capture = pendingCapture;
        ID3D12Resource captureBuf = null;
        uint captureRowPitch = 0;
        if (capture != null)
        {
            pendingCapture = null;
            captureRowPitch = (uint)((Width * 4 + 255) & ~255);
            captureBuf = CreateBuffer("Capture", (ulong)captureRowPitch * (ulong)Height, HeapType.Readback, ResourceFlags.None, ResourceStates.CopyDest);
            Transition(bb, ResourceStates.CopySource);
            var footprint = new PlacedSubresourceFootPrint
            {
                Offset = 0,
                Footprint = new SubresourceFootPrint(Format.R8G8B8A8_UNorm, (uint)Width, (uint)Height, 1, captureRowPitch),
            };
            cl.CopyTextureRegion(new TextureCopyLocation(captureBuf, footprint), 0, 0, 0, new TextureCopyLocation(bb.Resource, 0), null);
        }

        Transition(bb, ResourceStates.Present);
        cl.EndQuery(queryHeap, QueryType.Timestamp, (uint)(fi * 2 + 1));
        cl.ResolveQueryData(queryHeap, QueryType.Timestamp, (uint)(fi * 2), 2, queryReadback, (ulong)(fi * 16));
        cl.Close();
        queue.ExecuteCommandList(cl);

        var hr = swapChain.Present(s.VSync ? 1u : 0u, !s.VSync && tearingSupported ? PresentFlags.AllowTearing : PresentFlags.None);
        frameFence[fi] = ++fenceValue;
        queue.Signal(fence, fenceValue);
        DrainDebugMessages();
        if (hr.Failure)
            throw new InvalidOperationException($"Present failed: {hr} / device removed reason: {device.DeviceRemovedReason}");

        if (captureBuf != null)
        {
            WaitIdle();
            SaveCapture(captureBuf, captureRowPitch, capture);
            captureBuf.Dispose();
        }

        prevViewProj = viewProj;
        prevCamPos = camera.Position;
        historyValid = true;
        lastMode = s.Mode;
        frameCounter++;
    }

    void SetViewport(int w, int h)
    {
        cl.RSSetViewport(new Viewport(0, 0, w, h, 0, 1));
        cl.RSSetScissorRect(w, h);
    }

    void UploadHud(UploadRing ring)
    {
        var data = pendingHud;
        if (data == null) return;
        pendingHud = null;
        const uint rowPitch = HudWidth * 4;   // 2080 -> must be 256 aligned
        uint pitch = (rowPitch + 255) & ~255u;
        var a = ring.Alloc((ulong)pitch * HudHeight, 512);
        fixed (byte* src = data)
            for (int y = 0; y < HudHeight; y++)
                Buffer.MemoryCopy(src + y * rowPitch, (byte*)a.Cpu + y * pitch, pitch, rowPitch);
        Transition(hud, ResourceStates.CopyDest);
        var footprint = new PlacedSubresourceFootPrint
        {
            Offset = a.Offset,
            Footprint = new SubresourceFootPrint(Format.B8G8R8A8_UNorm, HudWidth, HudHeight, 1, pitch),
        };
        cl.CopyTextureRegion(new TextureCopyLocation(hud.Resource, 0), 0, 0, 0, new TextureCopyLocation(ring.Buffer, footprint), null);
    }

    // ---------------------------------------------------------------------------------------------
    // Stage 1: rasterization
    // ---------------------------------------------------------------------------------------------
    static readonly Vector3[] CubeDirs = { Vector3.UnitX, -Vector3.UnitX, Vector3.UnitY, -Vector3.UnitY, Vector3.UnitZ, -Vector3.UnitZ };
    static readonly Vector3[] CubeUps = { Vector3.UnitY, Vector3.UnitY, -Vector3.UnitZ, Vector3.UnitZ, Vector3.UnitY, Vector3.UnitY };

    static Matrix4x4 CubeFaceViewProj(Vector3 pos, int face, float near, float far) =>
        Matrix4x4.CreateLookAtLeftHanded(pos, pos + CubeDirs[face], CubeUps[face]) *
        Matrix4x4.CreatePerspectiveFieldOfViewLeftHanded(MathF.PI / 2, 1, near, far);

    void DrawScene(Func<SceneObject, bool> filter, bool lightBulbs)
    {
        foreach (var o in scene.Objects)
        {
            if (!filter(o)) continue;
            RootConstants(false, (uint)o.FirstIndex, 0xFFFFFFFF);
            cl.DrawInstanced((uint)o.IndexCount, 1, 0, 0);
        }
        if (lightBulbs)
        {
            for (int i = 0; i < scene.Lights.Length; i++)
            {
                RootConstants(false, (uint)scene.SphereFirstIndex, (uint)i);
                cl.DrawInstanced((uint)scene.SphereIndexCount, 1, 0, 0);
            }
        }
    }

    void RenderRaster(UploadRing ring, Camera camera, Matrix4x4 view, Matrix4x4 proj, ulong mainPass, bool msaa)
    {
        var noClip = new Vector4(0, 0, 0, 1);
        var glassObjects = scene.Objects.Where(scene.IsGlass).ToList();

        // 1) omnidirectional shadow maps: 6 faces per light
        Transition(shadowMap, ResourceStates.DepthWrite);
        cl.SetPipelineState(psoShadow);
        SetViewport(ShadowSize, ShadowSize);
        for (int l = 0; l < scene.Lights.Length; l++)
        {
            var lp = scene.Lights[l].Pos;
            for (int f = 0; f < 6; f++)
            {
                var dsv = shadowDsv[l * 6 + f];
                cl.OMSetRenderTargets(0, (CpuDescriptorHandle*)null, false, dsv);
                cl.ClearDepthStencilView(dsv, ClearFlags.Depth, 1, 0);
                cl.SetGraphicsRootConstantBufferView(2, ring.Push(new PassData { ViewProj = CubeFaceViewProj(lp, f, ShadowNear, ShadowFar), CamPos = lp, ClipPlane = noClip }));
                // mirrors hug the walls (nothing behind them to shadow); glass lets the light through
                DrawScene(o => o.MirrorIndex < 0 && !scene.IsGlass(o), false);
            }
        }
        Transition(shadowMap, ResourceStates.AllShaderResource);

        // 2) dynamic reflection probes from the chrome sphere / mesh centers (each skips its own object)
        cl.SetPipelineState(psoMain);
        SetViewport(ProbeSize, ProbeSize);
        for (int p = 0; p < (scene.HasMesh ? 2 : 1); p++)
        {
            var pos = scene.ProbePositions[p];
            Transition(probes[p], ResourceStates.RenderTarget);
            for (int f = 0; f < 6; f++)
            {
                cl.OMSetRenderTargets(probeRtv[p, f], probeDepth.Dsv);
                cl.ClearRenderTargetView(probeRtv[p, f], new Color4(0, 0, 0, 1));
                cl.ClearDepthStencilView(probeDepth.Dsv, ClearFlags.Depth, 1, 0);
                cl.SetGraphicsRootConstantBufferView(2, ring.Push(new PassData
                {
                    ViewProj = CubeFaceViewProj(pos, f, 0.05f, 50f),
                    CamPos = pos,
                    Flags = PassData.NoMirror | PassData.NoProbe,
                    ClipPlane = noClip,
                }));
                int self = p;
                DrawScene(o => o.ProbeCenter != self, true);
            }
            Transition(probes[p], ResourceStates.AllShaderResource);
        }

        // 3) planar reflections: re-render the scene through each mirror
        SetViewport(Width, Height);
        cl.SetPipelineState(psoMainMirrored);
        foreach (var m in scene.Mirrors)
        {
            int mi = scene.Objects[m.ObjectIndex].MirrorIndex;
            var target = refl[mi];
            var reflect = Matrix4x4.CreateReflection(m.Plane);
            Transition(target, ResourceStates.RenderTarget);
            cl.OMSetRenderTargets(target.Rtv, depth.Dsv);
            cl.ClearRenderTargetView(target.Rtv, new Color4(0, 0, 0, 1));
            cl.ClearDepthStencilView(depth.Dsv, ClearFlags.Depth, 1, 0);
            cl.SetGraphicsRootConstantBufferView(2, ring.Push(new PassData
            {
                ViewProj = reflect * view * proj,
                CamPos = Vector3.Transform(camera.Position, reflect),
                Flags = PassData.NoMirror,
                ClipPlane = new Vector4(m.Plane.Normal, m.Plane.D),
            }));
            int skip = m.ObjectIndex;
            DrawScene(o => scene.Objects.IndexOf(o) != skip, true);
            Transition(target, ResourceStates.AllShaderResource);
        }

        // 4) main view (optionally 4x MSAA, resolved into the HDR target)
        var color = msaa ? msaaColor : hdr;
        var dsvMain = msaa ? msaaDepth.Dsv : depth.Dsv;
        cl.SetPipelineState(msaa ? psoMainMsaa : psoMain);
        Transition(color, ResourceStates.RenderTarget);
        cl.OMSetRenderTargets(color.Rtv, dsvMain);
        cl.ClearRenderTargetView(color.Rtv, new Color4(0, 0, 0, 1));
        cl.ClearDepthStencilView(dsvMain, ClearFlags.Depth, 1, 0);
        cl.SetGraphicsRootConstantBufferView(2, mainPass);
        DrawScene(o => !scene.IsGlass(o), true);

        // 5) glass: needs the finished opaque image to refract, so copy it and draw the mesh on top
        if (glassObjects.Count > 0)
        {
            if (msaa) ResolveMsaa();
            Transition(hdr, ResourceStates.CopySource);
            Transition(sceneColor, ResourceStates.CopyDest);
            cl.CopyResource(sceneColor.Resource, hdr.Resource);
            Transition(sceneColor, ResourceStates.AllShaderResource);
            Transition(color, ResourceStates.RenderTarget);
            cl.OMSetRenderTargets(color.Rtv, dsvMain);
            cl.SetPipelineState(msaa ? psoGlassMsaa : psoGlass);
            foreach (var o in glassObjects)
            {
                RootConstants(false, (uint)o.FirstIndex, 0xFFFFFFFF, SlotSceneColor);
                cl.DrawInstanced((uint)o.IndexCount, 1, 0, 0);
            }
        }
        if (msaa) ResolveMsaa();
    }

    void ResolveMsaa()
    {
        Transition(msaaColor, ResourceStates.ResolveSource);
        Transition(hdr, ResourceStates.ResolveDest);
        cl.ResolveSubresource(hdr.Resource, 0, msaaColor.Resource, 0, HdrFormat);
    }

    // ---------------------------------------------------------------------------------------------
    // Stage 2/3: acceleration structure update
    // ---------------------------------------------------------------------------------------------
    void BuildTlas(UploadRing ring)
    {
        int count = TlasInstanceCount;
        var a = ring.Alloc((ulong)(count * sizeof(InstanceDesc)), 256);
        var inst = (InstanceDesc*)a.Cpu;
        inst[0] = InstanceDesc.Create(Matrix4x4.Identity, 0, 0x01, blasStatic.GPUVirtualAddress);
        for (int i = 0; i < scene.Lights.Length; i++)
        {
            var L = scene.Lights[i];
            var m = Matrix4x4.CreateScale(L.Radius) * Matrix4x4.CreateTranslation(L.Pos);
            inst[1 + i] = InstanceDesc.Create(m, (uint)(1 + i), 0x02, blasSphere.GPUVirtualAddress);
        }
        int next = 1 + scene.Lights.Length;
        if (scene.HasMesh)
            inst[next++] = InstanceDesc.Create(Matrix4x4.Identity, 100, 0x04, blasMesh.GPUVirtualAddress);    // INSTANCE_MESH, MASK_MESH
        if (blasGlass != null)
            inst[next++] = InstanceDesc.Create(Matrix4x4.Identity, 101, 0x08, blasGlass.GPUVirtualAddress);   // INSTANCE_GLASS, MASK_GLASS
        tlasInputs.InstanceDescriptions = a.Gpu;
        cl.BuildRaytracingAccelerationStructure(new BuildRaytracingAccelerationStructureDescription
        {
            DestinationAccelerationStructureData = tlas.GPUVirtualAddress,
            Inputs = tlasInputs,
            ScratchAccelerationStructureData = tlasScratch.GPUVirtualAddress,
        });
        cl.ResourceBarrierUnorderedAccessView(tlas);
    }

    // ---------------------------------------------------------------------------------------------
    // Stage 3: path tracing + denoiser
    // ---------------------------------------------------------------------------------------------
    // ---------------------------------------------------------------------------------------------
    // Stage 4 (hybrid): rasterize primary visibility into a G-buffer, path trace the rest from it
    // ---------------------------------------------------------------------------------------------
    void RenderGBuffer(ulong mainPass)
    {
        Transition(gbufPos, ResourceStates.RenderTarget);
        Transition(gbufNrm, ResourceStates.RenderTarget);
        var rtvs = stackalloc CpuDescriptorHandle[2] { gbufPos.Rtv, gbufNrm.Rtv };
        cl.OMSetRenderTargets(2, rtvs, false, depth.Dsv);
        cl.ClearRenderTargetView(gbufPos.Rtv, new Color4(0, 0, 0, 0));
        cl.ClearRenderTargetView(gbufNrm.Rtv, new Color4(0, 0, 0, 0));
        cl.ClearDepthStencilView(depth.Dsv, ClearFlags.Depth, 1, 0);
        SetViewport(Width, Height);
        cl.SetPipelineState(psoGBuffer);
        cl.SetGraphicsRootConstantBufferView(2, mainPass);
        DrawScene(_ => true, true);
        Transition(gbufPos, ResourceStates.AllShaderResource);
        Transition(gbufNrm, ResourceStates.AllShaderResource);
    }

    void RenderPathTrace(RenderSettings s, int fi, bool hybrid)
    {
        int cur = (int)(frameCounter & 1), prev = 1 - cur;

        Transition(ptNoisy, ResourceStates.UnorderedAccess);
        Transition(ptGuide[cur], ResourceStates.UnorderedAccess);
        Transition(ptAlbedo, ResourceStates.UnorderedAccess);
        cl.SetPipelineState(psoPathTrace);
        RootConstants(true, (uint)ptNoisy.Uav, (uint)ptGuide[cur].Uav, (uint)ptAlbedo.Uav,
            hybrid ? (uint)gbufPos.Srv : 0xFFFFFFFFu, hybrid ? (uint)gbufNrm.Srv : 0xFFFFFFFFu);
        Dispatch2D();

        // temporal accumulation
        Transition(ptNoisy, ResourceStates.AllShaderResource);
        Transition(ptGuide[cur], ResourceStates.AllShaderResource);
        Transition(ptAlbedo, ResourceStates.AllShaderResource);
        Transition(ptGuide[prev], ResourceStates.AllShaderResource);
        Transition(ptHist[prev], ResourceStates.AllShaderResource);
        Transition(ptMom[prev], ResourceStates.AllShaderResource);
        Transition(ptHist[cur], ResourceStates.UnorderedAccess);
        Transition(ptMom[cur], ResourceStates.UnorderedAccess);
        uint maxHistory = s.LightsAnimating ? 12u : 4096u;
        uint reset = historyValid ? 0u : 1u;
        cl.SetPipelineState(psoTemporal);
        RootConstants(true, (uint)ptNoisy.Srv, (uint)ptGuide[cur].Srv, (uint)ptGuide[prev].Srv, (uint)ptHist[prev].Srv,
            (uint)ptHist[cur].Uav, (uint)ptMom[prev].Srv, (uint)ptMom[cur].Uav, reset | (maxHistory << 1));
        Dispatch2D();
        Transition(ptHist[cur], ResourceStates.AllShaderResource);
        Transition(ptMom[cur], ResourceStates.AllShaderResource);

        // spatial a-trous wavelet filter (or plain re-modulation when the denoiser is off)
        cl.SetPipelineState(psoAtrous);
        int iterations = s.Denoise ? 4 : 1;
        GpuTexture src = ptHist[cur];
        for (int i = 0; i < iterations; i++)
        {
            bool last = i == iterations - 1;
            GpuTexture dst = last ? hdr : ptFilt[i & 1];
            Transition(src, ResourceStates.AllShaderResource);
            Transition(dst, ResourceStates.UnorderedAccess);
            uint step = s.Denoise ? (1u << i) : 0u;
            RootConstants(true, (uint)src.Srv, (uint)dst.Uav, (uint)ptGuide[cur].Srv, (uint)ptMom[cur].Srv, (uint)ptAlbedo.Srv, step, last ? 1u : 0u);
            Dispatch2D();
            src = dst;
        }
    }

    // =============================================================================================
    // Sync / misc
    // =============================================================================================
    void WaitForFence(ulong value)
    {
        if (fence.CompletedValue >= value) return;
        fence.SetEventOnCompletion(value, fenceEvent).CheckError();
        fenceEvent.WaitOne();
    }

    public void WaitIdle()
    {
        queue.Signal(fence, ++fenceValue);
        WaitForFence(fenceValue);
    }

    void ReadGpuTime(int fi)
    {
        if (frameCounter < FrameCount) return;
        void* p;
        var range = new Vortice.Direct3D12.Range((nuint)(fi * 16), (nuint)(fi * 16 + 16));
        if (queryReadback.Map(0, range, &p).Failure) return;
        var ts = (ulong*)((byte*)p + fi * 16);
        if (ts[1] > ts[0])
            GpuTimeMs = GpuTimeMs * 0.9 + (ts[1] - ts[0]) * timestampPeriodMs * 0.1;
        queryReadback.Unmap(0, new Vortice.Direct3D12.Range(0, 0));
    }

    void SaveCapture(ID3D12Resource buf, uint rowPitch, string path)
    {
        void* p;
        buf.Map(0, null, &p).CheckError();
        using var bmp = new System.Drawing.Bitmap(Width, Height, System.Drawing.Imaging.PixelFormat.Format32bppRgb);
        var data = bmp.LockBits(new System.Drawing.Rectangle(0, 0, Width, Height), System.Drawing.Imaging.ImageLockMode.WriteOnly, bmp.PixelFormat);
        for (int y = 0; y < Height; y++)
        {
            var srcRow = (byte*)p + y * rowPitch;
            var dstRow = (byte*)data.Scan0 + y * data.Stride;
            for (int x = 0; x < Width; x++)
            {
                dstRow[x * 4 + 0] = srcRow[x * 4 + 2];
                dstRow[x * 4 + 1] = srcRow[x * 4 + 1];
                dstRow[x * 4 + 2] = srcRow[x * 4 + 0];
                dstRow[x * 4 + 3] = 255;
            }
        }
        bmp.UnlockBits(data);
        buf.Unmap(0);
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path)));
        bmp.Save(path, System.Drawing.Imaging.ImageFormat.Png);
    }

    void DrainDebugMessages()
    {
        if (infoQueue == null) return;
        ulong n = infoQueue.NumStoredMessages;
        if (n == 0) return;
        using var w = File.AppendText("d3d12-debug.log");
        for (ulong i = 0; i < n; i++)
        {
            var m = infoQueue.GetMessage(i);
            w.WriteLine($"[{m.Severity}] {m.Id}: {m.Description}");
        }
        infoQueue.ClearStoredMessages();
    }

    bool disposed;

    public void Dispose()
    {
        // WinForms disposes the form on close and `using` in Main disposes it again.
        if (disposed) return;
        disposed = true;
        WaitIdle();
        ReleaseSizeDependent();
        DisposePipelines();
        foreach (var d in new IDisposable[] { shadowMap, probes[0], probes[1], probeDepth, hud, meshSdf, vb, ib, matBuf, sdfBuf, blasStatic, blasGlass, blasSphere, blasMesh, tlas, tlasScratch, queryHeap, queryReadback })
            d?.Dispose();
        foreach (var r in rings) r.Dispose();
        foreach (var a in allocators) a.Dispose();
        cl.Dispose();
        fence.Dispose();
        srvHeap.Dispose(); rtvHeap.Dispose(); dsvHeap.Dispose();
        swapChain.Dispose();
        queue.Dispose();
        infoQueue?.Dispose();
        device.Dispose();
        factory.Dispose();
    }
}
