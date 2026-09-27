using System.Numerics;
using System.Text.RegularExpressions;
using Vortice.Direct3D;
using Vortice.Direct3D12;
using Vortice.Direct3D12.Debug;
using Vortice.Dxc;
using Vortice.DXGI;
using Vortice.Mathematics;
using ShaderResourceViewDimension = Vortice.Direct3D12.ShaderResourceViewDimension;

namespace RtxSlop;

/// <summary>
/// Device, swap chain, pipelines and the per-frame passes of all five modes. Per-scene GPU data
/// (buffers, acceleration structures, textures, SDF grids) lives in <see cref="SceneGpu"/> (RendererScene.cs).
/// </summary>
public sealed unsafe partial class Renderer : IDisposable
{
    const int FrameCount = 2;          // frames in flight
    const int BackBufferCount = 3;
    const int MaxLights = 3;
    const int ShadowSize = 1024;
    const int ProbeSize = 256;
    const float ShadowNear = 0.05f;
    const int HudWidth = 560, HudHeight = 252;
    const Format HdrFormat = Format.R16G16B16A16_Float;
    const int MsaaSamples = 4;

    // Fixed bindless descriptor slots (per-scene resources are allocated from FirstDynamicSlot upwards)
    const int SrvHeapSize = 4096;
    const int FirstDynamicSlot = 96;
    const int SlotShadow = 4, SlotProbe = 5, SlotHud = 6, SlotProbe1 = 7;
    const int SlotHdrSrv = 10, SlotHdrUav = 11;
    public const int SlotRefl0 = 12;
    const int SlotPtBase = 20;   // PT textures: srv = base + 2k, uav = base + 2k + 1 (slots 20..39)
    const int SlotGBufPos = 40, SlotGBufNrm = 41, SlotSceneColor = 43;
    const int SlotLights = 64;   // + frame in flight: the light buffer (a StructuredBuffer view into the upload ring)
    const int SlotRestir = 68;   // ReSTIR sets k = 0..2 at 68 + 8k: SRVs +0..3 (reservoir A/B, surface A/B), UAVs +4..7
    public static uint ProbeSlot(int i) => (uint)(i == 0 ? SlotProbe : SlotProbe1);
    const Format GBufPosFormat = Format.R32G32B32A32_Float, GBufNrmFormat = Format.R32G32B32A32_Float;

    public string AdapterName { get; }
    public int Width { get; private set; }
    public int Height { get; private set; }
    public double GpuTimeMs { get; private set; }

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
        psoGlass, psoGlassMsaa, psoSky, psoSkyMsaa, psoBakeClear, psoBakeSeed, psoBakeJfa, psoBakeFinalize, psoSkyVis,
        psoRestirInitial, psoRestirSpatial;

    readonly List<(int Index, Material Material)> pendingMaterials = new();

    // fixed size render targets
    GpuTexture shadowMap, probeDepth, hud;
    GpuTexture[] probes = new GpuTexture[2];   // 0 = chrome sphere (room), 1 = model
    CpuDescriptorHandle[] shadowDsv = new CpuDescriptorHandle[6 * MaxLights];
    CpuDescriptorHandle[,] probeRtv = new CpuDescriptorHandle[2, 6];

    // size dependent
    GpuTexture[] backBuffers = new GpuTexture[BackBufferCount];
    GpuTexture depth, hdr, msaaColor, msaaDepth, sceneColor;
    GpuTexture[] refl = new GpuTexture[2];
    GpuTexture ptNoisy, ptAlbedo, gbufPos, gbufNrm;
    GpuTexture[] ptGuide = new GpuTexture[2], ptHist = new GpuTexture[2], ptMom = new GpuTexture[2], ptFilt = new GpuTexture[2];
    CpuDescriptorHandle[] backBufferRtvs = new CpuDescriptorHandle[BackBufferCount];
    GpuTexture[,] restir = new GpuTexture[3, 4];   // [set, reservoir A / reservoir B / surface A / surface B]; sets 0/1 ping-pong, 2 = temporary

    // timing
    ID3D12QueryHeap queryHeap;
    ID3D12Resource queryReadback;
    double timestampPeriodMs;

    // per-frame state
    Matrix4x4 prevViewProj;
    Vector3 prevCamPos;
    bool historyValid;
    int activeLights, lastActiveLights = -1;
    ulong restirFrame = ulong.MaxValue;   // last frame that ran the ReSTIR passes (their history)
    RenderMode lastMode = (RenderMode)(-1);
    byte[] pendingHud;
    string pendingCapture;

    public Renderer(IntPtr hwnd, int width, int height, Scene initialScene, string shaderDir, bool debug)
    {
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

        srvHeap = device.CreateDescriptorHeap(new DescriptorHeapDescription(DescriptorHeapType.ConstantBufferViewShaderResourceViewUnorderedAccessView, SrvHeapSize, DescriptorHeapFlags.ShaderVisible));
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
        CreateFixedTargets();
        CreateSizeDependent();
        SetScene(initialScene);
    }

    // =============================================================================================
    // Descriptors
    // =============================================================================================
    CpuDescriptorHandle SrvCpu(int slot) => srvHeap.GetCPUDescriptorHandleForHeapStart() + (int)(slot * srvInc);
    CpuDescriptorHandle AllocRtv() => rtvHeap.GetCPUDescriptorHandleForHeapStart() + (int)(nextRtv++ * rtvInc);
    CpuDescriptorHandle AllocDsv() => dsvHeap.GetCPUDescriptorHandleForHeapStart() + (int)(nextDsv++ * dsvInc);

    void CreateStructuredSrv(ID3D12Resource res, int slot, int count, int stride, ulong first = 0)
    {
        device.CreateShaderResourceView(res, new ShaderResourceViewDescription
        {
            Format = Format.Unknown,
            ViewDimension = ShaderResourceViewDimension.Buffer,
            Shader4ComponentMapping = ShaderComponentMapping.Default,
            Buffer = new BufferShaderResourceView { FirstElement = first, NumElements = (uint)count, StructureByteStride = (uint)stride },
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

    void CreateFixedTargets()
    {
        // omnidirectional shadow maps: 3 lights x 6 faces in one cube array
        shadowMap = CreateTexture("Shadow cube array", Format.R32_Typeless, ShadowSize, ShadowSize, 6 * MaxLights,
            ResourceFlags.AllowDepthStencil, ResourceStates.AllShaderResource, new ClearValue(Format.D32_Float, 1.0f, 0));
        for (int i = 0; i < 6 * MaxLights; i++)
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
            TextureCubeArray = new TextureCubeArrayShaderResourceView { MipLevels = 1, First2DArrayFace = 0, NumCubes = MaxLights },
        }, SrvCpu(SlotShadow));

        // dynamic reflection probes (cube maps rendered from the chrome sphere's / the model's center)
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
        nextDsv = 6 * MaxLights + 1;

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
        gbufNrm = CreateTexture("GBuffer normal+triangle", GBufNrmFormat, Width, Height, 1, ResourceFlags.AllowRenderTarget, ResourceStates.AllShaderResource, new ClearValue(GBufNrmFormat, zero));
        gbufNrm.Rtv = AllocRtv();
        device.CreateRenderTargetView(gbufNrm.Resource, null, gbufNrm.Rtv);
        CreateTexSrv(gbufNrm, SlotGBufNrm, GBufNrmFormat);

        // ReSTIR DI reservoirs + the surfaces they belong to (the temporary set needs no surfaces)
        for (int set = 0; set < 3; set++)
            for (int j = 0; j < (set == 2 ? 2 : 4); j++)
            {
                var fmt = j == 1 ? Format.R32G32_Float : Format.R32G32B32A32_Float;
                var t = CreateTexture($"ReSTIR {set}.{j}", fmt, Width, Height, 1, ResourceFlags.AllowUnorderedAccess, ResourceStates.AllShaderResource);
                CreateTexSrv(t, SlotRestir + 8 * set + j, fmt);
                CreateTexUav(t, SlotRestir + 8 * set + 4 + j, fmt);
                restir[set, j] = t;
            }

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
        foreach (var t in restir) t?.Dispose();
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
        var jobs = new (string File, string Entry, string Profile)[]
        {
            ("Raster.hlsl", "VS_Main", "vs_6_6"), ("Raster.hlsl", "PS_Main", "ps_6_6"), ("Raster.hlsl", "PS_ShadowClip", "ps_6_6"),
            ("Raster.hlsl", "PS_GBuffer", "ps_6_6"), ("Raster.hlsl", "PS_Glass", "ps_6_6"), ("Raster.hlsl", "VS_Sky", "vs_6_6"),
            ("Raster.hlsl", "PS_Sky", "ps_6_6"), ("Post.hlsl", "VS_Fullscreen", "vs_6_6"), ("Post.hlsl", "PS_Tonemap", "ps_6_6"),
            ("RayTrace.hlsl", "CS_RayTrace", "cs_6_6"), ("PathTrace.hlsl", "CS_PathTrace", "cs_6_6"),
            ("Denoise.hlsl", "CS_Temporal", "cs_6_6"), ("Denoise.hlsl", "CS_Atrous", "cs_6_6"), ("Sdf.hlsl", "CS_Sdf", "cs_6_6"),
            ("SdfBake.hlsl", "CS_SdfClear", "cs_6_6"), ("SdfBake.hlsl", "CS_SdfSeed", "cs_6_6"), ("SdfBake.hlsl", "CS_SdfJfa", "cs_6_6"),
            ("SdfBake.hlsl", "CS_SdfFinalize", "cs_6_6"), ("SceneBake.hlsl", "CS_SkyVisibility", "cs_6_6"),
            ("Restir.hlsl", "CS_RestirInitial", "cs_6_6"), ("Restir.hlsl", "CS_RestirSpatial", "cs_6_6"),
        };
        var code = new byte[jobs.Length][];
        Parallel.For(0, jobs.Length, i => code[i] = Compile(dir, jobs[i].File, jobs[i].Entry, jobs[i].Profile));
        byte[] Code(string entry) => code[Array.FindIndex(jobs, j => j.Entry == entry)];

        DisposePipelines();
        var vsMain = Code("VS_Main");
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

        // Two-sided everywhere: foliage and much of the architecture are single-sided sheets.
        var noCull = new RasterizerDescription(CullMode.None, FillMode.Solid);
        var noCullMsaa = new RasterizerDescription(CullMode.None, FillMode.Solid) { MultisampleEnable = true };
        var shadowRs = new RasterizerDescription(CullMode.None, FillMode.Solid) { SlopeScaledDepthBias = 1.5f, DepthBias = 0, DepthBiasClamp = 0.0f };
        var skyDepth = new DepthStencilDescription(true, DepthWriteMask.Zero, ComparisonFunction.LessEqual);
        var hdrRt = new[] { HdrFormat };

        psoMain = device.CreateGraphicsPipelineState(Gfx(vsMain, Code("PS_Main"), noCull, hdrRt, Format.D32_Float, DepthStencilDescription.Default));
        psoMainMsaa = device.CreateGraphicsPipelineState(Gfx(vsMain, Code("PS_Main"), noCullMsaa, hdrRt, Format.D32_Float, DepthStencilDescription.Default, MsaaSamples));
        psoMainMirrored = device.CreateGraphicsPipelineState(Gfx(vsMain, Code("PS_Main"), noCull, hdrRt, Format.D32_Float, DepthStencilDescription.Default));
        psoGBuffer = device.CreateGraphicsPipelineState(Gfx(vsMain, Code("PS_GBuffer"), noCull, new[] { GBufPosFormat, GBufNrmFormat }, Format.D32_Float, DepthStencilDescription.Default));
        psoGlass = device.CreateGraphicsPipelineState(Gfx(vsMain, Code("PS_Glass"), noCull, hdrRt, Format.D32_Float, DepthStencilDescription.Default));
        psoGlassMsaa = device.CreateGraphicsPipelineState(Gfx(vsMain, Code("PS_Glass"), noCullMsaa, hdrRt, Format.D32_Float, DepthStencilDescription.Default, MsaaSamples));
        psoShadow = device.CreateGraphicsPipelineState(Gfx(vsMain, Code("PS_ShadowClip"), shadowRs, Array.Empty<Format>(), Format.D32_Float, DepthStencilDescription.Default));
        psoSky = device.CreateGraphicsPipelineState(Gfx(Code("VS_Sky"), Code("PS_Sky"), noCull, hdrRt, Format.D32_Float, skyDepth));
        psoSkyMsaa = device.CreateGraphicsPipelineState(Gfx(Code("VS_Sky"), Code("PS_Sky"), noCullMsaa, hdrRt, Format.D32_Float, skyDepth, MsaaSamples));
        psoTonemap = device.CreateGraphicsPipelineState(Gfx(Code("VS_Fullscreen"), Code("PS_Tonemap"), noCull,
            new[] { Format.R8G8B8A8_UNorm_SRgb }, Format.Unknown, DepthStencilDescription.None));

        ID3D12PipelineState Cs(string entry) => device.CreateComputePipelineState(new ComputePipelineStateDescription { RootSignature = rootSig, ComputeShader = Code(entry) });
        psoRayTrace = Cs("CS_RayTrace");
        psoPathTrace = Cs("CS_PathTrace");
        psoTemporal = Cs("CS_Temporal");
        psoAtrous = Cs("CS_Atrous");
        psoSdf = Cs("CS_Sdf");
        psoBakeClear = Cs("CS_SdfClear");
        psoBakeSeed = Cs("CS_SdfSeed");
        psoBakeJfa = Cs("CS_SdfJfa");
        psoBakeFinalize = Cs("CS_SdfFinalize");
        psoSkyVis = Cs("CS_SkyVisibility");
        psoRestirInitial = Cs("CS_RestirInitial");
        psoRestirSpatial = Cs("CS_RestirSpatial");
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
            psoGlass, psoGlassMsaa, psoSky, psoSkyMsaa, psoBakeClear, psoBakeSeed, psoBakeJfa, psoBakeFinalize, psoSkyVis, psoRestirInitial, psoRestirSpatial, rootSig })
            p?.Dispose();
    }

    // =============================================================================================
    // Frame
    // =============================================================================================
    public void SetHud(byte[] bgra) => pendingHud = bgra;
    public void RequestCapture(string path) => pendingCapture = path;
    public static (int W, int H) HudSize => (HudWidth, HudHeight);

    /// <summary>Replaces one entry of the current scene's GPU material buffer at the start of the next frame.</summary>
    public void SetMaterial(int index, Material material)
    {
        pendingMaterials.Add((index, gpu.Remap(material)));
        historyValid = false;   // the path tracer's history is now wrong
    }

    void UploadMaterials(UploadRing ring)
    {
        if (pendingMaterials.Count == 0) return;
        cl.ResourceBarrierTransition(gpu.MatBuf, ResourceStates.AllShaderResource, ResourceStates.CopyDest, D3D12.ResourceBarrierAllSubResources, ResourceBarrierFlags.None);
        foreach (var (index, material) in pendingMaterials)
        {
            var a = ring.Alloc((ulong)sizeof(Material), 16);
            *(Material*)a.Cpu = material;
            cl.CopyBufferRegion(gpu.MatBuf, (ulong)(index * sizeof(Material)), ring.Buffer, a.Offset, (ulong)sizeof(Material));
        }
        cl.ResourceBarrierTransition(gpu.MatBuf, ResourceStates.CopyDest, ResourceStates.AllShaderResource, D3D12.ResourceBarrierAllSubResources, ResourceBarrierFlags.None);
        pendingMaterials.Clear();
    }

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
        var proj = Matrix4x4.CreatePerspectiveFieldOfViewLeftHanded(camera.FovY, aspect, 0.05f, 200f);
        var viewProj = view * proj;
        Matrix4x4.Invert(viewProj, out var invViewProj);
        // the lights that shine this frame: none, the 3 main ones, or all of them
        activeLights = s.LightsOn ? (s.ManyLights ? scene.Lights.Length : Scene.MainLights) : 0;
        if (lastMode != s.Mode || activeLights != lastActiveLights) historyValid = false;
        if (!historyValid) { prevViewProj = viewProj; prevCamPos = camera.Position; }
        bool pathTraced = s.Mode is RenderMode.PathTrace or RenderMode.Hybrid;

        var lightsAlloc = ring.Alloc((ulong)(Math.Max(1, activeLights) * sizeof(LightData)), (ulong)sizeof(LightData));
        for (int i = 0; i < activeLights; i++) ((LightData*)lightsAlloc.Cpu)[i] = scene.Lights[i];
        CreateStructuredSrv(ring.Buffer, SlotLights + fi, Math.Max(1, activeLights), sizeof(LightData), lightsAlloc.Offset / (ulong)sizeof(LightData));

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
            NumLights = (uint)activeLights,
            LightsIndex = (uint)(SlotLights + fi),
            ShadowedLights = (uint)Math.Min(activeLights, Scene.MainLights),
            MaxBounces = (uint)(pathTraced ? s.PtBounces : s.RtBounces),
            Spp = (uint)s.Spp,
            Ambient = pathTraced ? Vector3.Zero : scene.Ambient,
            Exposure = s.Exposure,
            Light0 = scene.Lights[0], Light1 = scene.Lights[1], Light2 = scene.Lights[2],
            VbIndex = (uint)gpu.VbSlot, IbIndex = (uint)gpu.IbSlot, MatIndex = (uint)gpu.MatSlot, TlasIndex = (uint)gpu.TlasSlot,
            ShadowIndex = SlotShadow, ProbeIndex = SlotProbe, StaticIndexCount = (uint)scene.StaticIndexCount,
            ProbePos = scene.ProbePositions[0], ShadowNear = ShadowNear, ShadowFar = scene.ShadowFar,
            MeshFirstIndex = (uint)scene.MeshFirstIndex,
            GlassFirstIndex = (uint)scene.GlassFirstIndex,
            AlphaFirstIndex = (uint)scene.AlphaFirstIndex,
            MeshMaterial = scene.HasMesh ? (uint)scene.MeshMaterialIndex : 0xFFFFFFFFu,
            SkyIntensity = scene.SkyIntensity,
            PixelSpread = 2 * MathF.Tan(camera.FovY / 2) / Height,
        };
        ulong frameCb = ring.Push(fd);
        cl.SetGraphicsRootConstantBufferView(1, frameCb);
        cl.SetComputeRootConstantBufferView(1, frameCb);
        ulong mainPass = ring.Push(new PassData { ViewProj = viewProj, InvViewProj = invViewProj, CamPos = camera.Position, ClipPlane = new Vector4(0, 0, 0, 1) });
        cl.SetComputeRootConstantBufferView(2, mainPass);

        UploadHud(ring);
        UploadMaterials(ring);

        switch (s.Mode)
        {
            case RenderMode.Raster:
                RenderRaster(ring, camera, view, proj, mainPass, s.AntiAliasing, s.LightsOn);
                break;
            case RenderMode.RayTrace:
                BuildTlas(ring, s.LightsOn);
                Transition(hdr, ResourceStates.UnorderedAccess);
                cl.SetPipelineState(psoRayTrace);
                RootConstants(true, SlotHdrUav, s.AntiAliasing ? 4u : 1u);
                Dispatch2D();
                break;
            case RenderMode.PathTrace:
                BuildTlas(ring, s.LightsOn);
                RenderPathTrace(s, fi, false);
                break;
            case RenderMode.Sdf:
                Transition(hdr, ResourceStates.UnorderedAccess);
                cl.SetPipelineState(psoSdf);
                RootConstants(true, SlotHdrUav, (uint)gpu.SdfSlot, (uint)scene.SdfPrims.Count, s.SdfAntiAliasing ? 4u : 1u, (uint)scene.SdfPlaneCount);
                Dispatch2D();
                break;
            case RenderMode.Hybrid:
                BuildTlas(ring, s.LightsOn);
                RenderGBuffer(mainPass, s.LightsOn);
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
        lastActiveLights = activeLights;
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
        const uint rowPitch = HudWidth * 4;
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

    static PassData MakePass(Matrix4x4 viewProj, Vector3 camPos, uint flags, Vector4 clip)
    {
        Matrix4x4.Invert(viewProj, out var inv);
        return new PassData { ViewProj = viewProj, InvViewProj = inv, CamPos = camPos, Flags = flags, ClipPlane = clip };
    }

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
            for (int i = 0; i < activeLights; i++)
            {
                RootConstants(false, (uint)scene.SphereFirstIndex, (uint)i);
                cl.DrawInstanced((uint)scene.SphereIndexCount, 1, 0, 0);
            }
        }
    }

    void DrawSky(bool msaa, int w, int h)
    {
        if (scene.SkyIntensity <= 0) return;
        cl.SetPipelineState(msaa ? psoSkyMsaa : psoSky);
        RootConstants(false, (uint)w, (uint)h);
        cl.DrawInstanced(3, 1, 0, 0);
    }

    void RenderRaster(UploadRing ring, Camera camera, Matrix4x4 view, Matrix4x4 proj, ulong mainPass, bool msaa, bool lightsOn)
    {
        var noClip = new Vector4(0, 0, 0, 1);
        var glassObjects = scene.Objects.Where(scene.IsGlass).ToList();

        // 1) omnidirectional shadow maps: 6 faces per light
        if (lightsOn)
        {
            Transition(shadowMap, ResourceStates.DepthWrite);
            cl.SetPipelineState(psoShadow);
            SetViewport(ShadowSize, ShadowSize);
            for (int l = 0; l < Math.Min(activeLights, Scene.MainLights); l++)
            {
                var lp = scene.Lights[l].Pos;
                for (int f = 0; f < 6; f++)
                {
                    var dsv = shadowDsv[l * 6 + f];
                    cl.OMSetRenderTargets(0, (CpuDescriptorHandle*)null, false, dsv);
                    cl.ClearDepthStencilView(dsv, ClearFlags.Depth, 1, 0);
                    cl.SetGraphicsRootConstantBufferView(2, ring.Push(MakePass(CubeFaceViewProj(lp, f, ShadowNear, scene.ShadowFar), lp, 0, noClip)));
                    // mirrors hug the walls (nothing behind them to shadow); glass lets the light through
                    DrawScene(o => o.MirrorIndex < 0 && !scene.IsGlass(o), false);
                }
            }
            Transition(shadowMap, ResourceStates.AllShaderResource);
        }

        // 2) dynamic reflection probes from the chrome sphere / model centers (each skips its own object)
        SetViewport(ProbeSize, ProbeSize);
        for (int p = 0; p < 2; p++)
        {
            if (!scene.ProbeUsed[p]) continue;
            var pos = scene.ProbePositions[p];
            Transition(probes[p], ResourceStates.RenderTarget);
            for (int f = 0; f < 6; f++)
            {
                cl.OMSetRenderTargets(probeRtv[p, f], probeDepth.Dsv);
                cl.ClearRenderTargetView(probeRtv[p, f], new Color4(0, 0, 0, 1));
                cl.ClearDepthStencilView(probeDepth.Dsv, ClearFlags.Depth, 1, 0);
                cl.SetGraphicsRootConstantBufferView(2, ring.Push(MakePass(CubeFaceViewProj(pos, f, 0.05f, 100f), pos, PassData.NoMirror | PassData.NoProbe, noClip)));
                int self = p;
                cl.SetPipelineState(psoMain);
                DrawScene(o => o.ProbeCenter != self, lightsOn);
                DrawSky(false, ProbeSize, ProbeSize);
            }
            Transition(probes[p], ResourceStates.AllShaderResource);
        }

        // 3) planar reflections: re-render the scene through each mirror
        SetViewport(Width, Height);
        foreach (var m in scene.Mirrors)
        {
            int mi = scene.Objects[m.ObjectIndex].MirrorIndex;
            var target = refl[mi];
            var reflect = Matrix4x4.CreateReflection(m.Plane);
            Transition(target, ResourceStates.RenderTarget);
            cl.OMSetRenderTargets(target.Rtv, depth.Dsv);
            cl.ClearRenderTargetView(target.Rtv, new Color4(0, 0, 0, 1));
            cl.ClearDepthStencilView(depth.Dsv, ClearFlags.Depth, 1, 0);
            cl.SetGraphicsRootConstantBufferView(2, ring.Push(MakePass(reflect * view * proj, Vector3.Transform(camera.Position, reflect),
                PassData.NoMirror, new Vector4(m.Plane.Normal, m.Plane.D))));
            int skip = m.ObjectIndex;
            cl.SetPipelineState(psoMainMirrored);
            DrawScene(o => scene.Objects.IndexOf(o) != skip, lightsOn);
            DrawSky(false, Width, Height);
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
        DrawScene(o => !scene.IsGlass(o), lightsOn);
        DrawSky(msaa, Width, Height);

        // 5) glass: needs the finished opaque image to refract, so copy it and draw the glass on top
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
    // Stage 4 (hybrid): rasterize primary visibility into a G-buffer, path trace the rest from it
    // ---------------------------------------------------------------------------------------------
    void RenderGBuffer(ulong mainPass, bool lightsOn)
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
        DrawScene(_ => true, lightsOn);
        Transition(gbufPos, ResourceStates.AllShaderResource);
        Transition(gbufNrm, ResourceStates.AllShaderResource);
    }

    // ---------------------------------------------------------------------------------------------
    // Stage 3: path tracing + denoiser
    // ---------------------------------------------------------------------------------------------
    void RenderPathTrace(RenderSettings s, int fi, bool hybrid)
    {
        int cur = (int)(frameCounter & 1), prev = 1 - cur;

        Transition(ptNoisy, ResourceStates.UnorderedAccess);
        Transition(ptGuide[cur], ResourceStates.UnorderedAccess);
        Transition(ptAlbedo, ResourceStates.UnorderedAccess);
        uint gPos = hybrid ? (uint)gbufPos.Srv : 0xFFFFFFFFu, gNrm = hybrid ? (uint)gbufNrm.Srv : 0xFFFFFFFFu;

        // ReSTIR DI: initial candidates + temporal reuse -> set 2, spatial reuse -> set cur (read by the path tracer)
        bool useRestir = s.Restir && activeLights > 0;
        uint setCur = (uint)(SlotRestir + 8 * cur), setPrev = (uint)(SlotRestir + 8 * prev), setTmp = SlotRestir + 16;
        if (useRestir)
        {
            for (int j = 0; j < 4; j++)
            {
                Transition(restir[cur, j], ResourceStates.UnorderedAccess);
                Transition(restir[prev, j], ResourceStates.AllShaderResource);
            }
            Transition(restir[2, 0], ResourceStates.UnorderedAccess);
            Transition(restir[2, 1], ResourceStates.UnorderedAccess);
            cl.SetPipelineState(psoRestirInitial);
            bool restirHistory = historyValid && restirFrame == frameCounter - 1;
            restirFrame = frameCounter;
            RootConstants(true, setCur, setPrev, setTmp, gPos, gNrm, restirHistory ? 0u : 1u);
            Dispatch2D();
            Transition(restir[2, 0], ResourceStates.AllShaderResource);
            Transition(restir[2, 1], ResourceStates.AllShaderResource);
            Transition(restir[cur, 2], ResourceStates.AllShaderResource);
            Transition(restir[cur, 3], ResourceStates.AllShaderResource);
            cl.SetPipelineState(psoRestirSpatial);
            RootConstants(true, setCur, setPrev, setTmp);
            Dispatch2D();
            Transition(restir[cur, 0], ResourceStates.AllShaderResource);
            Transition(restir[cur, 1], ResourceStates.AllShaderResource);
        }

        // rc7: 0 = PT, 1 = PT + ReSTIR, 2 = hybrid + ReSTIR, 3 = hybrid
        uint ptMode = hybrid ? (useRestir ? 2u : 3u) : (useRestir ? 1u : 0u);
        cl.SetPipelineState(psoPathTrace);
        RootConstants(true, (uint)ptNoisy.Uav, (uint)ptGuide[cur].Uav, (uint)ptAlbedo.Uav, gPos, gNrm, setCur, setCur + 1, ptMode);
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
        uint maxHistory = s.LightsAnimating && s.LightsOn ? 12u : 4096u;
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
        foreach (var g in sceneGpus.Values) g.Dispose();
        foreach (var d in new IDisposable[] { shadowMap, probes[0], probes[1], probeDepth, hud, queryHeap, queryReadback })
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
