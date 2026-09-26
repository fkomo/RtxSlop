# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

C# / .NET 10 WinForms + Direct3D 12 demo (via Vortice 3.8.3) that renders one room scene in five switchable modes: 1 rasterization, 2 Whitted ray tracing (DXR 1.1 inline `RayQuery`), 3 path tracing + denoiser, 4 hybrid (raster G-buffer + path tracing), 5 SDF sphere tracing. Needs a DXR 1.1 + Shader Model 6.6 GPU. README.md has the user-facing controls.

## Commands

```bash
dotnet build -c Release src/RtxSlop.slnx
dotnet run -c Release --project src/RtxSlop
```

There are no unit tests. Verification is visual, via the built-in capture mode:

```bash
src/RtxSlop/bin/Release/net10.0-windows/win-x64/RtxSlop.exe --capture <outDir>
```

`--bench <dir>` only writes `bench.txt` (GPU ms per mode × model material, front view, moving lights). GPU timings are unreliable while other apps use the GPU (the user often has a game or Blender open). `--capture` renders 3 fixed viewpoints in every mode (path tracing modes accumulate 256 frames with lights paused), a moving-lights PT shot, 1-spp raw/denoised shots, a resize test, and writes `timings.txt` (GPU ms per mode), then exits. Read the PNGs to check changes. In capture mode, fatal errors go to `error.log` next to the exe instead of a MessageBox. `--debug` enables the D3D12 debug layer if Graphics Tools is installed (it falls back silently otherwise; messages go to `d3d12-debug.log` in the working directory).

Shaders (`src/RtxSlop/Shaders/*.hlsl`) are copied to the output folder and compiled at startup with DXC. `F5` in the app recompiles them from the **output** folder, not from `src`. A running app locks the exe, so builds into `bin/` fail until it is closed.

## Architecture

- **One root signature for everything**, defined as the `RS` string in `Shaders/Common.hlsli` and embedded in every entry point via `[RootSignature(RS)]`. The C# side creates it from the VS_Main bytecode. Layout: `b0` = 8 root constants (`rc0..rc7`, meaning differs per shader and is documented at the top of each .hlsl), `b1` = FrameCB, `b2` = PassCB, plus static samplers.
- **Bindless (SM 6.6 `ResourceDescriptorHeap[]`)**: no descriptor tables or input layouts. Vertex shaders fetch vertices from structured buffers through the index buffer (`rc0` = first index), and draws are `DrawInstanced` without an IA. Descriptor slots are fixed constants at the top of `Renderer.cs` (`Slot*`). The PT textures occupy slots 20..39 (SRV = even, UAV = odd).
- **CPU/GPU struct layouts must match by hand**: `FrameData`/`PassData` (GpuTypes.cs) ↔ `FrameCB`/`PassCB` (Common.hlsli); `Vertex`, `Material` (Scene.cs) ↔ Common.hlsli; `SdfPrim` (Scene.cs) ↔ Sdf.hlsl. Matrices are System.Numerics row-vector, declared `row_major` in HLSL and used as `mul(v, M)`. The coordinate system is left-handed, +Y up (`*LeftHanded` matrix helpers), and triangles are clockwise-front.
- **Scene.cs** builds everything procedurally: one shared vertex/index buffer (static objects first, `StaticIndexCount`, then a unit sphere used for the light bulbs), materials, per-object index ranges (used by the raster passes to skip objects), mirror planes, and a parallel list of analytic `SdfPrim`s for mode 5. The 6 room half-spaces must stay first in `SdfPrims` (`SdfPlaneCount`), because the SDF marcher intersects them analytically.
- **Ray tracing**: one static BLAS plus one unit-sphere BLAS instanced per light. The TLAS is rebuilt every frame (`BuildTlas`). InstanceID 0 = static scene, 1+i = light i; mask 0x01 = geometry, 0x02 = lights (shadow rays use 0x01). Only static-instance hits fetch triangle data (`GetSurface` in Common.hlsli).
- **Renderer.Render** is a single command list per frame with 2 frames in flight; per-frame CPU data goes through `UploadRing`. Resource states are tracked manually on `GpuTexture.State` via `Transition()`. Every mode writes the HDR target, then `PS_Tonemap` (ACES + GDI-drawn HUD texture) writes the sRGB back buffer.
- **Path tracer + denoiser** (modes 3/4): `CS_PathTrace` writes albedo-demodulated radiance, albedo, and guides (normal + "virtual" distance). The guides come from a deterministic pixel-center walk through perfect mirrors (primary surface replacement). Then `CS_Temporal` (reprojection of the virtual position, ping-pong history/moments by frame parity) and 4 `CS_Atrous` iterations, the last of which remodulates into HDR. In hybrid mode, `rc3`/`rc4` point at the G-buffer, and depth 0 of both the path and the guide walk is read from it instead of traced.
- **Raster mode** passes, in order: 18 cube-shadow faces (cube array, cull-front) → 6 reflection-probe faces at the chrome sphere → one planar-reflection pass per mirror (mirrored PSO, clip plane) → main pass (optional 4× MSAA + resolve). `PassFlags` stop mirrors and the probe from sampling themselves in the secondary passes.

- **Polygon model** (`ObjLoader.cs`, `Scene.AddMesh`): its index range sits after the static geometry (`MeshFirstIndex`) and is drawn as a normal raster object, but it has its own BLAS and TLAS instance: `InstanceID` 100 (`INSTANCE_MESH`), mask 0x04 (`MASK_MESH`). Shaders use `IsLightInstance()`, and `GetSurface` adds `gMeshFirstIndex` for mesh hits. Its material is a single buffer entry (`MeshMaterialIndex`) that is rewritten at runtime (`Scene.SetMeshPreset` → `Renderer.SetMaterial`, uploaded at the start of the next frame, which also resets the PT history). Glass (`MAT_GLASS`) is handled separately in each mode; see the README table.
- **Glass generally**: any material with `MAT_GLASS` works in PT/hybrid, raster (drawn last with `PS_Glass`, no raster shadow), RT and SDF. Static glass objects must be added last among the static objects (`Scene.GlassFirstIndex`..`StaticIndexCount`): they get their own BLAS/instance (`INSTANCE_GLASS` 101, `MASK_GLASS` 0x08, `gGlassFirstIndex`) so the RT `ShadowTransmittance` stays a few cheap queries. SDF tracks glass per primitive (`gsGlass`) and marches inside the primitive it entered.
- **SDF performance**: soft shadows and AO only evaluate primitives whose bounding sphere can matter (`PrimsNearSegment`/`PrimsNearPoint` bitmasks). Without that culling the dragon made SDF ~3× slower.
- **Raster probes**: probe 0 is at the chrome sphere, probe 1 at the model. Materials select their probe through `Material.TexIndex` (`Renderer.ProbeSlot`), and `SceneObject.ProbeCenter` excludes the object from its own probe. Glass is drawn last (`PS_Glass`) after copying the opaque HDR image into `sceneColor`.
- **SDF mesh**: `SdfBake.hlsl` runs once at startup inside `CreateSceneResources` (it needs the TLAS for the inside/outside sign). `Sdf.hlsl` samples the result as primitive type `SDF_MESH` (always last in `SdfPrims`).

## Gotchas

- `BlendDescription.Opaque` in Vortice only fills `RenderTarget[0]`; use `OpaqueBlend()` for MRT PSOs. Integer-format render targets failed PSO creation, so the G-buffer stores the material id as a float.
- Vortice.Direct3D12 3.8.3 contains a DRED type that fails to load under reflection; avoid DRED APIs.
- `Renderer.Dispose` must stay idempotent (WinForms plus `using` dispose the form twice), and `MainForm.OnKeyDown` must not touch the HUD after `Close()`.
