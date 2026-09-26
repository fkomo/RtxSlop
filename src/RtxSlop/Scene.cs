using System.Numerics;
using System.Runtime.InteropServices;

namespace RtxSlop;

/// <summary>40 bytes, matches Vertex in Common.hlsli.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct Vertex
{
    public Vector3 Pos;
    public Vector3 Normal;
    public Vector2 UV;
    public uint Material;
    public float SkyVis;     // fraction of the hemisphere that sees the sky (baked on the GPU at load)
}

/// <summary>64 bytes, matches Material in Common.hlsli.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct Material
{
    public Vector3 Albedo;
    public float Roughness;
    public Vector3 Emission;
    public float Metallic;
    public uint Type;
    public uint TexIndex;    // SRV of the planar reflection (mirror) or reflection probe (probe/glass materials)
    public uint Flags;
    public float Ior;        // glass only
    public uint AlbedoTex;   // scene texture id + 1 (0 = none); the renderer swaps in the SRV slot on upload
    public uint AlphaTex;    // same, alpha-test mask (red channel)
    public uint Pad0, Pad1;

    public const uint Standard = 0, Checker = 1, Mirror = 2, Glass = 3;
    public const uint FlagProbe = 1, FlagAlphaTest = 2, FlagAlphaInAlbedo = 4;   // alpha test on the base color's alpha channel
}

[StructLayout(LayoutKind.Sequential)]
public struct LightData
{
    public Vector3 Pos;
    public float Radius;
    public Vector3 Color;
    public float Intensity;
}

/// <summary>Analytic primitive for the SDF ray marcher (48 bytes, matches SdfPrim in Sdf.hlsl).</summary>
[StructLayout(LayoutKind.Sequential)]
public struct SdfPrim
{
    public Vector3 Center;   // plane: Center.X = plane distance d
    public uint Type;
    public Vector3 Size;     // box: half extents, sphere: X = radius, plane: normal, mesh: grid half extents
    public uint Material;
    public float CosR, SinR; // box rotation around Y; mesh: bit-cast SRV slots of the distance / triangle-id grids
    public Vector2 Pad;      // mesh: bit-cast flags, voxel size

    public const uint Plane = 0, Box = 1, Sphere = 2, Mesh = 3;
}

/// <summary>A triangle range the renderer bakes into a distance grid for the SDF mode.</summary>
public sealed class SdfMesh
{
    public int PrimIndex;            // its SdfPrim (Type = Mesh), filled in by the renderer after baking
    public int FirstIndex, IndexCount;
    public bool Signed;              // closed solid -> signed; open architecture -> unsigned shell
    public int Resolution;           // voxels along the longest axis
}

public sealed class SceneObject
{
    public string Name;
    public int FirstIndex;
    public int IndexCount;
    public int MirrorIndex = -1;     // >= 0 if this object is a planar mirror
    public int ProbeCenter = -1;     // >= 0: reflection probe captured from inside this object (it is skipped there)
    public int Material = -1;        // material of the object's vertices
    public bool IsMesh;              // the switchable .obj model
}

public sealed class Mirror
{
    public Plane Plane;              // normal points into the room
    public int ObjectIndex;
    public int MaterialIndex;
}

/// <summary>
/// Everything the renderer needs to draw one scene. Two scenes exist:
///  - <see cref="CreateRoom"/>: the procedural test room (mirrors, spheres, glass ball, dragon)
///  - <see cref="CreateSponza"/>: a pure triangle mesh (Crytek Sponza, textured) with the dragon in its atrium
/// Index buffer layout (the ray tracers rely on it):
///   [0, AlphaFirstIndex) opaque static | [AlphaFirstIndex, GlassFirstIndex) alpha-tested static |
///   [GlassFirstIndex, StaticIndexCount) static glass | model | light bulb sphere
/// Coordinate system: left-handed, +Y up, meters.
/// </summary>
public sealed class Scene
{
    public string Name;
    public readonly List<Vertex> Vertices = new();
    public readonly List<uint> Indices = new();
    public readonly List<Material> Materials = new();
    public readonly List<SceneObject> Objects = new();
    public readonly List<Mirror> Mirrors = new();
    public readonly LightData[] Lights = new LightData[3];
    public readonly List<SdfPrim> SdfPrims = new();
    public readonly List<SdfMesh> SdfMeshes = new();
    public int SdfPlaneCount;        // leading half-space prims (the room only)
    public readonly List<(string Path, bool Srgb)> Textures = new();

    public int AlphaFirstIndex, GlassFirstIndex, StaticIndexCount;
    public int SphereFirstIndex, SphereIndexCount;
    public readonly Vector3[] ProbePositions = new Vector3[2];   // 0 = chrome sphere (room), 1 = model
    public readonly bool[] ProbeUsed = new bool[2];

    // the switchable model (index range after the static geometry, own BLAS / instance)
    public bool HasMesh;
    public int MeshFirstIndex, MeshIndexCount, MeshMaterialIndex = -1;
    public Vector3 MeshMin, MeshMax;
    public int MeshPreset;
    public (string Name, Material Material)[] MeshPresets;

    // lighting / camera
    public float SkyIntensity;       // 0 = closed room, no sky
    public Vector3 Ambient;          // constant fill light of the non path traced modes
    public float ShadowFar = 30;
    public Vector3 BoundsMin, BoundsMax;
    public Vector3 CameraPos;
    public float CameraYaw, CameraPitch;
    Action<Scene, float> animateLights;

    public void UpdateLights(float t) => animateLights(this, t);

    public Material SetMeshPreset(int preset)
    {
        MeshPreset = ((preset % MeshPresets.Length) + MeshPresets.Length) % MeshPresets.Length;
        Materials[MeshMaterialIndex] = MeshPresets[MeshPreset].Material;
        return Materials[MeshMaterialIndex];
    }

    public bool IsGlass(SceneObject o) => o.Material >= 0 && Materials[o.Material].Type == Material.Glass;

    static (string, Material)[] ModelPresets(Func<int, uint> probeSlot) => new (string, Material)[]
    {
        ("jade", new Material { Albedo = new Vector3(0.18f, 0.55f, 0.36f), Roughness = 0.45f }),
        ("glass", new Material { Albedo = new Vector3(0.88f, 0.97f, 0.93f), Roughness = 0, Type = Material.Glass, Ior = 1.5f, TexIndex = probeSlot(1) }),
        ("gold", new Material { Albedo = new Vector3(1.0f, 0.78f, 0.34f), Roughness = 0.2f, Metallic = 1, Flags = Material.FlagProbe, TexIndex = probeSlot(1) }),
        ("mirror", new Material { Albedo = new Vector3(0.95f, 0.95f, 0.95f), Roughness = 0.01f, Metallic = 1, Flags = Material.FlagProbe, TexIndex = probeSlot(1) }),
    };

    // =============================================================================================
    // Scene 1: the room
    // =============================================================================================
    public const float HalfX = 6f, HalfZ = 6f, Height = 4f;

    /// <summary>
    /// 12 x 4 x 12 m room, Cornell-box colored side walls, two facing mirrors, a few obstacles,
    /// chrome / gold / glass spheres, the model in the middle and three animated sphere lights.
    /// </summary>
    public static Scene CreateRoom(Func<int, uint> mirrorTextureSlot, Func<int, uint> probeSlot, ObjMesh mesh)
    {
        var s = new Scene
        {
            Name = "room",
            Ambient = new Vector3(0.025f, 0.025f, 0.03f),
            ShadowFar = 30,
            BoundsMin = new Vector3(-HalfX + 0.25f, 0.3f, -HalfZ + 0.25f),
            BoundsMax = new Vector3(HalfX - 0.25f, Height - 0.2f, HalfZ - 0.25f),
            CameraPos = new Vector3(0, 1.7f, -4.6f),
            MeshPresets = ModelPresets(probeSlot),
            animateLights = AnimateRoomLights,
        };
        int mFloor = s.AddMaterial(new Vector3(0.85f, 0.85f, 0.82f), 0.08f, 0, Material.Checker);
        int mWhite = s.AddMaterial(new Vector3(0.75f, 0.75f, 0.72f), 0.9f, 0);
        int mRed = s.AddMaterial(new Vector3(0.70f, 0.10f, 0.08f), 0.9f, 0);
        int mGreen = s.AddMaterial(new Vector3(0.12f, 0.48f, 0.14f), 0.9f, 0);
        int mCeiling = s.AddMaterial(new Vector3(0.8f, 0.8f, 0.8f), 0.95f, 0);
        int mStone = s.AddMaterial(new Vector3(0.6f, 0.58f, 0.55f), 0.8f, 0);
        int mBlue = s.AddMaterial(new Vector3(0.08f, 0.2f, 0.65f), 0.35f, 0);
        int mOrange = s.AddMaterial(new Vector3(0.9f, 0.4f, 0.05f), 0.5f, 0);
        int mChrome = s.AddMaterial(new Vector3(0.95f, 0.95f, 0.95f), 0.02f, 1, flags: Material.FlagProbe, tex: probeSlot(0));
        int mGold = s.AddMaterial(new Vector3(1.0f, 0.78f, 0.34f), 0.25f, 1, flags: Material.FlagProbe, tex: probeSlot(0));
        int mGlass = s.AddMaterial(new Vector3(0.92f, 0.97f, 1.0f), 0, 0, Material.Glass, tex: probeSlot(0));
        s.Materials[mGlass] = s.Materials[mGlass] with { Ior = 1.5f };
        s.mFrame = s.AddMaterial(new Vector3(0.09f, 0.055f, 0.03f), 0.5f, 0);

        // SDF version of the shell: 6 inward-facing half spaces (dot(p, n) + d), they must come first
        s.AddSdfPlane(Vector3.UnitY, 0, mFloor);
        s.AddSdfPlane(-Vector3.UnitY, Height, mCeiling);
        s.AddSdfPlane(Vector3.UnitX, HalfX, mRed);
        s.AddSdfPlane(-Vector3.UnitX, HalfX, mGreen);
        s.AddSdfPlane(-Vector3.UnitZ, HalfZ, mWhite);
        s.AddSdfPlane(Vector3.UnitZ, HalfZ, mWhite);
        s.SdfPlaneCount = s.SdfPrims.Count;

        // ---- room shell (inward facing) ----
        s.Begin("Floor");
        s.AddQuad(new(-HalfX, 0, -HalfZ), new(HalfX, 0, -HalfZ), new(HalfX, 0, HalfZ), new(-HalfX, 0, HalfZ), Vector3.UnitY, mFloor);
        s.End();
        s.Begin("Ceiling");
        s.AddQuad(new(-HalfX, Height, -HalfZ), new(HalfX, Height, -HalfZ), new(HalfX, Height, HalfZ), new(-HalfX, Height, HalfZ), -Vector3.UnitY, mCeiling);
        s.End();
        s.Begin("WallLeft");
        s.AddQuad(new(-HalfX, 0, -HalfZ), new(-HalfX, 0, HalfZ), new(-HalfX, Height, HalfZ), new(-HalfX, Height, -HalfZ), Vector3.UnitX, mRed);
        s.End();
        s.Begin("WallRight");
        s.AddQuad(new(HalfX, 0, -HalfZ), new(HalfX, 0, HalfZ), new(HalfX, Height, HalfZ), new(HalfX, Height, -HalfZ), -Vector3.UnitX, mGreen);
        s.End();
        s.Begin("WallBack");
        s.AddQuad(new(-HalfX, 0, HalfZ), new(HalfX, 0, HalfZ), new(HalfX, Height, HalfZ), new(-HalfX, Height, HalfZ), -Vector3.UnitZ, mWhite);
        s.End();
        s.Begin("WallFront");
        s.AddQuad(new(-HalfX, 0, -HalfZ), new(HalfX, 0, -HalfZ), new(HalfX, Height, -HalfZ), new(-HalfX, Height, -HalfZ), Vector3.UnitZ, mWhite);
        s.End();

        // ---- two mirrors facing each other ----
        s.AddMirror("MirrorBack", new Vector3(0, 1.9f, HalfZ), 2.6f, 1.5f, -Vector3.UnitZ, mirrorTextureSlot(0));
        s.AddMirror("MirrorFront", new Vector3(0, 1.75f, -HalfZ), 2.0f, 1.25f, Vector3.UnitZ, mirrorTextureSlot(1));

        // ---- obstacles ----
        s.Begin("Pillar"); s.AddBox(new(-3.5f, 1.4f, 3.2f), new(0.4f, 1.4f, 0.4f), 0, mStone); s.End();
        s.Begin("BoxBlue"); s.AddBox(new(3.0f, 0.6f, -2.5f), new(0.7f, 0.6f, 0.7f), 30, mBlue); s.End();
        s.Begin("BoxOrange"); s.AddBox(new(3.0f, 1.5f, -2.5f), new(0.3f, 0.3f, 0.3f), 15, mOrange); s.End();
        s.Begin("LowWall"); s.AddBox(new(-2.5f, 0.5f, -4.0f), new(1.25f, 0.5f, 0.15f), 0, mWhite); s.End();
        s.Begin("PillarRight"); s.AddBox(new(4.6f, 1.0f, 2.6f), new(0.25f, 1.0f, 0.25f), 45, mStone); s.End();

        s.ProbePositions[0] = new Vector3(1.5f, 0.9f, 3.3f);
        s.ProbeUsed[0] = true;
        var chrome = s.Begin("ChromeSphere"); chrome.ProbeCenter = 0;
        s.AddSphere(s.ProbePositions[0], 0.9f, mChrome, 64, 32); s.End();
        s.Begin("GoldSphere"); s.AddSphere(new(-1.5f, 0.6f, -2.5f), 0.6f, mGold, 48, 24); s.End();
        s.AlphaFirstIndex = s.Indices.Count;   // no alpha-tested geometry in this scene

        // static glass goes last: it gets its own BLAS/instance so shadow rays can treat it separately
        s.GlassFirstIndex = s.Indices.Count;
        s.Begin("GlassSphere"); s.AddSphere(new(3.2f, 0.7f, 1.5f), 0.7f, mGlass, 64, 32); s.End();
        s.StaticIndexCount = s.Indices.Count;

        if (mesh != null) s.AddMesh(mesh, new Vector3(0, 0, 0), 3.0f, 2.0f, 90);
        s.AddLightBulbMesh(mWhite);
        s.UpdateLights(0);
        return s;
    }

    static void AnimateRoomLights(Scene s, float t)
    {
        s.Lights[0] = new LightData
        {
            Pos = new Vector3(3.8f * MathF.Cos(0.5f * t), 2.6f + 0.4f * MathF.Sin(1.3f * t), 3.8f * MathF.Sin(0.5f * t)),
            Radius = 0.12f, Color = new Vector3(1.0f, 0.72f, 0.42f), Intensity = 7f,
        };
        s.Lights[1] = new LightData
        {
            Pos = new Vector3(2.2f * MathF.Cos(-0.8f * t + 2f), 1.2f + 0.3f * MathF.Sin(2.1f * t), 2.2f * MathF.Sin(-0.8f * t + 2f)),
            Radius = 0.1f, Color = new Vector3(0.45f, 0.65f, 1.0f), Intensity = 4f,
        };
        s.Lights[2] = new LightData
        {
            Pos = new Vector3(4.0f * MathF.Sin(0.35f * t), 3.2f, 3.0f * MathF.Sin(0.7f * t)),
            Radius = 0.12f, Color = new Vector3(1.0f, 0.35f, 0.7f), Intensity = 4f,
        };
    }

    // =============================================================================================
    // Scene 2: Crytek Sponza (pure triangle mesh, textured, open to the sky) + the model
    // =============================================================================================
    Vector3 lightCenter;   // the lights orbit around this point
    Vector3 lightSpread = Vector3.One;

    /// <summary>Where things go in an imported site. All in meters, after <see cref="Scale"/>.</summary>
    public sealed record SiteLayout(float Scale, Vector3 Center, Vector3 CameraPos, float CameraYaw, float CameraPitch,
                                    float ModelYawDeg, Vector3 LightSpread, int SdfResolution);

    /// <summary>Crytek Sponza (data/sponza2): centimeters, atrium centered slightly off the origin.</summary>
    public static Scene CreateSponza(string objPath, Func<int, uint> probeSlot, ObjMesh mesh)
    {
        var c = new Vector3(-0.65f, 0, 0.35f);
        return CreateOpenArchitecture("sponza (crytek)", objPath, new SiteLayout(0.01f, c, c + new Vector3(-10.85f, 1.7f, 0), MathF.PI / 2, 0.08f, 0, Vector3.One, 384), probeSlot, mesh);
    }

    /// <summary>The original Sponza by Marko Dabrovic (data/sponza): meters, no normals in the file.</summary>
    public static Scene CreateDabrovicSponza(string objPath, Func<int, uint> probeSlot, ObjMesh mesh)
    {
        var c = new Vector3(0, 0.09f, 0);
        return CreateOpenArchitecture("sponza (dabrovic)", objPath, new SiteLayout(1.0f, c, c + new Vector3(-10.85f, 1.7f, 0), MathF.PI / 2, 0.08f, 0, Vector3.One, 384), probeSlot, mesh);
    }

    /// <summary>San Miguel 2.1 (data/san_miguel; low-poly 5.6M or full 10M triangles): Mexican courtyard with lots of foliage.</summary>
    public static Scene CreateSanMiguel(string name, string objPath, Func<int, uint> probeSlot, ObjMesh mesh, SiteLayout layout) =>
        CreateOpenArchitecture(name, objPath, layout, probeSlot, mesh);

    /// <summary>
    /// A textured .obj site open to the sky, the model standing in the middle of its courtyard
    /// (layout.Center = floor point in meters) and three lights orbiting above it.
    /// </summary>
    static Scene CreateOpenArchitecture(string name, string objPath, SiteLayout layout, Func<int, uint> probeSlot, ObjMesh mesh)
    {
        float scale = layout.Scale;
        var atriumCenter = layout.Center;
        var s = new Scene
        {
            Name = name,
            SkyIntensity = 5.0f,
            Ambient = Vector3.Zero,
            ShadowFar = 80,
            CameraPos = layout.CameraPos,
            CameraYaw = layout.CameraYaw,
            CameraPitch = layout.CameraPitch,
            MeshPresets = ModelPresets(probeSlot),
            animateLights = AnimateSponzaLights,
            lightCenter = atriumCenter,
            lightSpread = layout.LightSpread,
        };

        var obj = ObjLoader.Load(objPath);
        string dir = Path.GetDirectoryName(Path.GetFullPath(objPath));
        var mtls = obj.MtlLib != null && File.Exists(Path.Combine(dir, obj.MtlLib))
            ? ObjLoader.LoadMtl(Path.Combine(dir, obj.MtlLib)).GroupBy(m => m.Name).ToDictionary(g => g.Key, g => g.First())
            : new Dictionary<string, ObjMaterial>();

        // one scene material per .obj material (+ a fallback for faces without one)
        int fallback = s.AddMaterial(new Vector3(0.7f), 0.8f, 0);
        // kind per .obj material: 0 = opaque, 1 = alpha tested (map_d, or a base color PNG with alpha), 2 = glass (d < 1)
        int matCount = obj.MaterialNames.Count;
        var matOf = new int[matCount + 1];
        var kind = new int[matCount + 1];
        matOf[matCount] = fallback;   // index matCount = faces without a material
        for (int i = 0; i < matCount; i++)
        {
            mtls.TryGetValue(obj.MaterialNames[i], out var m);
            string kdPath = m?.MapKd != null ? Path.Combine(dir, m.MapKd) : null;
            bool alphaInAlbedo = m?.MapD == null && ObjLoader.PngHasAlpha(kdPath);
            kind[i] = m != null && m.D < 1 ? 2 : (m?.MapD != null || alphaInAlbedo) ? 1 : 0;
            uint albedoTex = s.AddTexture(dir, m?.MapKd, srgb: true);
            s.Materials.Add(kind[i] == 2
                ? new Material { Albedo = m.Kd, Type = Material.Glass, Ior = 1.5f, TexIndex = probeSlot(1) }
                : new Material
                {
                    Albedo = m?.Kd ?? new Vector3(0.7f),
                    Roughness = 0.8f,
                    AlbedoTex = albedoTex,
                    AlphaTex = alphaInAlbedo ? albedoTex : s.AddTexture(dir, m?.MapD, srgb: false),
                    Flags = kind[i] == 1 ? Material.FlagAlphaTest | (alphaInAlbedo ? Material.FlagAlphaInAlbedo : 0) : 0,
                });
            matOf[i] = s.Materials.Count - 1;
        }

        // bucket the triangles per material (millions of them for San Miguel - no LINQ grouping)
        var tris = obj.TriangleMaterials;
        var bucketStart = new int[matCount + 2];
        foreach (int tm in tris) bucketStart[(tm < 0 ? matCount : tm) + 1]++;
        for (int i = 0; i < matCount + 1; i++) bucketStart[i + 1] += bucketStart[i];
        var fillPos = (int[])bucketStart.Clone();
        var order = new int[tris.Length];
        for (int t = 0; t < tris.Length; t++) order[fillPos[tris[t] < 0 ? matCount : tris[t]]++] = t;

        // one object per material; opaque first, then alpha tested (foliage, chains), then glass
        s.Vertices.Capacity = obj.Positions.Length + 1024;
        s.Indices.Capacity = obj.Indices.Length + 4096;
        var remap = new int[obj.Positions.Length];
        var stamp = new int[obj.Positions.Length];   // remap[v] is valid while stamp[v] == current bucket + 1
        s.AlphaFirstIndex = s.GlassFirstIndex = -1;
        for (int pass = 0; pass < 3; pass++)
        {
            if (pass == 1) s.AlphaFirstIndex = s.Indices.Count;
            if (pass == 2) s.GlassFirstIndex = s.Indices.Count;
            for (int b = 0; b <= matCount; b++)
            {
                if (bucketStart[b] == bucketStart[b + 1] || kind[b] != pass) continue;
                int mat = matOf[b];
                s.Begin(b < matCount ? obj.MaterialNames[b] : "default");
                s.current.Material = mat;
                for (int o = bucketStart[b]; o < bucketStart[b + 1]; o++)
                {
                    int t = order[o];
                    for (int k = 0; k < 3; k++)
                    {
                        int v = obj.Indices[t * 3 + k];
                        if (stamp[v] != b + 1)
                        {
                            stamp[v] = b + 1;
                            remap[v] = s.Vertices.Count;
                            s.Vertices.Add(new Vertex { Pos = obj.Positions[v] * scale, Normal = obj.Normals[v], UV = obj.UVs[v], Material = (uint)mat });
                        }
                        s.Indices.Add((uint)remap[v]);
                    }
                }
                s.End();
            }
        }
        s.StaticIndexCount = s.Indices.Count;

        s.BoundsMin = obj.Min * scale + new Vector3(0.3f);
        s.BoundsMax = obj.Max * scale - new Vector3(0.3f);
        // the whole site as one open (unsigned) distance grid
        s.SdfMeshes.Add(new SdfMesh { FirstIndex = 0, IndexCount = s.StaticIndexCount, Signed = false, Resolution = layout.SdfResolution });
        s.SdfPrims.Add(new SdfPrim { Type = SdfPrim.Mesh, Center = (obj.Min + obj.Max) * 0.5f * scale, Size = (obj.Max - obj.Min) * 0.5f * scale + new Vector3(0.1f) });
        s.SdfMeshes[^1].PrimIndex = s.SdfPrims.Count - 1;

        int mWhite = s.AddMaterial(new Vector3(0.75f), 0.9f, 0);
        if (mesh != null) s.AddMesh(mesh, atriumCenter, 3.0f, 2.0f, layout.ModelYawDeg);
        s.AddLightBulbMesh(mWhite);
        s.UpdateLights(0);
        return s;
    }

    static void AnimateSponzaLights(Scene s, float t)
    {
        var c = s.lightCenter;
        var k = s.lightSpread;   // x/z = orbit size, y = height scale
        s.Lights[0] = new LightData
        {
            Pos = c + k * new Vector3(7.0f * MathF.Cos(0.3f * t), 3.5f + 0.8f * MathF.Sin(0.9f * t), 1.6f * MathF.Sin(0.3f * t)),
            Radius = 0.15f, Color = new Vector3(1.0f, 0.72f, 0.42f), Intensity = 40f,
        };
        s.Lights[1] = new LightData
        {
            Pos = c + k * new Vector3(3.8f * MathF.Cos(-0.5f * t + 2f), 2.8f + 0.3f * MathF.Sin(1.7f * t), 1.7f * MathF.Sin(-0.5f * t + 2f)),
            Radius = 0.12f, Color = new Vector3(0.45f, 0.65f, 1.0f), Intensity = 20f,
        };
        s.Lights[2] = new LightData
        {
            Pos = c + k * new Vector3(9.0f * MathF.Sin(0.2f * t), 7.5f, 1.0f * MathF.Sin(0.4f * t)),
            Radius = 0.15f, Color = new Vector3(1.0f, 0.35f, 0.7f), Intensity = 40f,
        };
    }

    // =============================================================================================
    // Building blocks
    // =============================================================================================
    int mFrame;
    SceneObject current;

    /// <summary>Texture reference for a material: file path relative to the .mtl -> id + 1 (0 = none).</summary>
    uint AddTexture(string dir, string relative, bool srgb)
    {
        if (string.IsNullOrEmpty(relative)) return 0;
        string path = Path.GetFullPath(Path.Combine(dir, relative));
        if (!File.Exists(path)) return 0;
        int i = Textures.FindIndex(t => string.Equals(t.Path, path, StringComparison.OrdinalIgnoreCase) && t.Srgb == srgb);
        if (i < 0) { Textures.Add((path, srgb)); i = Textures.Count - 1; }
        return (uint)(i + 1);
    }

    /// <summary>
    /// Places the model standing on the floor at <paramref name="floorCenter"/>, turned to be seen in profile
    /// and scaled to fit maxLength x maxHeight. Gets its own material (switchable presets), a reflection probe
    /// at its center and a signed distance grid for the SDF mode.
    /// </summary>
    void AddMesh(ObjMesh mesh, Vector3 floorCenter, float maxLength, float maxHeight, float rotYDeg)
    {
        var rot = Matrix4x4.CreateRotationY(rotYDeg * MathF.PI / 180f);
        var positions = mesh.Positions.Select(p => Vector3.Transform(p, rot)).ToArray();
        var normals = mesh.Normals.Select(n => Vector3.TransformNormal(n, rot)).ToArray();
        var bmin = positions.Aggregate(new Vector3(float.MaxValue), Vector3.Min);
        var bmax = positions.Aggregate(new Vector3(float.MinValue), Vector3.Max);

        var size = bmax - bmin;
        float scale = MathF.Min(maxLength / MathF.Max(size.X, size.Z), maxHeight / size.Y);
        var center = (bmin + bmax) * 0.5f;
        var offset = floorCenter + new Vector3(-center.X * scale, -bmin.Y * scale, -center.Z * scale);

        MeshPreset = 1;   // glass
        MeshMaterialIndex = Materials.Count;
        Materials.Add(MeshPresets[MeshPreset].Material);

        uint baseV = (uint)Vertices.Count;
        for (int i = 0; i < positions.Length; i++)
            Vertices.Add(new Vertex { Pos = positions[i] * scale + offset, Normal = normals[i], UV = mesh.UVs[i], Material = (uint)MeshMaterialIndex });

        var obj = Begin("Mesh");
        obj.IsMesh = true;
        obj.Material = MeshMaterialIndex;
        obj.ProbeCenter = 1;
        MeshFirstIndex = Indices.Count;
        foreach (int i in mesh.Indices) Indices.Add(baseV + (uint)i);
        MeshIndexCount = Indices.Count - MeshFirstIndex;
        End();

        MeshMin = bmin * scale + offset;
        MeshMax = bmax * scale + offset;
        ProbePositions[1] = (MeshMin + MeshMax) * 0.5f;
        ProbeUsed[1] = true;
        HasMesh = true;

        SdfPrims.Add(new SdfPrim { Type = SdfPrim.Mesh, Center = (MeshMin + MeshMax) * 0.5f, Size = (MeshMax - MeshMin) * 0.5f + new Vector3(0.1f), Material = (uint)MeshMaterialIndex });
        SdfMeshes.Add(new SdfMesh { PrimIndex = SdfPrims.Count - 1, FirstIndex = MeshFirstIndex, IndexCount = MeshIndexCount, Signed = true, Resolution = 128 });
    }

    /// <summary>Unit sphere used (instanced) for the light bulbs.</summary>
    void AddLightBulbMesh(int material)
    {
        SphereFirstIndex = Indices.Count;
        AddSphere(Vector3.Zero, 1f, material, 24, 12, sdf: false);
        SphereIndexCount = Indices.Count - SphereFirstIndex;
    }

    int AddMaterial(Vector3 albedo, float roughness, float metallic, uint type = Material.Standard, uint flags = 0, uint tex = 0)
    {
        Materials.Add(new Material { Albedo = albedo, Roughness = roughness, Metallic = metallic, Type = type, Flags = flags, TexIndex = tex });
        return Materials.Count - 1;
    }

    SceneObject Begin(string name)
    {
        current = new SceneObject { Name = name, FirstIndex = Indices.Count };
        return current;
    }

    void End()
    {
        current.IndexCount = Indices.Count - current.FirstIndex;
        Objects.Add(current);
        current = null;
    }

    void AddMirror(string name, Vector3 center, float halfW, float halfH, Vector3 normal, uint texSlot)
    {
        var right = Vector3.Normalize(Vector3.Cross(Vector3.UnitY, normal));
        var up = Vector3.UnitY;

        // wooden frame, slightly larger, a few cm deep
        Begin(name + "Frame");
        AddBox(center + normal * 0.03f, new Vector3(MathF.Abs(right.X) * (halfW + 0.12f) + MathF.Abs(normal.X) * 0.03f,
                                                     halfH + 0.12f,
                                                     MathF.Abs(right.Z) * (halfW + 0.12f) + MathF.Abs(normal.Z) * 0.03f), 0, mFrame);
        End();

        int mat = AddMaterial(new Vector3(0.95f, 0.95f, 0.95f), 0.0f, 1, Material.Mirror, tex: texSlot);
        var c = center + normal * 0.065f;
        var obj = Begin(name);
        obj.MirrorIndex = Mirrors.Count;
        AddQuad(c - right * halfW - up * halfH, c + right * halfW - up * halfH, c + right * halfW + up * halfH, c - right * halfW + up * halfH, normal, mat);
        // SDF: a thin slab whose front face is the mirror quad
        SdfPrims.Add(new SdfPrim
        {
            Type = SdfPrim.Box, Center = c - normal * 0.004f, Material = (uint)mat, CosR = 1, SinR = 0,
            Size = new Vector3(MathF.Abs(right.X) * halfW + MathF.Abs(normal.X) * 0.004f, halfH, MathF.Abs(right.Z) * halfW + MathF.Abs(normal.Z) * 0.004f),
        });
        End();
        Mirrors.Add(new Mirror { Plane = new Plane(normal, -Vector3.Dot(normal, c)), ObjectIndex = Objects.Count - 1, MaterialIndex = mat });
    }

    void AddTriangle(uint a, uint b, uint c, Vector3 expectedNormal)
    {
        var pa = Vertices[(int)a].Pos; var pb = Vertices[(int)b].Pos; var pc = Vertices[(int)c].Pos;
        var n = Vector3.Cross(pb - pa, pc - pa);
        if (n.LengthSquared() < 1e-14f) return; // degenerate
        // D3D default: clockwise = front face. In a left-handed system that means cross(b-a, c-a) points at the viewer.
        if (Vector3.Dot(n, expectedNormal) >= 0) { Indices.Add(a); Indices.Add(b); Indices.Add(c); }
        else { Indices.Add(a); Indices.Add(c); Indices.Add(b); }
    }

    uint AddVertex(Vector3 p, Vector3 n, int mat)
    {
        Vertices.Add(new Vertex { Pos = p, Normal = n, Material = (uint)mat });
        if (current != null) current.Material = mat;
        return (uint)(Vertices.Count - 1);
    }

    void AddQuad(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, Vector3 n, int mat)
    {
        uint a = AddVertex(p0, n, mat), b = AddVertex(p1, n, mat), c = AddVertex(p2, n, mat), d = AddVertex(p3, n, mat);
        AddTriangle(a, b, c, n);
        AddTriangle(a, c, d, n);
    }

    void AddSdfPlane(Vector3 n, float d, int mat) =>
        SdfPrims.Add(new SdfPrim { Type = SdfPrim.Plane, Size = n, Center = new Vector3(d, 0, 0), Material = (uint)mat });

    void AddBox(Vector3 center, Vector3 half, float rotYDeg, int mat)
    {
        float rad = rotYDeg * MathF.PI / 180f;
        SdfPrims.Add(new SdfPrim { Type = SdfPrim.Box, Center = center, Size = half, Material = (uint)mat, CosR = MathF.Cos(rad), SinR = MathF.Sin(rad) });
        var rot = Matrix4x4.CreateRotationY(rad);
        Vector3 T(Vector3 v) => Vector3.Transform(v, rot) + center;
        Vector3 R(Vector3 v) => Vector3.TransformNormal(v, rot);
        var hx = half.X; var hy = half.Y; var hz = half.Z;
        // +X, -X, +Y, -Y, +Z, -Z
        AddQuad(T(new(hx, -hy, -hz)), T(new(hx, -hy, hz)), T(new(hx, hy, hz)), T(new(hx, hy, -hz)), R(Vector3.UnitX), mat);
        AddQuad(T(new(-hx, -hy, -hz)), T(new(-hx, -hy, hz)), T(new(-hx, hy, hz)), T(new(-hx, hy, -hz)), R(-Vector3.UnitX), mat);
        AddQuad(T(new(-hx, hy, -hz)), T(new(hx, hy, -hz)), T(new(hx, hy, hz)), T(new(-hx, hy, hz)), R(Vector3.UnitY), mat);
        AddQuad(T(new(-hx, -hy, -hz)), T(new(hx, -hy, -hz)), T(new(hx, -hy, hz)), T(new(-hx, -hy, hz)), R(-Vector3.UnitY), mat);
        AddQuad(T(new(-hx, -hy, hz)), T(new(hx, -hy, hz)), T(new(hx, hy, hz)), T(new(-hx, hy, hz)), R(Vector3.UnitZ), mat);
        AddQuad(T(new(-hx, -hy, -hz)), T(new(hx, -hy, -hz)), T(new(hx, hy, -hz)), T(new(-hx, hy, -hz)), R(-Vector3.UnitZ), mat);
    }

    void AddSphere(Vector3 center, float radius, int mat, int slices, int stacks, bool sdf = true)
    {
        if (sdf) SdfPrims.Add(new SdfPrim { Type = SdfPrim.Sphere, Center = center, Size = new Vector3(radius, 0, 0), Material = (uint)mat });
        uint baseV = (uint)Vertices.Count;
        for (int i = 0; i <= stacks; i++)
        {
            float theta = MathF.PI * i / stacks;
            for (int j = 0; j <= slices; j++)
            {
                float phi = 2 * MathF.PI * j / slices;
                var n = new Vector3(MathF.Sin(theta) * MathF.Cos(phi), MathF.Cos(theta), MathF.Sin(theta) * MathF.Sin(phi));
                AddVertex(center + n * radius, n, mat);
            }
        }
        for (int i = 0; i < stacks; i++)
        {
            for (int j = 0; j < slices; j++)
            {
                uint a = baseV + (uint)(i * (slices + 1) + j);
                uint b = a + 1;
                uint c = a + (uint)(slices + 1);
                uint d = c + 1;
                var mid = (Vertices[(int)a].Normal + Vertices[(int)d].Normal) * 0.5f;
                AddTriangle(a, c, b, mid);
                AddTriangle(b, c, d, mid);
            }
        }
    }
}
