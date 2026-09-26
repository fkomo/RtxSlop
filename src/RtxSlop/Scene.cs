using System.Numerics;
using System.Runtime.InteropServices;

namespace RtxSlop;

[StructLayout(LayoutKind.Sequential)]
public struct Vertex
{
    public Vector3 Pos;
    public Vector3 Normal;
    public uint Material;
    public uint Pad;
}

[StructLayout(LayoutKind.Sequential)]
public struct Material
{
    public Vector3 Albedo;
    public float Roughness;
    public Vector3 Emission;
    public float Metallic;
    public uint Type;
    public uint TexIndex;
    public uint Flags;
    public float Ior;        // glass only

    public const uint Standard = 0, Checker = 1, Mirror = 2, Glass = 3;
    public const uint FlagProbe = 1;
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
    public Vector3 Size;     // box: half extents, sphere: X = radius, plane: normal
    public uint Material;
    public float CosR, SinR; // box rotation around Y
    public Vector2 Pad;

    public const uint Plane = 0, Box = 1, Sphere = 2, Mesh = 3;   // Mesh: baked 3D distance texture, Size = half extents
}

public sealed class SceneObject
{
    public string Name;
    public int FirstIndex;
    public int IndexCount;
    public int MirrorIndex = -1;     // >= 0 if this object is a planar mirror
    public int ProbeCenter = -1;     // >= 0: reflection probe captured from inside this object (it is skipped there)
    public int Material = -1;        // material of the object's vertices
    public bool IsMesh;              // the loaded .obj model
}

public sealed class Mirror
{
    public Plane Plane;              // normal points into the room
    public int ObjectIndex;
    public int MaterialIndex;
}

/// <summary>
/// The demo room: 12 x 4 x 12 m, Cornell-box colored side walls, two facing mirrors,
/// a few obstacles, a chrome and a gold sphere, three animated sphere lights.
/// Coordinate system: left-handed, +Y up, +Z forward (D3D convention).
/// </summary>
public sealed class Scene
{
    public readonly List<Vertex> Vertices = new();
    public readonly List<uint> Indices = new();
    public readonly List<Material> Materials = new();
    public readonly List<SceneObject> Objects = new();
    public readonly List<Mirror> Mirrors = new();
    public readonly LightData[] Lights = new LightData[3];
    public readonly List<SdfPrim> SdfPrims = new();
    public int SdfPlaneCount;   // the same room as signed distance fields

    public int StaticIndexCount;
    public int GlassFirstIndex;      // [GlassFirstIndex, StaticIndexCount) = static glass objects
    public int SphereFirstIndex, SphereIndexCount;
    public readonly Vector3[] ProbePositions = new Vector3[2];   // 0 = chrome sphere, 1 = mesh

    // loaded .obj model (index range after the static geometry, own BLAS / instance)
    public bool HasMesh;
    public int MeshFirstIndex, MeshIndexCount, MeshMaterialIndex = -1;
    public Vector3 MeshMin, MeshMax;
    public int MeshPreset;
    public (string Name, Material Material)[] MeshPresets;

    public const float HalfX = 6f, HalfZ = 6f, Height = 4f;

    // Material indices
    int mFloor, mWhite, mRed, mGreen, mCeiling, mFrame, mStone, mBlue, mOrange, mChrome, mGold, mGlass;

    public Scene(Func<int, uint> mirrorTextureSlot, Func<int, uint> probeSlot, ObjMesh mesh)
    {
        mFloor = AddMaterial(new Vector3(0.85f, 0.85f, 0.82f), 0.08f, 0, Material.Checker);
        mWhite = AddMaterial(new Vector3(0.75f, 0.75f, 0.72f), 0.9f, 0);
        mRed = AddMaterial(new Vector3(0.70f, 0.10f, 0.08f), 0.9f, 0);
        mGreen = AddMaterial(new Vector3(0.12f, 0.48f, 0.14f), 0.9f, 0);
        mCeiling = AddMaterial(new Vector3(0.8f, 0.8f, 0.8f), 0.95f, 0);
        mFrame = AddMaterial(new Vector3(0.09f, 0.055f, 0.03f), 0.5f, 0);
        mStone = AddMaterial(new Vector3(0.6f, 0.58f, 0.55f), 0.8f, 0);
        mBlue = AddMaterial(new Vector3(0.08f, 0.2f, 0.65f), 0.35f, 0);
        mOrange = AddMaterial(new Vector3(0.9f, 0.4f, 0.05f), 0.5f, 0);
        mChrome = AddMaterial(new Vector3(0.95f, 0.95f, 0.95f), 0.02f, 1, flags: Material.FlagProbe, tex: probeSlot(0));
        mGold = AddMaterial(new Vector3(1.0f, 0.78f, 0.34f), 0.25f, 1, flags: Material.FlagProbe, tex: probeSlot(0));
        mGlass = AddMaterial(new Vector3(0.92f, 0.97f, 1.0f), 0, 0, Material.Glass, tex: probeSlot(0));
        Materials[mGlass] = Materials[mGlass] with { Ior = 1.5f };
        MeshPresets = new (string, Material)[]
        {
            ("jade", new Material { Albedo = new Vector3(0.18f, 0.55f, 0.36f), Roughness = 0.45f }),
            ("glass", new Material { Albedo = new Vector3(0.88f, 0.97f, 0.93f), Roughness = 0, Type = Material.Glass, Ior = 1.5f, TexIndex = probeSlot(1) }),
            ("gold", new Material { Albedo = new Vector3(1.0f, 0.78f, 0.34f), Roughness = 0.2f, Metallic = 1, Flags = Material.FlagProbe, TexIndex = probeSlot(1) }),
            ("mirror", new Material { Albedo = new Vector3(0.95f, 0.95f, 0.95f), Roughness = 0.01f, Metallic = 1, Flags = Material.FlagProbe, TexIndex = probeSlot(1) }),
        };

        // ---- room shell (inward facing) ----
        // SDF version of the shell: 6 inward-facing half spaces (dot(p, n) + d)
        AddSdfPlane(Vector3.UnitY, 0, mFloor);
        AddSdfPlane(-Vector3.UnitY, Height, mCeiling);
        AddSdfPlane(Vector3.UnitX, HalfX, mRed);
        AddSdfPlane(-Vector3.UnitX, HalfX, mGreen);
        AddSdfPlane(-Vector3.UnitZ, HalfZ, mWhite);
        AddSdfPlane(Vector3.UnitZ, HalfZ, mWhite);
        SdfPlaneCount = SdfPrims.Count;   // half-spaces must come first (the marcher treats them analytically)

        Begin("Floor");
        AddQuad(new(-HalfX, 0, -HalfZ), new(HalfX, 0, -HalfZ), new(HalfX, 0, HalfZ), new(-HalfX, 0, HalfZ), Vector3.UnitY, mFloor);
        End();
        Begin("Ceiling");
        AddQuad(new(-HalfX, Height, -HalfZ), new(HalfX, Height, -HalfZ), new(HalfX, Height, HalfZ), new(-HalfX, Height, HalfZ), -Vector3.UnitY, mCeiling);
        End();
        Begin("WallLeft");
        AddQuad(new(-HalfX, 0, -HalfZ), new(-HalfX, 0, HalfZ), new(-HalfX, Height, HalfZ), new(-HalfX, Height, -HalfZ), Vector3.UnitX, mRed);
        End();
        Begin("WallRight");
        AddQuad(new(HalfX, 0, -HalfZ), new(HalfX, 0, HalfZ), new(HalfX, Height, HalfZ), new(HalfX, Height, -HalfZ), -Vector3.UnitX, mGreen);
        End();
        Begin("WallBack");
        AddQuad(new(-HalfX, 0, HalfZ), new(HalfX, 0, HalfZ), new(HalfX, Height, HalfZ), new(-HalfX, Height, HalfZ), -Vector3.UnitZ, mWhite);
        End();
        Begin("WallFront");
        AddQuad(new(-HalfX, 0, -HalfZ), new(HalfX, 0, -HalfZ), new(HalfX, Height, -HalfZ), new(-HalfX, Height, -HalfZ), Vector3.UnitZ, mWhite);
        End();

        // ---- two mirrors facing each other ----
        AddMirror("MirrorBack", new Vector3(0, 1.9f, HalfZ), 2.6f, 1.5f, -Vector3.UnitZ, mirrorTextureSlot(0));
        AddMirror("MirrorFront", new Vector3(0, 1.75f, -HalfZ), 2.0f, 1.25f, Vector3.UnitZ, mirrorTextureSlot(1));

        // ---- obstacles ----
        Begin("Pillar"); AddBox(new(-3.5f, 1.4f, 3.2f), new(0.4f, 1.4f, 0.4f), 0, mStone); End();
        Begin("BoxBlue"); AddBox(new(3.0f, 0.6f, -2.5f), new(0.7f, 0.6f, 0.7f), 30, mBlue); End();
        Begin("BoxOrange"); AddBox(new(3.0f, 1.5f, -2.5f), new(0.3f, 0.3f, 0.3f), 15, mOrange); End();
        Begin("LowWall"); AddBox(new(-2.5f, 0.5f, -4.0f), new(1.25f, 0.5f, 0.15f), 0, mWhite); End();
        Begin("PillarRight"); AddBox(new(4.6f, 1.0f, 2.6f), new(0.25f, 1.0f, 0.25f), 45, mStone); End();

        ProbePositions[0] = new Vector3(1.5f, 0.9f, 3.3f);
        var chrome = Begin("ChromeSphere"); chrome.ProbeCenter = 0;
        AddSphere(ProbePositions[0], 0.9f, mChrome, 64, 32); End();
        Begin("GoldSphere"); AddSphere(new(-1.5f, 0.6f, -2.5f), 0.6f, mGold, 48, 24); End();

        // static glass goes last: it gets its own BLAS/instance so shadow rays can treat it separately
        GlassFirstIndex = Indices.Count;
        Begin("GlassSphere"); AddSphere(new(3.2f, 0.7f, 1.5f), 0.7f, mGlass, 64, 32); End();

        StaticIndexCount = Indices.Count;

        ProbePositions[1] = new Vector3(0, 1.0f, 0);
        if (mesh != null) AddMesh(mesh);

        // unit sphere used (instanced) for the light bulbs
        SphereFirstIndex = Indices.Count;
        AddSphere(Vector3.Zero, 1f, mWhite, 24, 12, sdf: false);
        SphereIndexCount = Indices.Count - SphereFirstIndex;

        UpdateLights(0);
    }

    /// <summary>
    /// Places the model in the middle of the room: centered on the XZ origin, standing on the floor,
    /// scaled so it fits inside the orbit of the low (blue) light and below the high ones.
    /// </summary>
    void AddMesh(ObjMesh mesh, float rotYDeg = 90)
    {
        // turn it so the camera sees it in profile
        var rot = Matrix4x4.CreateRotationY(rotYDeg * MathF.PI / 180f);
        var positions = mesh.Positions.Select(p => Vector3.Transform(p, rot)).ToArray();
        var normals = mesh.Normals.Select(n => Vector3.TransformNormal(n, rot)).ToArray();
        var bmin = positions.Aggregate(new Vector3(float.MaxValue), Vector3.Min);
        var bmax = positions.Aggregate(new Vector3(float.MinValue), Vector3.Max);

        var size = bmax - bmin;
        float scale = MathF.Min(3.0f / MathF.Max(size.X, size.Z), 2.0f / size.Y);
        var center = (bmin + bmax) * 0.5f;
        var offset = new Vector3(-center.X * scale, -bmin.Y * scale, -center.Z * scale);

        MeshPreset = 1;   // glass
        MeshMaterialIndex = Materials.Count;
        Materials.Add(MeshPresets[MeshPreset].Material);

        uint baseV = (uint)Vertices.Count;
        foreach (var (p, n) in positions.Zip(normals))
            Vertices.Add(new Vertex { Pos = p * scale + offset, Normal = n, Material = (uint)MeshMaterialIndex });

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
        ProbePositions[1] = new Vector3(0, (MeshMin.Y + MeshMax.Y) * 0.5f, 0);
        HasMesh = true;

        // SDF mode: a baked distance texture over the (padded) bounds
        var pad = new Vector3(0.1f);
        SdfPrims.Add(new SdfPrim
        {
            Type = SdfPrim.Mesh, Center = (MeshMin + MeshMax) * 0.5f, Size = (MeshMax - MeshMin) * 0.5f + pad,
            Material = (uint)MeshMaterialIndex, CosR = 1,
        });
    }

    public Material SetMeshPreset(int preset)
    {
        MeshPreset = ((preset % MeshPresets.Length) + MeshPresets.Length) % MeshPresets.Length;
        Materials[MeshMaterialIndex] = MeshPresets[MeshPreset].Material;
        return Materials[MeshMaterialIndex];
    }

    public bool IsGlass(SceneObject o) => o.Material >= 0 && Materials[o.Material].Type == Material.Glass;

    public void UpdateLights(float t)
    {
        Lights[0] = new LightData
        {
            Pos = new Vector3(3.8f * MathF.Cos(0.5f * t), 2.6f + 0.4f * MathF.Sin(1.3f * t), 3.8f * MathF.Sin(0.5f * t)),
            Radius = 0.12f, Color = new Vector3(1.0f, 0.72f, 0.42f), Intensity = 7f,
        };
        Lights[1] = new LightData
        {
            Pos = new Vector3(2.2f * MathF.Cos(-0.8f * t + 2f), 1.2f + 0.3f * MathF.Sin(2.1f * t), 2.2f * MathF.Sin(-0.8f * t + 2f)),
            Radius = 0.1f, Color = new Vector3(0.45f, 0.65f, 1.0f), Intensity = 4f,
        };
        Lights[2] = new LightData
        {
            Pos = new Vector3(4.0f * MathF.Sin(0.35f * t), 3.2f, 3.0f * MathF.Sin(0.7f * t)),
            Radius = 0.12f, Color = new Vector3(1.0f, 0.35f, 0.7f), Intensity = 4f,
        };
    }

    // ------------------------------------------------------------------------------------------
    int AddMaterial(Vector3 albedo, float roughness, float metallic, uint type = Material.Standard, uint flags = 0, uint tex = 0)
    {
        Materials.Add(new Material { Albedo = albedo, Roughness = roughness, Metallic = metallic, Type = type, Flags = flags, TexIndex = tex });
        return Materials.Count - 1;
    }

    SceneObject current;

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
