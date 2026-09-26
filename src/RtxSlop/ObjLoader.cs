using System.Globalization;
using System.Numerics;

namespace RtxSlop;

/// <summary>Triangle mesh loaded from a Wavefront .obj, already converted to the demo's left-handed space.</summary>
public sealed class ObjMesh
{
    public Vector3[] Positions;    // one entry per unique (position, uv, normal) corner combination
    public Vector3[] Normals;
    public Vector2[] UVs;          // D3D convention (v flipped)
    public int[] Indices;          // clockwise-front triangles (D3D convention)
    public int[] TriangleMaterials;   // index into MaterialNames, -1 = none
    public List<string> MaterialNames = new();
    public string MtlLib;          // file name from "mtllib", relative to the .obj
    public Vector3 Min, Max;
}

/// <summary>The subset of a .mtl material the demo uses.</summary>
public sealed class ObjMaterial
{
    public string Name;
    public Vector3 Kd = Vector3.One;
    public Vector3 Ke;
    public float Ns = 10;
    public string MapKd;           // base color texture (relative to the .mtl)
    public string MapD;            // alpha mask texture
    public float D = 1;            // dissolve: < 1 = translucent (treated as glass)
}

/// <summary>
/// Fast .obj reader working directly on the UTF-8 bytes: "v", "vt", "vn", "usemtl", "mtllib" and faces
/// (any polygon, fan-triangulated; "v", "v/vt", "v//vn", "v/vt/vn"; negative/relative indices).
/// Corners are de-duplicated per (position, uv, normal). The Stanford dragon (74 MB, 871k triangles)
/// loads in well under a second.
/// </summary>
public static class ObjLoader
{
    /// <summary>
    /// Loads an .obj, using a binary cache next to it ("file.obj.rtxcache") when it is newer than the file.
    /// Parsing San Miguel (628 MB) takes a while; reading the cache is mostly disk speed.
    /// </summary>
    public static ObjMesh Load(string path)
    {
        string cache = path + ".rtxcache";
        var info = new FileInfo(path);
        try
        {
            if (File.Exists(cache) && TryReadCache(cache, info, out var cached)) return cached;
        }
        catch { /* corrupt / old cache: parse again */ }
        var mesh = Parse(path);
        try { WriteCache(cache, info, mesh); } catch { /* read-only location: fine */ }
        return mesh;
    }

    const int CacheVersion = 2;

    static void WriteCache(string cache, FileInfo source, ObjMesh m)
    {
        using var w = new BinaryWriter(File.Create(cache));
        w.Write("RTXOBJ"); w.Write(CacheVersion); w.Write(source.Length); w.Write(source.LastWriteTimeUtc.Ticks);
        void Arr<T>(T[] a) where T : unmanaged { w.Write(a.Length); w.Write(System.Runtime.InteropServices.MemoryMarshal.AsBytes(a.AsSpan())); }
        Arr(m.Positions); Arr(m.Normals); Arr(m.UVs); Arr(m.Indices); Arr(m.TriangleMaterials);
        w.Write(m.MaterialNames.Count);
        foreach (var n in m.MaterialNames) w.Write(n);
        w.Write(m.MtlLib ?? "");
        w.Write(m.Min.X); w.Write(m.Min.Y); w.Write(m.Min.Z); w.Write(m.Max.X); w.Write(m.Max.Y); w.Write(m.Max.Z);
    }

    static bool TryReadCache(string cache, FileInfo source, out ObjMesh m)
    {
        m = null;
        using var r = new BinaryReader(File.OpenRead(cache));
        if (r.ReadString() != "RTXOBJ" || r.ReadInt32() != CacheVersion || r.ReadInt64() != source.Length || r.ReadInt64() != source.LastWriteTimeUtc.Ticks)
            return false;
        T[] Arr<T>() where T : unmanaged
        {
            var a = new T[r.ReadInt32()];
            var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(a.AsSpan());
            if (r.Read(bytes) != bytes.Length) throw new EndOfStreamException();
            return a;
        }
        m = new ObjMesh { Positions = Arr<Vector3>(), Normals = Arr<Vector3>(), UVs = Arr<Vector2>(), Indices = Arr<int>(), TriangleMaterials = Arr<int>() };
        int names = r.ReadInt32();
        for (int i = 0; i < names; i++) m.MaterialNames.Add(r.ReadString());
        m.MtlLib = r.ReadString();
        if (m.MtlLib.Length == 0) m.MtlLib = null;
        m.Min = new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle());
        m.Max = new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle());
        return true;
    }

    /// <summary>True for PNGs whose color type carries an alpha channel (gray+alpha / RGBA).</summary>
    public static bool PngHasAlpha(string path)
    {
        if (path == null || !File.Exists(path)) return false;
        Span<byte> h = stackalloc byte[26];
        using var f = File.OpenRead(path);
        if (f.Read(h) < 26 || h[0] != 0x89 || h[1] != (byte)'P' || h[2] != (byte)'N' || h[3] != (byte)'G') return false;
        return h[25] == 4 || h[25] == 6;
    }

    static ObjMesh Parse(string path)
    {
        var data = File.ReadAllBytes(path);
        var positions = new List<Vector3>(1 << 16);
        var fileNormals = new List<Vector3>(1 << 16);
        var fileUVs = new List<Vector2>(1 << 16);
        var mesh = new ObjMesh();
        var materialIds = new Dictionary<string, int>();
        int currentMaterial = -1;

        var cornerIds = new Dictionary<long, int>(1 << 18);
        var cornerPos = new List<int>(1 << 18);
        var cornerUv = new List<int>(1 << 18);
        var cornerNrm = new List<int>(1 << 18);
        var indices = new List<int>(1 << 18);
        var triMaterials = new List<int>(1 << 16);
        Span<int> face = stackalloc int[64];

        int pos = 0;
        while (pos < data.Length)
        {
            int end = Array.IndexOf(data, (byte)'\n', pos);
            if (end < 0) end = data.Length;
            var line = data.AsSpan(pos, end - pos).TrimEnd((byte)'\r').TrimStart((byte)' ').TrimStart((byte)'\t');
            pos = end + 1;
            if (line.Length < 2) continue;

            if (line[0] == 'v' && line[1] == ' ')
                positions.Add(ReadVector(line[2..]));
            else if (line[0] == 'v' && line[1] == 'n' && line.Length > 2 && line[2] == ' ')
                fileNormals.Add(ReadVector(line[3..]));
            else if (line[0] == 'v' && line[1] == 't' && line.Length > 2 && line[2] == ' ')
            {
                var t = ReadVector(line[3..], 2);
                fileUVs.Add(new Vector2(t.X, 1 - t.Y));
            }
            else if (line[0] == 'f' && line[1] == ' ')
            {
                int count = 0;
                var rest = line[2..];
                while (count < face.Length)
                {
                    rest = rest.TrimStart((byte)' ');
                    if (rest.IsEmpty) break;
                    int sp = rest.IndexOf((byte)' ');
                    var token = sp < 0 ? rest : rest[..sp];
                    rest = sp < 0 ? default : rest[sp..];

                    // token: v | v/vt | v//vn | v/vt/vn
                    int vi, ti = -1, ni = -1;
                    int s1 = token.IndexOf((byte)'/');
                    vi = Resolve(ParseInt(s1 < 0 ? token : token[..s1]), positions.Count);
                    if (s1 >= 0)
                    {
                        var after = token[(s1 + 1)..];
                        int s2 = after.IndexOf((byte)'/');
                        var tPart = s2 < 0 ? after : after[..s2];
                        if (!tPart.IsEmpty) ti = Resolve(ParseInt(tPart), fileUVs.Count);
                        if (s2 >= 0 && s2 + 1 < after.Length) ni = Resolve(ParseInt(after[(s2 + 1)..]), fileNormals.Count);
                    }
                    long key = ((long)vi << 42) | ((long)(ti + 1) << 21) | (long)(ni + 1);
                    if (!cornerIds.TryGetValue(key, out int id))
                    {
                        id = cornerPos.Count;
                        cornerIds.Add(key, id);
                        cornerPos.Add(vi); cornerUv.Add(ti); cornerNrm.Add(ni);
                    }
                    face[count++] = id;
                }
                for (int i = 1; i + 1 < count; i++)
                {
                    indices.Add(face[0]); indices.Add(face[i]); indices.Add(face[i + 1]);
                    triMaterials.Add(currentMaterial);
                }
            }
            else if (line.StartsWith("usemtl "u8))
            {
                string name = System.Text.Encoding.UTF8.GetString(line[7..]).Trim();
                if (!materialIds.TryGetValue(name, out currentMaterial))
                {
                    currentMaterial = mesh.MaterialNames.Count;
                    materialIds.Add(name, currentMaterial);
                    mesh.MaterialNames.Add(name);
                }
            }
            else if (line.StartsWith("mtllib "u8))
                mesh.MtlLib = System.Text.Encoding.UTF8.GetString(line[7..]).Trim();
        }

        if (positions.Count == 0 || indices.Count == 0)
            throw new InvalidDataException($"{path}: no geometry found");

        // OBJ is right-handed: mirror Z into our left-handed space. Mirroring also turns the file's
        // counter-clockwise front faces into clockwise ones, which is what D3D and our BLAS expect.
        int n = cornerPos.Count;
        var P = new Vector3[n];
        var N = new Vector3[n];
        var UV = new Vector2[n];
        bool allNormals = true;
        for (int i = 0; i < n; i++)
        {
            var p = positions[cornerPos[i]];
            P[i] = new Vector3(p.X, p.Y, -p.Z);
            if (cornerUv[i] >= 0) UV[i] = fileUVs[cornerUv[i]];
            if (cornerNrm[i] >= 0) { var fn = fileNormals[cornerNrm[i]]; N[i] = new Vector3(fn.X, fn.Y, -fn.Z); }
            else allNormals = false;
        }
        var I = indices.ToArray();

        // Some files wind the other way: vote with the file's normals (if present) and flip if needed.
        double vote = 0;
        for (int t = 0; t < I.Length; t += 3)
        {
            if (cornerNrm[I[t]] < 0) continue;
            var a = P[I[t]]; var b = P[I[t + 1]]; var c = P[I[t + 2]];
            vote += Vector3.Dot(Vector3.Cross(b - a, c - a), N[I[t]]) > 0 ? 1 : -1;
        }
        if (vote < 0)
            for (int t = 0; t < I.Length; t += 3) (I[t + 1], I[t + 2]) = (I[t + 2], I[t + 1]);

        if (!allNormals)
            (P, N, UV, I) = CreaseNormals(P, UV, I, cornerPos, positions.Count, 45f);
        n = P.Length;

        var min = new Vector3(float.MaxValue);
        var max = new Vector3(float.MinValue);
        for (int i = 0; i < n; i++)
        {
            float len = N[i].Length();
            N[i] = len > 0 ? N[i] / len : Vector3.UnitY;
            min = Vector3.Min(min, P[i]);
            max = Vector3.Max(max, P[i]);
        }
        mesh.Positions = P;
        mesh.Normals = N;
        mesh.UVs = UV;
        mesh.Indices = I;
        mesh.TriangleMaterials = triMaterials.ToArray();
        mesh.Min = min;
        mesh.Max = max;
        return mesh;
    }

    public static List<ObjMaterial> LoadMtl(string path)
    {
        var list = new List<ObjMaterial>();
        ObjMaterial cur = null;
        foreach (var raw in File.ReadLines(path))
        {
            var line = raw.Trim();
            if (line.Length == 0 || line[0] == '#') continue;
            int sp = line.IndexOfAny(new[] { ' ', '\t' });
            string key = sp < 0 ? line : line[..sp];
            string val = sp < 0 ? "" : line[(sp + 1)..].Trim();
            if (key == "newmtl") list.Add(cur = new ObjMaterial { Name = val });
            else if (cur == null) continue;
            else if (key == "Kd") cur.Kd = ParseVec(val);
            else if (key == "Ke") cur.Ke = ParseVec(val);
            else if (key == "Ns") cur.Ns = float.Parse(val, CultureInfo.InvariantCulture);
            else if (key == "map_Kd") cur.MapKd = val.Replace('\\', Path.DirectorySeparatorChar);
            else if (key == "map_d") cur.MapD = val.Replace('\\', Path.DirectorySeparatorChar);
            else if (key == "d") cur.D = float.Parse(val, CultureInfo.InvariantCulture);
        }
        return list;

        static Vector3 ParseVec(string s)
        {
            var p = s.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            return new Vector3(float.Parse(p[0], CultureInfo.InvariantCulture), float.Parse(p[1], CultureInfo.InvariantCulture), float.Parse(p[2], CultureInfo.InvariantCulture));
        }
    }

    /// <summary>
    /// Normals for files without any: every triangle corner gets its own vertex whose normal averages the
    /// (area weighted) face normals around that position, but only of faces within <paramref name="creaseDeg"/>
    /// of its own face - curved surfaces come out smooth, the hard edges of architecture stay sharp.
    /// </summary>
    static (Vector3[] P, Vector3[] N, Vector2[] UV, int[] I) CreaseNormals(Vector3[] P, Vector2[] UV, int[] I, List<int> cornerPos, int positionCount, float creaseDeg)
    {
        int triCount = I.Length / 3;
        var face = new Vector3[triCount];
        for (int t = 0; t < triCount; t++)
            face[t] = Vector3.Cross(P[I[3 * t + 1]] - P[I[3 * t]], P[I[3 * t + 2]] - P[I[3 * t]]);

        // position -> triangles (CSR)
        var start = new int[positionCount + 1];
        for (int c = 0; c < I.Length; c++) start[cornerPos[I[c]] + 1]++;
        for (int i = 0; i < positionCount; i++) start[i + 1] += start[i];
        var fill = (int[])start.Clone();
        var tris = new int[I.Length];
        for (int c = 0; c < I.Length; c++) tris[fill[cornerPos[I[c]]]++] = c / 3;

        float cosCrease = MathF.Cos(creaseDeg * MathF.PI / 180f);
        var nP = new Vector3[I.Length];
        var nN = new Vector3[I.Length];
        var nUV = new Vector2[I.Length];
        var nI = new int[I.Length];
        for (int c = 0; c < I.Length; c++)
        {
            int t = c / 3;
            var own = face[t].LengthSquared() > 0 ? Vector3.Normalize(face[t]) : Vector3.UnitY;
            var sum = Vector3.Zero;
            int pi = cornerPos[I[c]];
            for (int k = start[pi]; k < start[pi + 1]; k++)
            {
                var f = face[tris[k]];
                float len = f.Length();
                if (len > 0 && Vector3.Dot(f / len, own) >= cosCrease) sum += f;
            }
            nP[c] = P[I[c]];
            nUV[c] = UV[I[c]];
            nN[c] = sum.LengthSquared() > 0 ? Vector3.Normalize(sum) : own;
            nI[c] = c;
        }
        return (nP, nN, nUV, nI);
    }

    static int Resolve(int index, int count) => index < 0 ? count + index : index - 1;

    static int ParseInt(ReadOnlySpan<byte> s)
    {
        int i = 0, sign = 1, v = 0;
        if (i < s.Length && s[i] == '-') { sign = -1; i++; }
        for (; i < s.Length && s[i] >= '0' && s[i] <= '9'; i++) v = v * 10 + (s[i] - '0');
        return v * sign;
    }

    static Vector3 ReadVector(ReadOnlySpan<byte> s, int components = 3)
    {
        Span<float> v = stackalloc float[3];
        for (int k = 0; k < components; k++)
        {
            s = s.TrimStart((byte)' ');
            int sp = s.IndexOf((byte)' ');
            var tok = sp < 0 ? s : s[..sp];
            v[k] = tok.IsEmpty ? 0 : float.Parse(tok, NumberStyles.Float, CultureInfo.InvariantCulture);
            s = sp < 0 ? default : s[sp..];
        }
        return new Vector3(v[0], v[1], v[2]);
    }
}
