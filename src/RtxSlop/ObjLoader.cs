using System.Globalization;
using System.Numerics;

namespace RtxSlop;

/// <summary>Triangle mesh loaded from a Wavefront .obj, already converted to the demo's left-handed space.</summary>
public sealed class ObjMesh
{
    public Vector3[] Positions;
    public Vector3[] Normals;      // smooth, area weighted, one per position
    public int[] Indices;          // clockwise-front triangles (D3D convention)
    public Vector3 Min, Max;
}

/// <summary>
/// Minimal, fast .obj reader: positions + faces (any polygon, fan-triangulated; "v", "v/vt", "v//vn",
/// "v/vt/vn"; negative/relative indices). Materials, texture coordinates and groups are ignored.
/// Works directly on the UTF-8 bytes - the Stanford dragon (74 MB, 871k triangles) loads in about a second.
/// </summary>
public static class ObjLoader
{
    public static ObjMesh Load(string path)
    {
        var data = File.ReadAllBytes(path);
        var positions = new List<Vector3>(1 << 16);
        var fileNormals = new List<Vector3>(1 << 16);
        var indices = new List<int>(1 << 18);
        var triNormalRefs = new List<int>(1 << 16);   // file normal index of each triangle's first corner (-1 = none)
        Span<int> faceV = stackalloc int[64];
        Span<int> faceN = stackalloc int[64];

        int pos = 0;
        while (pos < data.Length)
        {
            int end = Array.IndexOf(data, (byte)'\n', pos);
            if (end < 0) end = data.Length;
            var line = data.AsSpan(pos, end - pos).TrimEnd((byte)'\r');
            pos = end + 1;
            line = line.TrimStart((byte)' ');
            if (line.Length < 2) continue;

            if (line[0] == 'v' && line[1] == ' ')
                positions.Add(ReadVector(line[2..]));
            else if (line[0] == 'v' && line[1] == 'n' && line.Length > 2 && line[2] == ' ')
                fileNormals.Add(ReadVector(line[3..]));
            else if (line[0] == 'f' && line[1] == ' ')
            {
                int count = 0;
                var rest = line[2..];
                while (count < faceV.Length)
                {
                    rest = rest.TrimStart((byte)' ');
                    if (rest.IsEmpty) break;
                    int sp = rest.IndexOf((byte)' ');
                    var token = sp < 0 ? rest : rest[..sp];
                    rest = sp < 0 ? default : rest[sp..];

                    // token: v | v/vt | v//vn | v/vt/vn
                    int s1 = token.IndexOf((byte)'/');
                    faceV[count] = Resolve(ParseInt(s1 < 0 ? token : token[..s1]), positions.Count);
                    faceN[count] = -1;
                    if (s1 >= 0)
                    {
                        var afterV = token[(s1 + 1)..];
                        int s2 = afterV.IndexOf((byte)'/');
                        if (s2 >= 0 && s2 + 1 < afterV.Length)
                            faceN[count] = Resolve(ParseInt(afterV[(s2 + 1)..]), fileNormals.Count);
                    }
                    count++;
                }
                for (int i = 1; i + 1 < count; i++)
                {
                    indices.Add(faceV[0]); indices.Add(faceV[i]); indices.Add(faceV[i + 1]);
                    triNormalRefs.Add(faceN[0]);
                }
            }
        }

        if (positions.Count == 0 || indices.Count == 0)
            throw new InvalidDataException($"{path}: no geometry found");

        // OBJ is right-handed: mirror Z into our left-handed space. Mirroring also turns the file's
        // counter-clockwise front faces into clockwise ones, which is what D3D and our BLAS expect.
        var P = positions.ToArray();
        for (int i = 0; i < P.Length; i++) P[i].Z = -P[i].Z;
        var I = indices.ToArray();

        // Some files wind the other way: vote with the file's normals (if present) and flip if needed.
        double vote = 0;
        for (int t = 0; t < triNormalRefs.Count; t++)
        {
            int ni = triNormalRefs[t];
            if (ni < 0 || ni >= fileNormals.Count) continue;
            var fn = fileNormals[ni];
            fn.Z = -fn.Z;
            var a = P[I[3 * t]]; var b = P[I[3 * t + 1]]; var c = P[I[3 * t + 2]];
            vote += Vector3.Dot(Vector3.Cross(b - a, c - a), fn) > 0 ? 1 : -1;
        }
        if (vote < 0)
            for (int t = 0; t < I.Length; t += 3) (I[t + 1], I[t + 2]) = (I[t + 2], I[t + 1]);

        // Smooth normals from the (now consistent) winding. cross(b-a, c-a) points outwards for
        // clockwise triangles in a left-handed system and is area weighted for free.
        var N = new Vector3[P.Length];
        for (int t = 0; t < I.Length; t += 3)
        {
            var a = P[I[t]]; var b = P[I[t + 1]]; var c = P[I[t + 2]];
            var n = Vector3.Cross(b - a, c - a);
            N[I[t]] += n; N[I[t + 1]] += n; N[I[t + 2]] += n;
        }
        var min = new Vector3(float.MaxValue);
        var max = new Vector3(float.MinValue);
        for (int i = 0; i < P.Length; i++)
        {
            float len = N[i].Length();
            N[i] = len > 0 ? N[i] / len : Vector3.UnitY;
            min = Vector3.Min(min, P[i]);
            max = Vector3.Max(max, P[i]);
        }
        return new ObjMesh { Positions = P, Normals = N, Indices = I, Min = min, Max = max };
    }

    static int Resolve(int index, int count) => index < 0 ? count + index : index - 1;

    static int ParseInt(ReadOnlySpan<byte> s)
    {
        int i = 0, sign = 1, v = 0;
        if (i < s.Length && s[i] == '-') { sign = -1; i++; }
        for (; i < s.Length && s[i] >= '0' && s[i] <= '9'; i++) v = v * 10 + (s[i] - '0');
        return v * sign;
    }

    static Vector3 ReadVector(ReadOnlySpan<byte> s)
    {
        Span<float> v = stackalloc float[3];
        for (int k = 0; k < 3; k++)
        {
            s = s.TrimStart((byte)' ');
            int sp = s.IndexOf((byte)' ');
            var tok = sp < 0 ? s : s[..sp];
            v[k] = float.Parse(tok, NumberStyles.Float, CultureInfo.InvariantCulture);
            s = sp < 0 ? default : s[sp..];
        }
        return new Vector3(v[0], v[1], v[2]);
    }
}
