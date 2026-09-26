using System.Numerics;

namespace RtxSlop;

/// <summary>First-person camera (left-handed, +Y up). Yaw 0 looks down +Z.</summary>
public sealed class Camera
{
    public Vector3 Position = new(0, 1.7f, -4.6f);
    public float Yaw;
    public float Pitch;
    public float FovY = 70f * MathF.PI / 180f;

    public Vector3 Forward => new(MathF.Sin(Yaw) * MathF.Cos(Pitch), MathF.Sin(Pitch), MathF.Cos(Yaw) * MathF.Cos(Pitch));
    public Vector3 Right => new(MathF.Cos(Yaw), 0, -MathF.Sin(Yaw));
    public Matrix4x4 View => Matrix4x4.CreateLookAtLeftHanded(Position, Position + Forward, Vector3.UnitY);

    public void Look(float dx, float dy)
    {
        Yaw += dx;
        Pitch = Math.Clamp(Pitch - dy, -1.5f, 1.5f);
    }

    /// <summary>Walk on the XZ plane (move.X = strafe, move.Y = up/down, move.Z = forward).</summary>
    public void Move(Vector3 move, float dt, float speed, Vector3 boundsMin, Vector3 boundsMax)
    {
        var flatFwd = Vector3.Normalize(new Vector3(MathF.Sin(Yaw), 0, MathF.Cos(Yaw)));
        var delta = (Right * move.X + Vector3.UnitY * move.Y + flatFwd * move.Z) * speed * dt;
        Position += delta;
        // stay inside the scene
        Position = Vector3.Clamp(Position, boundsMin, boundsMax);
    }
}
