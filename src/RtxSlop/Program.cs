using System.Diagnostics;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Drawing.Text;
using System.Numerics;

namespace RtxSlop;

static class Program
{
    [STAThread]
    static void Main(string[] args)
    {
        Application.SetHighDpiMode(HighDpiMode.PerMonitorV2);
        Application.EnableVisualStyles();
        try
        {
            using var form = new MainForm(args);
            form.Run();
        }
        catch (Exception ex)
        {
            File.WriteAllText(Path.Combine(AppContext.BaseDirectory, "error.log"), ex.ToString());
            if (!args.Contains("--capture"))
                MessageBox.Show(ex.ToString(), "RtxSlop - fatal error", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }
}

sealed class MainForm : Form
{
    readonly Renderer renderer;
    readonly Scene scene;
    readonly Camera camera = new();
    readonly RenderSettings settings = new();
    readonly HashSet<Keys> keys = new();
    readonly string shaderDir = Path.Combine(AppContext.BaseDirectory, "Shaders");
    readonly string captureDir;

    bool mouseCaptured;
    readonly string modelInfo;
    bool resizePending;
    bool showHud = true;
    float lightTime = 3f;
    string statusMessage = "";
    double statusUntil;

    // stats
    readonly Stopwatch clock = Stopwatch.StartNew();
    double hudTimer, fpsAccum;
    int fpsFrames;
    double fps, cpuMs;

    public MainForm(string[] args)
    {
        Text = "RtxSlop - rasterization / ray tracing / path tracing";
        ClientSize = new Size(1600, 900);
        StartPosition = FormStartPosition.CenterScreen;
        KeyPreview = true;
        SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.UserPaint | ControlStyles.Opaque, true);

        bool debug = args.Contains("--debug");
        int ci = Array.IndexOf(args, "--capture");
        if (ci >= 0) captureDir = ci + 1 < args.Length ? args[ci + 1] : "captures";

        // polygon model: --model <file.obj>, else data/dragon/dragon.obj found above the exe; --no-model disables it
        ObjMesh mesh = null;
        int mi = Array.IndexOf(args, "--model");
        string modelPath = mi >= 0 && mi + 1 < args.Length ? args[mi + 1] : FindDataFile(Path.Combine("data", "dragon", "dragon.obj"));
        if (!args.Contains("--no-model") && modelPath != null && File.Exists(modelPath))
        {
            var sw = Stopwatch.StartNew();
            mesh = ObjLoader.Load(modelPath);
            modelInfo = $"{Path.GetFileNameWithoutExtension(modelPath)} {mesh.Indices.Length / 3000}k tris ({sw.ElapsedMilliseconds} ms load)";
        }

        scene = new Scene(i => (uint)(Renderer.SlotRefl0 + i), Renderer.ProbeSlot, mesh);
        CreateHandle();
        renderer = new Renderer(Handle, ClientSize.Width, ClientSize.Height, scene, shaderDir, debug);
        Text = $"RtxSlop - {renderer.AdapterName}";
    }

    static string FindDataFile(string relative)
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir != null; dir = dir.Parent)
        {
            var candidate = Path.Combine(dir.FullName, relative);
            if (File.Exists(candidate)) return candidate;
        }
        return null;
    }

    void CycleMeshMaterial()
    {
        if (!scene.HasMesh) return;
        var m = scene.SetMeshPreset(scene.MeshPreset + 1);
        renderer.SetMaterial(scene.MeshMaterialIndex, m);
    }

    public void Run()
    {
        Show();
        if (captureDir != null)
        {
            RunCaptureScript();
            return;
        }

        double last = clock.Elapsed.TotalSeconds;
        while (!IsDisposed && Visible)
        {
            Application.DoEvents();
            if (IsDisposed || !Visible) break;
            if (WindowState == FormWindowState.Minimized) { Thread.Sleep(16); continue; }

            double now = clock.Elapsed.TotalSeconds;
            float dt = (float)Math.Min(now - last, 0.1);
            last = now;

            if (resizePending)
            {
                resizePending = false;
                renderer.Resize(ClientSize.Width, ClientSize.Height);
            }

            UpdateInput(dt);
            if (settings.LightsAnimating) lightTime += dt;
            scene.UpdateLights(lightTime);

            renderer.Render(camera, settings, lightTime);

            cpuMs = (clock.Elapsed.TotalSeconds - now) * 1000;
            fpsAccum += dt; fpsFrames++; hudTimer += dt;
            if (fpsAccum >= 0.5) { fps = fpsFrames / fpsAccum; fpsAccum = 0; fpsFrames = 0; }
            if (hudTimer >= 0.25) { hudTimer = 0; renderer.SetHud(DrawHud()); }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Input
    // ---------------------------------------------------------------------------------------------
    Point ScreenCenter => PointToScreen(new Point(ClientSize.Width / 2, ClientSize.Height / 2));

    void CaptureMouse(bool on)
    {
        if (on == mouseCaptured) return;
        mouseCaptured = on;
        if (on) { Cursor.Hide(); Cursor.Position = ScreenCenter; Cursor.Clip = RectangleToScreen(ClientRectangle); }
        else { Cursor.Show(); Cursor.Clip = Rectangle.Empty; }
    }

    protected override void OnMouseDown(MouseEventArgs e)
    {
        base.OnMouseDown(e);
        if (e.Button == MouseButtons.Left) CaptureMouse(true);
    }

    protected override void OnDeactivate(EventArgs e)
    {
        base.OnDeactivate(e);
        CaptureMouse(false);
        keys.Clear();
    }

    protected override void OnResize(EventArgs e)
    {
        base.OnResize(e);
        resizePending = true;
    }

    protected override void OnKeyUp(KeyEventArgs e) => keys.Remove(e.KeyCode);

    protected override void OnKeyDown(KeyEventArgs e)
    {
        keys.Add(e.KeyCode);
        e.Handled = true;
        e.SuppressKeyPress = true;
        switch (e.KeyCode)
        {
            case Keys.Escape:
                if (mouseCaptured) CaptureMouse(false); else Close();
                break;
            case Keys.D1: case Keys.NumPad1: settings.Mode = RenderMode.Raster; break;
            case Keys.D2: case Keys.NumPad2: settings.Mode = RenderMode.RayTrace; break;
            case Keys.D3: case Keys.NumPad3: settings.Mode = RenderMode.PathTrace; break;
            case Keys.D4: case Keys.NumPad4: settings.Mode = RenderMode.Hybrid; break;
            case Keys.D5: case Keys.NumPad5: settings.Mode = RenderMode.Sdf; break;
            case Keys.Tab: settings.Mode = (RenderMode)(((int)settings.Mode + 1) % 5); break;
            case Keys.L: settings.LightsAnimating = !settings.LightsAnimating; break;
            case Keys.G: CycleMeshMaterial(); break;
            case Keys.F: settings.Denoise = !settings.Denoise; break;
            case Keys.V: settings.VSync = !settings.VSync; break;
            case Keys.M:
                if (settings.Mode == RenderMode.Sdf) settings.SdfAntiAliasing = !settings.SdfAntiAliasing;
                else settings.AntiAliasing = !settings.AntiAliasing;
                break;
            case Keys.H: showHud = !showHud; renderer.SetHud(DrawHud()); break;
            case Keys.B:
                if (settings.Mode is RenderMode.PathTrace or RenderMode.Hybrid) settings.PtBounces = Math.Max(1, settings.PtBounces - 1);
                else settings.RtBounces = Math.Max(0, settings.RtBounces - 1);
                break;
            case Keys.N:
                if (settings.Mode is RenderMode.PathTrace or RenderMode.Hybrid) settings.PtBounces = Math.Min(16, settings.PtBounces + 1);
                else settings.RtBounces = Math.Min(32, settings.RtBounces + 1);
                break;
            case Keys.Oemplus: case Keys.Add: settings.Spp = Math.Min(64, settings.Spp * 2); break;
            case Keys.OemMinus: case Keys.Subtract: settings.Spp = Math.Max(1, settings.Spp / 2); break;
            case Keys.OemOpenBrackets: settings.Exposure = MathF.Max(0.05f, settings.Exposure / 1.25f); break;
            case Keys.OemCloseBrackets: settings.Exposure = MathF.Min(20f, settings.Exposure * 1.25f); break;
            case Keys.F5:
                try { renderer.ReloadShaders(shaderDir); Flash("Shaders reloaded"); }
                catch (Exception ex) { Flash("Shader error - see shader-error.log"); File.WriteAllText(Path.Combine(AppContext.BaseDirectory, "shader-error.log"), ex.Message); }
                break;
            case Keys.F12:
                var path = Path.Combine(AppContext.BaseDirectory, "screenshots", $"shot_{DateTime.Now:yyyyMMdd_HHmmss}_{settings.Mode}.png");
                renderer.RequestCapture(path);
                Flash("Saved " + Path.GetFileName(path));
                break;
        }
        if (IsDisposed) return;   // Esc -> Close() already tore down the fonts and the renderer
        renderer.SetHud(DrawHud());
    }

    void Flash(string msg)
    {
        statusMessage = msg;
        statusUntil = clock.Elapsed.TotalSeconds + 2.5;
    }

    void UpdateInput(float dt)
    {
        if (mouseCaptured)
        {
            var c = ScreenCenter;
            var p = Cursor.Position;
            float sens = 0.0025f;
            camera.Look((p.X - c.X) * sens, (p.Y - c.Y) * sens);
            if (p != c) Cursor.Position = c;
        }

        var move = Vector3.Zero;
        if (keys.Contains(Keys.W) || keys.Contains(Keys.Up)) move.Z += 1;
        if (keys.Contains(Keys.S) || keys.Contains(Keys.Down)) move.Z -= 1;
        if (keys.Contains(Keys.D) || keys.Contains(Keys.Right)) move.X += 1;
        if (keys.Contains(Keys.A) || keys.Contains(Keys.Left)) move.X -= 1;
        if (keys.Contains(Keys.Space) || keys.Contains(Keys.E)) move.Y += 1;
        if (keys.Contains(Keys.C) || keys.Contains(Keys.Q)) move.Y -= 1;
        if (move != Vector3.Zero)
        {
            float speed = keys.Contains(Keys.ShiftKey) ? 6f : 2.5f;
            camera.Move(Vector3.Normalize(move), dt, speed);
        }
    }

    // ---------------------------------------------------------------------------------------------
    // HUD (GDI+ text rendered into a small texture, composited by the tonemap pass)
    // ---------------------------------------------------------------------------------------------
    static readonly string[] ModeNames = { "1  RASTERIZATION", "2  RAY TRACING", "3  PATH TRACING", "4  HYBRID: RASTER + PATH TRACING", "5  SDF RAY MARCHING" };
    static readonly string[] ModeInfo =
    {
        "forward PBR | cube shadow maps + PCF | planar mirrors | cube probe",
        "DXR 1.1 inline rays | exact hard shadows | recursive reflections",
        "GGX + NEE + MIS | soft shadows | GI | temporal + a-trous denoise",
        "raster G-buffer = primary hits | path traced from bounce 1 on",
        "no triangles: sphere tracing | SDF soft shadows + AO | reflections",
    };

    readonly Font hudFontBig = new("Consolas", 15f, FontStyle.Bold, GraphicsUnit.Pixel);
    readonly Font hudFont = new("Consolas", 13f, FontStyle.Regular, GraphicsUnit.Pixel);

    byte[] DrawHud()
    {
        var (w, h) = Renderer.HudSize;
        using var bmp = new Bitmap(w, h, PixelFormat.Format32bppArgb);
        using (var g = Graphics.FromImage(bmp))
        {
            g.Clear(Color.Transparent);
            if (showHud)
            {
                g.SmoothingMode = SmoothingMode.AntiAlias;
                g.TextRenderingHint = TextRenderingHint.AntiAliasGridFit;
                using (var bg = new SolidBrush(Color.FromArgb(175, 12, 12, 16)))
                    g.FillRectangle(bg, 0, 0, w, h);

                int m = (int)settings.Mode;
                var accent = m switch { 0 => Color.FromArgb(120, 200, 255), 1 => Color.FromArgb(118, 185, 0), 3 => Color.FromArgb(230, 110, 255), 4 => Color.FromArgb(90, 230, 200), _ => Color.FromArgb(255, 170, 60) };
                using var accentBrush = new SolidBrush(accent);
                float y = 8;
                g.DrawString(ModeNames[m], hudFontBig, accentBrush, 10, y); y += 21;
                g.DrawString(ModeInfo[m], hudFont, Brushes.Gainsboro, 10, y); y += 18;

                string detail = settings.Mode switch
                {
                    RenderMode.Raster => $"18 shadow + 6 probe + 2 mirror passes   MSAA {(settings.AntiAliasing ? "4x" : "off")} (M)",
                    RenderMode.RayTrace => $"max bounces {settings.RtBounces} (B/N)   SSAA {(settings.AntiAliasing ? "4x" : "off")} (M)",
                    RenderMode.Sdf => $"max bounces {settings.RtBounces} (B/N)   SSAA {(settings.SdfAntiAliasing ? "4x (slow)" : "off")} (M)",
                    _ => $"bounces {settings.PtBounces} (B/N)   spp {settings.Spp} (+/-)   denoise {(settings.Denoise ? "ON" : "OFF")} (F)",
                };
                g.DrawString(detail, hudFont, Brushes.Gainsboro, 10, y); y += 18;
                g.DrawString($"{fps,6:F0} fps   cpu {cpuMs,5:F2} ms   gpu {renderer.GpuTimeMs,6:F2} ms   {renderer.Width}x{renderer.Height}", hudFont, Brushes.White, 10, y); y += 18;
                g.DrawString($"lights {(settings.LightsAnimating ? "moving" : "paused")} (L)   exposure {settings.Exposure:F2} ([ ])   vsync {(settings.VSync ? "on" : "off")} (V)", hudFont, Brushes.Gainsboro, 10, y); y += 18;
                if (scene.HasMesh)
                    g.DrawString($"model {modelInfo}   material {scene.MeshPresets[scene.MeshPreset].Name} (G)", hudFont, Brushes.Gainsboro, 10, y);
                y += 22;
                g.DrawString("click: mouse look   WASD move   Space/C up/down   Shift fast", hudFont, Brushes.DarkGray, 10, y); y += 18;
                g.DrawString("1-5 or Tab: mode   F5 reload shaders   F12 screenshot   H hide", hudFont, Brushes.DarkGray, 10, y); y += 18;
                if (clock.Elapsed.TotalSeconds < statusUntil)
                    g.DrawString(statusMessage, hudFont, Brushes.Yellow, 10, y);
            }
        }
        var data = bmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
        var bytes = new byte[w * h * 4];
        for (int row = 0; row < h; row++)
            System.Runtime.InteropServices.Marshal.Copy(data.Scan0 + row * data.Stride, bytes, row * w * 4, w * 4);
        bmp.UnlockBits(data);
        return bytes;
    }

    // ---------------------------------------------------------------------------------------------
    // --capture <dir>: render fixed viewpoints in every mode and save PNGs (used for testing)
    // ---------------------------------------------------------------------------------------------
    void RunCaptureScript()
    {
        var views = new (string Name, Vector3 Pos, float Yaw, float Pitch)[]
        {
            ("front", new Vector3(0, 1.7f, -4.6f), 0f, -0.05f),
            ("corner", new Vector3(5.0f, 2.6f, -5.0f), -0.75f, -0.28f),
            ("mirror", new Vector3(-1.2f, 1.6f, 1.5f), 0.35f, -0.05f),
        };
        settings.LightsAnimating = false;
        showHud = false;
        renderer.SetHud(DrawHud());
        scene.UpdateLights(lightTime);
        foreach (var v in views)
        {
            camera.Position = v.Pos; camera.Yaw = v.Yaw; camera.Pitch = v.Pitch;
            foreach (RenderMode mode in Enum.GetValues<RenderMode>())
            {
                settings.Mode = mode;
                int frames = mode is RenderMode.PathTrace or RenderMode.Hybrid ? 256 : 3;
                for (int i = 0; i < frames; i++)
                {
                    Application.DoEvents();
                    if (i == frames - 1) renderer.RequestCapture(Path.Combine(captureDir, $"{v.Name}_{(int)mode}_{mode}.png"));
                    renderer.Render(camera, settings, lightTime);
                }
            }
        }
        // the model up close in every material, in the main modes
        if (scene.HasMesh)
        {
            camera.Position = new Vector3(0, 1.5f, -3.4f); camera.Yaw = 0; camera.Pitch = -0.15f;
            for (int preset = 0; preset < scene.MeshPresets.Length; preset++)
            {
                if (preset > 0) CycleMeshMaterial();
                foreach (var mode in new[] { RenderMode.Raster, RenderMode.RayTrace, RenderMode.PathTrace, RenderMode.Sdf })
                {
                    settings.Mode = mode;
                    int frames = mode == RenderMode.PathTrace ? 256 : 3;
                    for (int i = 0; i < frames; i++)
                    {
                        Application.DoEvents();
                        if (i == frames - 1) renderer.RequestCapture(Path.Combine(captureDir, $"model_{scene.MeshPresets[scene.MeshPreset].Name}_{(int)mode}_{mode}.png"));
                        renderer.Render(camera, settings, lightTime);
                    }
                }
            }
            CycleMeshMaterial();   // back to the first preset
            while (scene.MeshPreset != 1) CycleMeshMaterial();
        }

        // PT with moving lights (short history) - what you normally see interactively
        settings.Mode = RenderMode.PathTrace;
        camera.Position = views[0].Pos; camera.Yaw = views[0].Yaw; camera.Pitch = views[0].Pitch;
        settings.LightsAnimating = true;
        float t0 = lightTime;
        for (int i = 0; i < 90; i++)
        {
            lightTime += 1f / 90;
            scene.UpdateLights(lightTime);
            if (i == 89) renderer.RequestCapture(Path.Combine(captureDir, "front_2_PathTrace_movingLights.png"));
            renderer.Render(camera, settings, lightTime);
        }
        settings.LightsAnimating = false;
        lightTime = t0;
        scene.UpdateLights(lightTime);

        // PT noise check: single frame without accumulation / denoiser
        settings.Denoise = false;
        renderer.Render(camera, settings, lightTime);   // mode switch resets history
        settings.Mode = RenderMode.Raster; renderer.Render(camera, settings, lightTime);
        settings.Mode = RenderMode.PathTrace;
        renderer.RequestCapture(Path.Combine(captureDir, "front_2_PathTrace_1spp_raw.png"));
        renderer.Render(camera, settings, lightTime);
        settings.Denoise = true;
        settings.Mode = RenderMode.Raster; renderer.Render(camera, settings, lightTime);
        settings.Mode = RenderMode.PathTrace;
        renderer.RequestCapture(Path.Combine(captureDir, "front_2_PathTrace_1spp_denoised.png"));
        renderer.Render(camera, settings, lightTime);

        // timing sample
        var sb = new System.Text.StringBuilder();
        foreach (RenderMode mode in Enum.GetValues<RenderMode>())
        {
            settings.Mode = mode;
            settings.LightsAnimating = true;
            for (int i = 0; i < 120; i++) { lightTime += 1f / 120; scene.UpdateLights(lightTime); renderer.Render(camera, settings, lightTime); }
            sb.AppendLine($"{mode}: gpu {renderer.GpuTimeMs:F2} ms @ {renderer.Width}x{renderer.Height}");
        }
        settings.Mode = RenderMode.Sdf;
        settings.SdfAntiAliasing = true;
        for (int i = 0; i < 120; i++) { lightTime += 1f / 120; scene.UpdateLights(lightTime); renderer.Render(camera, settings, lightTime); }
        sb.AppendLine($"Sdf (4x SSAA): gpu {renderer.GpuTimeMs:F2} ms");
        settings.SdfAntiAliasing = false;
        if (scene.HasMesh) sb.AppendLine($"model: {modelInfo}, material {scene.MeshPresets[scene.MeshPreset].Name}");
        File.WriteAllText(Path.Combine(captureDir, "timings.txt"), sb.ToString());

        // resize path: swap chain + every size dependent target is recreated
        ClientSize = new Size(1280, 720);
        Application.DoEvents();
        renderer.Resize(ClientSize.Width, ClientSize.Height);
        foreach (RenderMode mode in Enum.GetValues<RenderMode>())
        {
            settings.Mode = mode;
            for (int i = 0; i < 8; i++)
            {
                if (i == 7) renderer.RequestCapture(Path.Combine(captureDir, $"resized_{(int)mode}_{mode}.png"));
                renderer.Render(camera, settings, lightTime);
            }
        }
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            renderer?.Dispose();
            hudFont.Dispose();
            hudFontBig.Dispose();
        }
        base.Dispose(disposing);
    }
}
