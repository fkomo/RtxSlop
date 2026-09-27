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
            if (!args.Contains("--capture") && !args.Contains("--bench"))
                MessageBox.Show(ex.ToString(), "RtxSlop - fatal error", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }
}

sealed class MainForm : Form
{
    readonly Renderer renderer;
    Scene scene;                          // the one being rendered
    readonly Scene roomScene;
    readonly Scene[] lazyScenes = new Scene[5];   // 1 = Crytek Sponza (F2), 2 = Dabrovic Sponza (F3), 3/4 = San Miguel low-poly / full (F4/F5), loaded on first use

    // courtyard: the model stands in the free strip between the fountain and the trees (the .obj's z is mirrored),
    // the camera looks across the café tables from the west door, the lights orbit over the whole courtyard
    static readonly Scene.SiteLayout SanMiguelLayout = new(1.0f, new Vector3(15.75f, 0, -7.1f), new Vector3(7.0f, 2.4f, -7.0f),
        MathF.PI / 2, -0.12f, 0, new Vector3(1.0f, 1.0f, 3.0f), 512);
    readonly ObjMesh modelMesh;
    readonly Dictionary<Scene, (Vector3 Pos, float Yaw, float Pitch)> savedCameras = new();
    readonly Camera camera = new();
    readonly RenderSettings settings = new();
    readonly HashSet<Keys> keys = new();
    readonly string shaderDir = Path.Combine(AppContext.BaseDirectory, "Shaders");
    readonly string captureDir, benchDir;
    HashSet<int> captureScenes = new() { 0, 1, 2, 3, 4 };   // --scenes 1,3 limits the capture script
    readonly bool restirCapture;                                // --restir: only the many lights / ReSTIR DI comparison

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
        restirCapture = args.Contains("--restir");
        int ci = Array.IndexOf(args, "--capture");
        if (ci >= 0) captureDir = ci + 1 < args.Length ? args[ci + 1] : "captures";
        int bi = Array.IndexOf(args, "--bench");
        if (bi >= 0) benchDir = bi + 1 < args.Length ? args[bi + 1] : ".";
        int si = Array.IndexOf(args, "--scenes");
        if (si >= 0 && si + 1 < args.Length) captureScenes = args[si + 1].Split(',').Select(int.Parse).ToHashSet();

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

        modelMesh = mesh;
        roomScene = scene = Scene.CreateRoom(i => (uint)(Renderer.SlotRefl0 + i), Renderer.ProbeSlot, mesh);
        ResetCamera();
        CreateHandle();
        renderer = new Renderer(Handle, ClientSize.Width, ClientSize.Height, scene, shaderDir, debug);
        Text = $"RtxSlop - {renderer.AdapterName}";
        if (args.Contains("--sponza")) SwitchScene(1);
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

    void ResetCamera()
    {
        camera.Position = scene.CameraPos;
        camera.Yaw = scene.CameraYaw;
        camera.Pitch = scene.CameraPitch;
    }

    /// <summary>0 = the room, 1 = Crytek Sponza, 2 = Dabrovic Sponza (loaded, uploaded and baked the first time).</summary>
    void SwitchScene(int which)
    {
        Scene target;
        if (which == 0) target = roomScene;
        else
        {
            if (lazyScenes[which] == null)
            {
                string file = which switch { 1 => "sponza2/sponza.obj", 2 => "sponza/sponza.obj", 3 => "san_miguel/san-miguel-low-poly.obj", _ => "san_miguel/san-miguel.obj" };
                string path = FindDataFile(Path.Combine("data", file));
                if (path == null) { Flash($"data/{file} not found"); return; }
                Cursor.Current = Cursors.WaitCursor;
                lazyScenes[which] = which switch
                {
                    1 => Scene.CreateSponza(path, Renderer.ProbeSlot, modelMesh),
                    2 => Scene.CreateDabrovicSponza(path, Renderer.ProbeSlot, modelMesh),
                    3 => Scene.CreateSanMiguel("san miguel", path, Renderer.ProbeSlot, modelMesh, SanMiguelLayout),
                    _ => Scene.CreateSanMiguel("san miguel (full)", path, Renderer.ProbeSlot, modelMesh, SanMiguelLayout),
                };
            }
            target = lazyScenes[which];
        }
        if (target == scene) return;
        savedCameras[scene] = (camera.Position, camera.Yaw, camera.Pitch);
        scene = target;
        scene.UpdateLights(lightTime);
        renderer.SetScene(scene);
        if (savedCameras.TryGetValue(scene, out var c)) { camera.Position = c.Pos; camera.Yaw = c.Yaw; camera.Pitch = c.Pitch; }
        else ResetCamera();
        Cursor.Current = Cursors.Default;
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
        if (benchDir != null)
        {
            showHud = false;
            renderer.SetHud(DrawHud());
            Directory.CreateDirectory(benchDir);
            string text = MeasureTimings(allPresets: true);
            SwitchScene(1);
            text += MeasureTimings(allPresets: true);
            SwitchScene(2);
            text += MeasureTimings(allPresets: true);
            SwitchScene(3);
            text += MeasureTimings(allPresets: true);
            SwitchScene(4);
            text += MeasureTimings(allPresets: true);
            File.WriteAllText(Path.Combine(benchDir, "bench.txt"), text);
            return;
        }
        if (captureDir != null)
        {
            if (restirCapture) RunRestirCapture();
            else RunCaptureScript();
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
            case Keys.K: settings.LightsOn = !settings.LightsOn; break;
            case Keys.J: settings.ManyLights = !settings.ManyLights; break;
            case Keys.T: settings.Restir = !settings.Restir; break;
            case Keys.F1: SwitchScene(0); break;
            case Keys.F2: SwitchScene(1); break;
            case Keys.F3: SwitchScene(2); break;
            case Keys.F4: SwitchScene(3); break;
            case Keys.F5: SwitchScene(4); break;
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
            case Keys.R:
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
            camera.Move(Vector3.Normalize(move), dt, speed, scene.BoundsMin, scene.BoundsMax);
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
                g.DrawString($"scene {scene.Name} (F1-F5)   lights {(settings.LightsOn ? "on" : "OFF")} (K) {(settings.LightsAnimating ? "moving" : "paused")} (L)", hudFont, Brushes.Gainsboro, 10, y); y += 18;
                g.DrawString($"many lights {(settings.ManyLights ? $"ON ({scene.Lights.Length})" : "off")} (J)   ReSTIR DI {(settings.Restir ? "on" : "off")} (T, path tracing)", hudFont, Brushes.Gainsboro, 10, y); y += 18;
                g.DrawString($"exposure {settings.Exposure:F2} ([ ])   vsync {(settings.VSync ? "on" : "off")} (V)", hudFont, Brushes.Gainsboro, 10, y); y += 18;
                if (scene.HasMesh)
                    g.DrawString($"model {modelInfo}   material {scene.MeshPresets[scene.MeshPreset].Name} (G)", hudFont, Brushes.Gainsboro, 10, y);
                y += 22;
                g.DrawString("click: mouse look   WASD move   Space/C up/down   Shift fast", hudFont, Brushes.DarkGray, 10, y); y += 18;
                g.DrawString("1-5 or Tab: mode   R reload shaders   F12 screenshot   H hide", hudFont, Brushes.DarkGray, 10, y); y += 18;
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
    /// <summary>GPU time of every mode from the scene's start view with moving lights (optionally for every model material).</summary>
    string MeasureTimings(bool allPresets)
    {
        ResetCamera();
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"== scene {scene.Name}: {renderer.AdapterName}, {renderer.Width}x{renderer.Height}" + (scene.HasMesh ? $", model: {modelInfo}" : ""));
        int presets = allPresets && scene.HasMesh ? scene.MeshPresets.Length : 1;
        for (int p = 0; p < presets; p++)
        {
            if (p > 0) CycleMeshMaterial();
            if (scene.HasMesh) sb.AppendLine($"-- model material: {scene.MeshPresets[scene.MeshPreset].Name}");
            foreach (RenderMode mode in Enum.GetValues<RenderMode>())
            {
                settings.Mode = mode;
                settings.LightsAnimating = true;
                for (int i = 0; i < 120; i++) { lightTime += 1f / 120; scene.UpdateLights(lightTime); renderer.Render(camera, settings, lightTime); }
                sb.AppendLine($"{mode,-10} {renderer.GpuTimeMs,7:F2} ms");
            }
        }
        if (presets > 1) CycleMeshMaterial();   // back to where we started
        return sb.ToString();
    }

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
        string timings = "";
        if (captureScenes.Contains(0)) timings = CaptureRoom(views);

        // ---- scenes 2 and 3: the two Sponzas ----
        foreach (var (index, prefix) in new[] { (1, "sponza"), (2, "dabrovic"), (3, "miguel"), (4, "miguelfull") })
        {
            if (!captureScenes.Contains(index)) continue;
            var sw = Stopwatch.StartNew();
            SwitchScene(index);
            timings += $"{scene.Name} load + upload + bakes: {sw.ElapsedMilliseconds} ms\n";
            CaptureSponza(prefix);
            timings += MeasureTimings(allPresets: false);
        }
        SwitchScene(0);
        File.WriteAllText(Path.Combine(captureDir, "timings.txt"), timings);
        if (!captureScenes.Contains(0)) return;

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

    string CaptureRoom((string Name, Vector3 Pos, float Yaw, float Pitch)[] views)
    {
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

        return MeasureTimings(allPresets: false);
    }

    /// <summary>--capture dir --restir: many lights with / without ReSTIR DI (raw 1 spp, moving lights, converged) + timings.</summary>
    void RunRestirCapture()
    {
        showHud = false;
        renderer.SetHud(DrawHud());
        var timings = new System.Text.StringBuilder();
        foreach (var (index, prefix) in new[] { (0, "room"), (1, "sponza"), (2, "dabrovic"), (3, "miguel"), (4, "miguelfull") })
        {
            if (!captureScenes.Contains(index)) continue;
            SwitchScene(index);
            ResetCamera();
            settings.ManyLights = true;
            timings.AppendLine($"== {scene.Name}, {scene.Lights.Length} lights, {renderer.Width}x{renderer.Height}");
            void Frames(int n, string file, bool animate)
            {
                for (int i = 0; i < n; i++)
                {
                    Application.DoEvents();
                    if (animate) { lightTime += 1f / 60; scene.UpdateLights(lightTime); }
                    if (i == n - 1 && file != null) renderer.RequestCapture(Path.Combine(captureDir, file + ".png"));
                    renderer.Render(camera, settings, lightTime);
                }
            }
            settings.Mode = RenderMode.Raster;
            settings.LightsAnimating = false;
            scene.UpdateLights(lightTime);
            Frames(3, $"{prefix}_many_0_Raster", false);
            foreach (var mode in new[] { RenderMode.PathTrace, RenderMode.Hybrid })
            {
                foreach (bool restir in new[] { false, true })
                {
                    settings.Restir = restir;
                    string tag = $"{prefix}_many_{(int)mode}_{mode}_{(restir ? "restir" : "nee")}";
                    // raw: first frame after a reset, no denoiser (history needs a few frames: 8 still-light frames first for ReSTIR)
                    settings.Denoise = false;
                    settings.Mode = RenderMode.Raster; Frames(1, null, false);
                    settings.Mode = mode;
                    Frames(restir ? 8 : 1, null, false);
                    settings.Mode = RenderMode.Raster; Frames(1, null, false);
                    settings.Mode = mode;
                    Frames(1, tag + "_raw", false);
                    // moving lights, denoised (what you see when playing)
                    settings.Denoise = true;
                    settings.LightsAnimating = true;
                    float t0 = lightTime;
                    Frames(60, tag + "_moving", true);
                    timings.AppendLine($"{mode,-10} {(restir ? "ReSTIR DI" : "NEE (1 random light)"),-22} {renderer.GpuTimeMs,7:F2} ms");
                    lightTime = t0;
                    settings.LightsAnimating = false;
                    scene.UpdateLights(lightTime);
                    if (mode == RenderMode.PathTrace) Frames(256, tag + "_converged", false);
                }
            }
            settings.Restir = true;
            settings.ManyLights = false;
        }
        File.WriteAllText(Path.Combine(captureDir, "restir-timings.txt"), timings.ToString());
    }

    Vector3 ModelCenter() => (scene.MeshMin + scene.MeshMax) * 0.5f;

    void CaptureSponza(string prefix)
    {
        settings.LightsAnimating = false;
        scene.UpdateLights(lightTime);
        var sponzaViews = new (string Name, Vector3 Pos, float Yaw, float Pitch)[]
        {
            ($"{prefix}_atrium", scene.CameraPos, scene.CameraYaw, scene.CameraPitch),
            ($"{prefix}_model", ModelCenter() + new Vector3(-5.5f, 0.5f, 0), MathF.PI / 2, -0.08f),
            ($"{prefix}_high", ModelCenter() + new Vector3(-7.5f, 7.5f, 0), MathF.PI / 2, -0.55f),
        };
        // top-down overview (placement check)
        camera.Position = ((scene.BoundsMin + scene.BoundsMax) * 0.5f) with { Y = scene.BoundsMax.Y + 25 };
        camera.Yaw = 0; camera.Pitch = -1.5f;
        settings.Mode = RenderMode.Raster;
        for (int i = 0; i < 3; i++)
        {
            if (i == 2) renderer.RequestCapture(Path.Combine(captureDir, $"{prefix}_top_0_Raster.png"));
            renderer.Render(camera, settings, lightTime);
        }
        foreach (var v in sponzaViews)
        {
            camera.Position = v.Pos; camera.Yaw = v.Yaw; camera.Pitch = v.Pitch;
            foreach (bool lights in new[] { true, false })
            {
                if (!lights && !v.Name.EndsWith("_atrium")) continue;
                settings.LightsOn = lights;
                foreach (RenderMode mode in Enum.GetValues<RenderMode>())
                {
                    settings.Mode = mode;
                    int frames = mode is RenderMode.PathTrace or RenderMode.Hybrid ? 256 : 3;
                    for (int i = 0; i < frames; i++)
                    {
                        Application.DoEvents();
                        if (i == frames - 1) renderer.RequestCapture(Path.Combine(captureDir, $"{v.Name}{(lights ? "" : "_lightsoff")}_{(int)mode}_{mode}.png"));
                        renderer.Render(camera, settings, lightTime);
                    }
                }
            }
        }
        settings.LightsOn = true;
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
