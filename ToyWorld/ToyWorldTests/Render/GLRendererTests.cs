using System;
using System.Drawing;
using System.IO;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using Game;
using GoodAI.ToyWorld.Control;
using OpenTK.Graphics.OpenGL;
using OpenTK.Input;
using Render.Renderer;
using Render.RenderRequests;
using Render.RenderRequests.RenderRequests;
using Render.Tests.RRs;
using TmxMapSerializer.Elements;
using TmxMapSerializer.Serializer;
using ToyWorldTests.Attributes;
using ToyWorldTests.Game;
using VRageMath;
using World.ToyWorldCore;
using Xunit;

namespace ToyWorldTests.Render
{
    [Collection("Renderer")]
    public class GLRendererTests : IDisposable
    {
        private readonly ToyWorld m_world;
        private readonly GLRenderer m_renderer;


        public GLRendererTests()
        {
            using (var tmxStreamReader = new StreamReader(FileStreams.GetTmxMemoryStream()))
            {
                var serializer = new TmxSerializer();
                Map map = serializer.Deserialize(tmxStreamReader);

                using (var tilesetTableStreamReader = new StreamReader(FileStreams.GetTilesetTableMemoryStream()))
                    m_world = new ToyWorld(map, tilesetTableStreamReader);
            }

            m_renderer = new GLRenderer();
            m_renderer.Init();
            m_renderer.CreateWindow("TestGameWindow", 1024, 1024);
            m_renderer.CreateContext();
        }

        public void Dispose()
        {
            m_renderer.Dispose();
        }


        [RunnableInDebugOnly]
        public void ShowRRLongRunning()
        {
            m_renderer.MakeContextCurrent();

            var rr = RenderRequestFactory.CreateRenderRequest<IFullMapRR>();
            //var rr = RenderRequestFactory.CreateRenderRequest<IFovAvatarRR>(0);
            (rr as RenderRequest).Init(m_renderer, m_world);
            m_renderer.EnqueueRequest(rr);

            CancellationToken token = SetupWindow(
                delta =>
                {
                    delta += new SizeF(rr.Rotation);
                    float rot = delta.X;
                    MathHelper.LimitRadians2PI(ref rot);
                    delta.X = rot;

                    rot = delta.Y;
                    MathHelper.LimitRadiansPI(ref rot);
                    delta.Y = rot;

                    rr.Rotation = delta;
                });

            while (m_renderer.Window.Exists && !token.IsCancellationRequested)
            {
                try
                {
                    Task.Delay(20, token).Wait(token);
                }
                catch (OperationCanceledException)
                {
                    break;
                }

                m_renderer.Context.SwapBuffers();
                m_renderer.ProcessRequests(m_world);
            }

            Assert.True(token.IsCancellationRequested);
        }

        private CancellationToken SetupWindow(Action<PointF> onDrag)
        {
            CancellationTokenSource tokenSource = new CancellationTokenSource(new TimeSpan(1, 0, 0));

            m_renderer.Window.KeyDown += (sender, args) =>
            {
                if (args.Key == Key.A)
                    tokenSource.Cancel();
            };

            m_renderer.Window.MouseDown += (sender, args) =>
            {
                if (args.Button == MouseButton.Right)
                    tokenSource.Cancel();
            };

            m_renderer.Window.MouseMove += (sender, args) =>
            {
                const float factor = 1 / 100f;

                if (args.Mouse.IsButtonDown(MouseButton.Left))
                    onDrag(new PointF(args.XDelta * factor, args.YDelta * factor));
            };

            m_renderer.Window.Visible = true;

            return tokenSource.Token;
        }


        [Fact]
        public void InitRepeated()
        {
            m_renderer.CreateWindow("TestGameWindow", 1024, 1024);
            m_renderer.CreateContext();

            m_renderer.Init();
            m_renderer.CreateWindow("TestGameWindow", 1024, 1024);
            m_renderer.CreateContext();
        }

        [Fact]
        public void Resize()
        {
            // TODO: Doesn't work -- how to invoke the Resize event on Window?
            //m_renderer.Window.Size = new System.Drawing.Size((int)(m_renderer.Window.Width * 1.3f), (int)(m_renderer.Window.Height * 1.3f));
            m_renderer.ProcessRequests(m_world);
        }
    }
}
