using System;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Game;
using GoodAI.ToyWorld.Control;
using OpenTK.Input;
using Render.Renderer;
using Render.RenderRequests;
using TmxMapSerializer.Elements;
using TmxMapSerializer.Serializer;
using VRageMath;
using World.ToyWorldCore;
using Xunit;

namespace ToyWorldTests.Render
{
    [Collection("Renderer")]
    public class GLRendererTestBase : IDisposable
    {
        private readonly GameControllerBase m_gameController;

        protected ToyWorld World { get { return m_gameController.World; } }
        protected GLRenderer Renderer { get { return (GLRenderer)m_gameController.Renderer; } }


        public GLRendererTestBase()
        {
            using (var tmxStreamReader = new StreamReader(FileStreams.FullTmxFileStream()))
            using (var tilesetTableStreamReader = new StreamReader(FileStreams.GetTilesetTableMemoryStream()))
            {
                var gameSetup = new GameSetup(tmxStreamReader, tilesetTableStreamReader);
                m_gameController = ControllerFactory.GetController(gameSetup);
                m_gameController.Init();
            }
        }

        public void Dispose()
        {
            m_gameController.Dispose();
        }


        protected void RunRRLongRunning()
        {
            Renderer.MakeContextCurrent();

            int aID = m_gameController.GetAvatarIds().First();
            var rr = m_gameController.RegisterRenderRequest<IFreeMapRR>();
            var ac = m_gameController.GetAvatarController(aID);
            var controls = new AvatarControls(5) { DesiredSpeed = 0.1f };

            CancellationToken token = SetupWindow(
                delta =>
                {
                    //rr.PositionCenter = new PointF(rr.PositionCenter.X - delta.X, rr.PositionCenter.Y + delta.Y);
                    controls.DesiredRotation += MathHelper.WrapAngle(controls.DesiredRotation.Value + 0.001f);
                    controls.DesiredRotation = new AvatarAction<float>(0, 0);
                    ac.SetActions(controls);
                });

            while (Renderer.Window.Exists && !token.IsCancellationRequested)
            {
                try
                {
                    Task.Delay(20, token).Wait(token);
                }
                catch (OperationCanceledException)
                {
                    break;
                }

                Renderer.ProcessRequests(World);
                Renderer.Context.SwapBuffers();
            }

            Assert.True(token.IsCancellationRequested);
        }

        private CancellationToken SetupWindow(Action<Vector3> onDrag)
        {
            CancellationTokenSource tokenSource = new CancellationTokenSource(new TimeSpan(1, 0, 0));

            Renderer.Window.KeyDown += (sender, args) =>
            {
                if (args.Key == Key.A)
                    tokenSource.Cancel();
            };

            Renderer.Window.MouseDown += (sender, args) =>
            {
                if (args.Button == MouseButton.Right)
                    tokenSource.Cancel();
            };

            Renderer.Window.MouseMove += (sender, args) =>
            {
                const float factor = 1 / 100f;

                if (args.Mouse.IsButtonDown(MouseButton.Left))
                    onDrag(new Vector3(args.XDelta, args.YDelta, 0) * factor);
            };

            Renderer.Window.Visible = true;

            return tokenSource.Token;
        }
    }


    public class GLRendererTests : GLRendererTestBase
    {
        //[RunnableInDebugOnly]
        /*
        [Fact]
        /*/
        [Fact(Skip = "Skipped -- requires manual input to end.")]
        //**/
        public void ShowRRLongRunning()
        {
            RunRRLongRunning();
        }


        [Fact]
        public void InitRepeated()
        {
            Renderer.CreateWindow("TestGameWindow", 1024, 1024);
            Renderer.CreateContext();

            Renderer.Init();
            Renderer.CreateWindow("TestGameWindow", 1024, 1024);
            Renderer.CreateContext();
        }

        [Fact]
        public void Resize()
        {
            // TODO: Doesn't work -- how to invoke the Resize event on Window?
            //m_renderer.Window.Size = new System.Drawing.Size((int)(m_renderer.Window.Width * 1.3f), (int)(m_renderer.Window.Height * 1.3f));
            Renderer.ProcessRequests(World);
        }
    }
}
