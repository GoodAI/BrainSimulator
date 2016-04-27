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
            using (var tmxStream = FileStreams.FullTmxStream())
            using (var tilesetTableStreamReader = new StreamReader(FileStreams.TilesetTableStream()))
            {
                var gameSetup = new GameSetup(tmxStream, tilesetTableStreamReader);
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
            var rr = m_gameController.RegisterRenderRequest<IFovAvatarRR>(aID);
            //var rr = m_gameController.RegisterRenderRequest<IFofAvatarRR>(aID);
            //rr1.Size = new SizeF(50, 50);
            //rr.FovAvatarRenderRequest = rr1;
            rr.GatherImage = true;

            var ac = m_gameController.GetAvatarController(aID);
            var controls = new AvatarControls(5) { DesiredSpeed = .3f };

            CancellationToken token = SetupWindow(
                delta =>
                {
                    //rr.PositionCenter = new PointF(rr.PositionCenter.X - delta.X, rr.PositionCenter.Y + delta.Y);
                    controls.DesiredRotation = MathHelper.WrapAngle(delta.X * 0.1f);
                    //controls.Fof = new PointF(0, controls.Fof.Value.Y + -delta.Y * 0.1f);
                },
                delta =>
                {
                    rr.Size = new SizeF(rr.Size.Width - delta, rr.Size.Height - delta);
                },
                key =>
                {
                    if (key == Key.Number1)
                        rr.DrawNoise = !rr.DrawNoise;
                });


            while (Renderer.Window.Exists && !token.IsCancellationRequested)
            {
                try
                {
                    Task.Delay(15, token).Wait(token);
                }
                catch (OperationCanceledException)
                {
                    break;
                }

                ac.SetActions(controls);

                m_gameController.MakeStep();
                Renderer.Context.SwapBuffers();
            }

            Assert.True(token.IsCancellationRequested);
        }

        private CancellationToken SetupWindow(Action<Vector3> onDrag, Action<float> onScroll, Action<Key> onToggle)
        {
            CancellationTokenSource tokenSource = new CancellationTokenSource(new TimeSpan(1, 0, 0));

            Renderer.Window.KeyDown += (sender, args) =>
            {
                switch (args.Key)
                {
                    case Key.A:
                        tokenSource.Cancel();
                        break;

                    case Key.Number1:
                    case Key.Number2:
                    case Key.Number3:
                    case Key.Number4:
                        onToggle(args.Key);
                        break;
                }
            };

            Renderer.Window.MouseDown += (sender, args) =>
            {
                switch (args.Button)
                {
                    case MouseButton.Right:
                        tokenSource.Cancel();
                        break;
                    case MouseButton.Left:
                        onDrag(Vector3.Zero);
                        break;
                }
            };

            Renderer.Window.MouseWheel += (sender, args) =>
            {
                if (onScroll != null)
                    onScroll(args.Delta);
            };

            Renderer.Window.MouseMove += (sender, args) =>
            {
                const float factor = 1 / 5f;

                if (args.Mouse.IsButtonDown(MouseButton.Left) && onDrag != null)
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
