using System;
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
using ToyWorldTests.Attributes;
using ToyWorldTests.Game;
using Xunit;

namespace ToyWorldTests.Render
{
    [Collection("Renderer")]
    public class GLRendererTests : IDisposable
    {
        private readonly GLRenderer m_renderer;


        public GLRendererTests()
        {
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

            m_renderer.Window.Visible = true;
            m_renderer.MakeContextCurrent();

            var rr = RenderRequestFactory.CreateRenderRequest<IFullMapRR>();
            //var rr = RenderRequestFactory.CreateRenderRequest<IFovAvatarRR>(0);
            (rr as RenderRequest).Init(m_renderer);
            m_renderer.EnqueueRequest(rr);

            while (m_renderer.Window.Exists && !tokenSource.IsCancellationRequested)
            {
                try
                {
                    Task.Delay(1000, tokenSource.Token).Wait(tokenSource.Token);
                }
                catch (OperationCanceledException)
                {
                    break;
                }

                m_renderer.Context.SwapBuffers();
                m_renderer.ProcessRequests();
            }

            Assert.True(tokenSource.IsCancellationRequested);
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
            m_renderer.ProcessRequests();
        }
    }
}
