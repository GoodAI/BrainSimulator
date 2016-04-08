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


        //[Fact(Skip = "Long-running; manual input needed for ending.")]
        [Fact]
        public void ShowRRLongRunning()
        {
            Key winKeypressResult = default(Key);
            m_renderer.Window.KeyDown += (sender, args) => winKeypressResult = args.Key;
            m_renderer.Window.Visible = true;

            m_renderer.MakeContextCurrent();

            var rr = RenderRequestFactory.CreateRenderRequest<IBasicTexRR>();
            //var rr = RenderRequestFactory.CreateRenderRequest<IFovAvatarRenderRequest>(0);
            (rr as RenderRequest).Init(m_renderer);
            m_renderer.EnqueueRequest(rr);

            while (winKeypressResult == default(Key) && m_renderer.Window.Exists)
            {
                Thread.Sleep(1000);
                m_renderer.ProcessRequests();
                m_renderer.Context.SwapBuffers();
            }

            Assert.Equal(winKeypressResult, Key.A);
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
