using System;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using Game;
using GoodAI.ToyWorld.Control;
using OpenTK;
using OpenTK.Input;
using Render.Renderer;
using Render.RenderRequests;
using Render.RenderRequests.Tests;
using Xunit;

namespace ToyWorldTests.Render
{
    public class RendererTests : IDisposable
    {
        private RendererBase m_renderer;


        public RendererTests()
        {
            m_renderer = new GLRenderer();
            m_renderer.Init();
            m_renderer.CreateWindow("TestGameWindow", 1024, 1024);
            m_renderer.CreateContext();
        }

        public void Dispose()
        {
            m_renderer.Dispose();
            m_renderer = null;
        }


        [Fact]
        public void Init()
        {
            Assert.NotNull(m_renderer.Window);
            Assert.NotNull(m_renderer.Context);

            m_renderer.CreateWindow("TestGameWindow", 1024, 1024);
            m_renderer.CreateContext();

            Assert.NotNull(m_renderer.Window);
            Assert.NotNull(m_renderer.Context);

            m_renderer.Reset();
            m_renderer.CreateWindow("TestGameWindow", 1024, 1024);
            m_renderer.CreateContext();

            Assert.NotNull(m_renderer.Window);
            Assert.NotNull(m_renderer.Context);
        }

        [Fact]
        public void Resize()
        {
            // TODO: Doesn't work -- how to invoke the Resize event on Window?
            //m_renderer.Window.Size = new System.Drawing.Size((int)(m_renderer.Window.Width * 1.3f), (int)(m_renderer.Window.Height * 1.3f));
            m_renderer.Window.WindowState = WindowState.Maximized;
            m_renderer.ProcessRequests();
        }
    }
}
