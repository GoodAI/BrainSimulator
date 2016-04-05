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
        public void InitRepeated()
        {
            m_renderer.CreateWindow("TestGameWindow", 1024, 1024);
            m_renderer.CreateContext();

            m_renderer.Reset();
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
