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
using Render.Tests.RRs;
using ToyWorldTests.Game;
using Xunit;

namespace ToyWorldTests.Render
{
    public class RenderRequestTests : ControllerTests
    {
        private readonly GameControllerBase m_gc;
        private readonly GLRenderer m_renderer;


        public RenderRequestTests()
        {
            m_gc = GameController as GameControllerBase;
            Assert.NotNull(m_gc);
            m_renderer = m_gc.Renderer as GLRenderer;
        }


        //[Fact(Skip = "Long-running; manual input needed for ending.")]
        [Fact]
        public void ShowRRLongRunning()
        {
            Key winKeypressResult = default(Key);
            m_renderer.Window.KeyDown += (sender, args) => winKeypressResult = args.Key;
            m_renderer.Window.Visible = true;

            var RRTest = m_gc.RegisterRenderRequest<IBasicTexRR>();

            while (winKeypressResult == default(Key) && m_renderer.Window.Exists)
            {
                Thread.Sleep(100);
                m_gc.MakeStep();
                m_renderer.MakeContextCurrent();
                m_renderer.Context.SwapBuffers();
            }

            Assert.Equal(winKeypressResult, Key.A);
        }

        [Fact]
        public void RRInits()
        {
            Assert.NotNull(m_renderer);
            m_renderer.MakeContextCurrent();

            foreach (var rr in RenderRequestFactory.RRs)
            {
                var r = rr as RenderRequest;
                Assert.NotNull(r);
                r.Init(m_renderer);
                m_renderer.CheckError();
            }

            foreach (var rr in RenderRequestFactory.ARRs)
            {
                var r = rr as RenderRequest;
                Assert.NotNull(r);
                r.Init(m_renderer);
                m_renderer.CheckError();
            }

            m_renderer.MakeContextNotCurrent();
        }


        [Fact]
        public void AvatarFoV()
        {
            var RRTest = m_gc.RegisterAvatarRenderRequest<IAvatarRenderRequestFoV>(0);

            Assert.NotEmpty(RRTest.Image);
            Assert.Equal(RRTest.Size, RRTest.Image.Length);

            m_gc.MakeStep();
            Assert.Contains(RRTest.Image, u => (u & 0xFFFFFF00) != 0);
            m_gc.MakeStep();
            Assert.Contains(RRTest.Image, u => (u & 0xFFFFFF00) != 0);
        }
    }
}
