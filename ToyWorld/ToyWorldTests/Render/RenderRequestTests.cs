using System;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using Game;
using GoodAI.ToyWorld.Control;
using OpenTK.Input;
using Render.RenderRequests;
using Render.RenderRequests.Tests;
using Xunit;

namespace ToyWorldTests.Render
{
    public class RenderRequestTests : IDisposable
    {
        private GameControllerBase m_gc;


        public RenderRequestTests()
        {
            m_gc = ControllerFactory.GetController() as GameControllerBase;
            Assert.NotNull(m_gc);
            m_gc.Init(null);
        }

        public void Dispose()
        {
            m_gc.Dispose();
            m_gc = null;
        }


        [Fact(Skip = "Long-running; manual input needed for ending.")]
        public void ShowRRLongRunning()
        {
            Key winKeypressResult = default(Key);
            m_gc.Renderer.Window.KeyDown += (sender, args) => winKeypressResult = args.Key;
            m_gc.Renderer.Window.Visible = true;


            var RRTest = m_gc.RegisterAvatarRenderRequest<IAvatarRenderRequestFoV>(0);
            Assert.NotEmpty(RRTest.Image);


            while (winKeypressResult == default(Key))
            {
                Thread.Sleep(100);
                m_gc.MakeStep();
                m_gc.Renderer.Context.MakeCurrent(m_gc.Renderer.Window.WindowInfo);
                m_gc.Renderer.Context.SwapBuffers();
            }


            Assert.Equal(winKeypressResult, Key.A);
        }

        [Fact]
        public void RRInits()
        {
            foreach (var rr in RenderRequestFactory.RRs)
            {
                var r = rr as RenderRequest;
                Assert.NotNull(r);
                r.Init(m_gc.Renderer);
            }

            foreach (var rr in RenderRequestFactory.ARRs)
            {
                var r = rr as RenderRequest;
                Assert.NotNull(r);
                r.Init(m_gc.Renderer);
            }
        }

        [Fact]
        public void AvatarFoV()
        {
            var RRTest = m_gc.RegisterAvatarRenderRequest<IAvatarRenderRequestFoV>(0);
            Assert.NotEmpty(RRTest.Image);

            Assert.Equal(RRTest.Size, RRTest.Image.Length);

            m_gc.MakeStep();
            Assert.Contains(RRTest.Image, u => (u & 0xFFFFFF00) != 0xFFFFFF00);
            m_gc.MakeStep();
            Assert.Contains(RRTest.Image, u => (u & 0xFFFFFF00) != 0xFFFFFF00);
        }
    }
}
