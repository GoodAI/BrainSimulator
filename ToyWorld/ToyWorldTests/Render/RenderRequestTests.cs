using System;
using System.IO;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using Game;
using GoodAI.ToyWorld.Control;
using OpenTK.Graphics.OpenGL;
using OpenTK.Input;
using Render.RenderRequests;
using Render.RenderRequests.Tests;
using ToyWorldTests.Game;
using Xunit;

namespace ToyWorldTests.Render
{
    public class RenderRequestTests : ControllerTests
    {
        private readonly GameControllerBase m_gc;


        public RenderRequestTests()
        {
            m_gc = GameController as GameControllerBase;
        }


        [Fact(Skip = "Long-running; manual input needed for ending.")]
        //[Fact]
        public void ShowRRLongRunning()
        {
            Key winKeypressResult = default(Key);
            m_gc.Renderer.Window.KeyDown += (sender, args) => winKeypressResult = args.Key;
            m_gc.Renderer.Window.Visible = true;


            var RRTest = m_gc.RegisterAvatarRenderRequest<IARRTest>(0);


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
            Assert.Contains(RRTest.Image, u => (u & 0xFFFFFF00) != 0);
            m_gc.MakeStep();
            Assert.Contains(RRTest.Image, u => (u & 0xFFFFFF00) != 0);
        }
    }
}
