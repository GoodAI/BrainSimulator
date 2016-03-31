using System;
using System.Threading;
using Game;
using GoodAI.ToyWorld.Control;
using OpenTK.Input;
using Render.RenderRequests;
using Render.RenderRequests.Tests;
using Xunit;

namespace ToyWorldTests.Game
{
    public class ControllerTests : IDisposable
    {
        private IGameController m_gc;


        public ControllerTests()
        {
            m_gc = ControllerFactory.GetController();
            m_gc.Init(null);
        }

        public void Dispose()
        {
            m_gc.Dispose();
            m_gc = null;
        }


        [Fact]
        public void BasicSetup()
        {
            Assert.NotNull(m_gc);

            m_gc.RegisterRenderRequest<IRRTest>();
            m_gc.RegisterAvatarRenderRequest<IARRTest>(0);

            m_gc.GetAvatarController(0);
        }

        [Fact]
        public void ControllerNotImplementedThrows()
        {
            Assert.ThrowsAny<RenderRequestNotImplementedException>((Func<object>)m_gc.RegisterRenderRequest<INotImplementedRR>);
            Assert.ThrowsAny<RenderRequestNotImplementedException>(() => m_gc.RegisterAvatarRenderRequest<INotImplementedARR>(0));

            // TODO: What to throw for an unknown aID? What should be an aID? How to get allowed aIDs?
            // var ac = gc.GetAvatarController(0);
        }

        [Fact]
        public void RenderNotNull()
        {
            var gcBase = m_gc as GameControllerBase;
            Assert.NotNull(gcBase);
            Assert.NotNull(gcBase.Renderer);
            Assert.NotNull(gcBase.Renderer.Window);
            Assert.NotNull(gcBase.Renderer.Context);
        }

        [Fact]
        public void GameNotNull()
        {
            // TODO: test world stuff for existence
        }

        [Fact]
        public void DoStep()
        {
            m_gc.MakeStep();
            m_gc.MakeStep();

            m_gc.Dispose();
        }
    }
}
