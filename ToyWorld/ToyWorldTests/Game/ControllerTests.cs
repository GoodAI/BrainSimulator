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
    public class ControllerTests
    {
        public static IGameController GetTestController()
        {
            var gc = ControllerFactory.GetController();
            gc.Init(null);

            return gc;
        }


        [Fact]
        public void BasicSetup()
        {
            var gc = GetTestController();
            Assert.NotNull(gc);

            gc.RegisterRenderRequest<IRRTest>();
            gc.RegisterAvatarRenderRequest<IARRTest>(0);

            gc.Dispose();
        }

        [Fact]
        public void ControllerNotImplementedThrows()
        {
            var gc = GetTestController();

            Assert.ThrowsAny<RenderRequestNotImplementedException>((Func<object>)gc.RegisterRenderRequest<INotImplementedRR>);
            Assert.ThrowsAny<RenderRequestNotImplementedException>(() => gc.RegisterAvatarRenderRequest<INotImplementedARR>(0));

            // TODO: What to throw for an unknown aID? What should be an aID? How to get allowed aIDs?
            // var ac = gc.GetAvatarController(0);
        }

        [Fact]
        public void RenderNotNull()
        {
            var gc = GetTestController();

            var gcBase = gc as GameControllerBase;
            Assert.NotNull(gcBase);
            Assert.NotNull(gcBase.Renderer);
            Assert.NotNull(gcBase.Renderer.Window);
            Assert.NotNull(gcBase.Renderer.Context);

            gc.Dispose();
        }

        [Fact]
        public void GameNotNull()
        {
            // TODO
        }

        [Fact]
        public void DoStep()
        {
            var gc = GetTestController();

            gc.MakeStep();
            gc.MakeStep();

            gc.Dispose();
        }
    }
}
