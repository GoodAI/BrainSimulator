using System;
using System.Threading;
using GoodAI.ToyWorld.Control;
using OpenTK.Input;
using Render.RenderRequests;
using Render.RenderRequests.Tests;
using Xunit;

namespace ToyWorldTests.Game
{
    public class ControllerTests
    {
        public IGameController GetTestController()
        {
            var gc = ControllerFactory.GetController();
            gc.Init(null);

            gc.RegisterRenderRequest<IRRTest>();
            gc.RegisterAgentRenderRequest<IARRTest>(0);

            gc.GetAvatarController(0);

            return gc;
        }


        [Fact]
        public void SetupController()
        {
            IGameController gc = null;

            try
            {
                gc = GetTestController();

                Assert.NotNull(gc);
            }
            catch (Exception)
            {
                Assert.False(true);
            }

            gc.Dispose();
        }

        [Fact(Skip = "Still requiring manual input -- should change later")]
        public void DoStep()
        {
            var gc = GetTestController();

            try
            {
                gc.MakeStep();
            }
            catch (Exception)
            {
                Assert.False(true);
            }
            finally
            {
                gc.Dispose();
            }
        }
    }
}
