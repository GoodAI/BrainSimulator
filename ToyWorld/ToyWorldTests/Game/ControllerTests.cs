using System.Threading;
using GoodAI.ToyWorld.Control;
using OpenTK.Input;
using Render.RenderRequests;
using Xunit;

namespace ToyWorldTests.Game
{
    public class ControllerTests
    {
        [Fact]
        public void SetupController()
        {
            var gc = ControllerFactory.GetController();
            gc.Init(null);

            var RRTest = gc.RegisterRenderRequest<IRRTest>();
            gc.RegisterAgentRenderRequest<IRenderRequestFoV>(0);

            var ac = gc.GetAvatarController(0);

            //Assert.True(ac != null);
        }
    }
}
