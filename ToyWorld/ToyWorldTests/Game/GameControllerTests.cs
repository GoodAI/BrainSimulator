using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Game;
using GoodAI.ToyWorld.Control;
using OpenTK.Input;
using Render.Renderer;
using Render.RenderRequests;
using Render.Tests.RRs;
using ToyWorldTests.Attributes;
using Xunit;

namespace ToyWorldTests.Game
{
    [Collection("Renderer")]
    public class GameControllerTestBase : IDisposable
    {
        protected GameControllerBase GameController;


        public GameControllerTestBase()
        {
            var tmxMemoryStream = FileStreams.GetTmxMemoryStream();
            var tilesetTableMemoryStream = FileStreams.GetTilesetTableMemoryStream();

            var tmxStreamReader = new StreamReader(tmxMemoryStream);
            var tilesetTableStreamReader = new StreamReader(tilesetTableMemoryStream);

            var gameSetup = new GameSetup(tmxStreamReader, tilesetTableStreamReader);

            GameController = GetController(gameSetup);

            GameController.Init();
        }

        public void Dispose()
        {
            GameController.Dispose();
            // Test repeated Dispose()
            GameController.Dispose();
            GameController = null;
        }

        protected virtual GameControllerBase GetController(GameSetup gameSetup)
        {
            return ControllerFactory.GetController(gameSetup);
        }
    }

    [Collection("Renderer")]
    public class BasicGameControllerTests : GameControllerTestBase
    {
        // Tests game factory and basic enqueuing
        [Fact]
        public void TestInit()
        {
            Assert.NotNull(GameController);
            Assert.NotNull(GameController.Renderer);
            Assert.NotNull(GameController.World);
        }

        [Fact]
        public void ControllerNotImplementedThrows()
        {
            Assert.ThrowsAny<RenderRequestNotImplementedException>((Func<object>)GameController.RegisterRenderRequest<INotImplementedRR>);
            Assert.ThrowsAny<RenderRequestNotImplementedException>(() => GameController.RegisterRenderRequest<INotImplementedAvatarRR>(0));

            // TODO: What to throw for an unknown aID? What should be an aID? How to get allowed aIDs?
            // var ac = gc.GetAvatarController(0);
        }

        [Fact]
        public void DoStep()
        {
            GameController.RegisterRenderRequest<IBasicTextureRR>();
            GameController.RegisterRenderRequest<IFovAvatarRR>(0);

            GameController.MakeStep();
            GameController.MakeStep();
        }
    }

    public class ThreadSafeGameControllerTests : GameControllerTestBase
    {
        protected override GameControllerBase GetController(GameSetup gameSetup)
        {
            return ControllerFactory.GetThreadSafeController(gameSetup);
        }
    }
}
