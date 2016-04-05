using System;
using System.IO;
using Game;
using GoodAI.ToyWorld.Control;
using Render.RenderRequests.Tests;
using Xunit;

namespace ToyWorldTests.Game
{
    public class ControllerTests : IDisposable
    {
        protected IGameController GameController;


        public ControllerTests()
        {
            var tmxMemoryStream = TestingFiles.Files.GetTmxMemoryStream();
            var tilesetTableMemoryStream = TestingFiles.Files.GetTilesetTableMemoryStream();

            var tmxStreamReader = new StreamReader(tmxMemoryStream);
            var tilesetTableStreamReader = new StreamReader(tilesetTableMemoryStream);

            var gameSetup = new GameSetup(tmxStreamReader, tilesetTableStreamReader);

            GameController = ControllerFactory.GetController(gameSetup);

            GameController.Init();
        }

        private static void WriteToMemoryStream(MemoryStream memoryStream, string stringToWrite)
        {
            var stringBytes = System.Text.Encoding.UTF8.GetBytes(stringToWrite);
            memoryStream.Write(stringBytes, 0, stringBytes.Length);
        }

        public void Dispose()
        {
            GameController.Dispose();
            // Test repeated Dispose()
            GameController.Dispose();
            GameController = null;
        }


        // Tests game factory and basic enqueuing
        [Fact]
        public void TestInitAndReset()
        {
            Assert.NotNull(GameController);

            GameController.Reset();
        }

        [Fact]
        public void ControllerNotImplementedThrows()
        {
            Assert.ThrowsAny<RenderRequestNotImplementedException>((Func<object>)GameController.RegisterRenderRequest<INotImplementedRR>);
            Assert.ThrowsAny<RenderRequestNotImplementedException>(() => GameController.RegisterAvatarRenderRequest<INotImplementedARR>(0));

            // TODO: What to throw for an unknown aID? What should be an aID? How to get allowed aIDs?
            // var ac = gc.GetAvatarController(0);
        }

        [Fact]
        public void RenderNotNull()
        {
            var gcBase = GameController as GameControllerBase;
            Assert.NotNull(gcBase);
            Assert.NotNull(gcBase.Renderer);
            Assert.NotNull(gcBase.World);
        }

        [Fact]
        public void GameNotNull()
        {
            // TODO: test world stuff for existence
        }

        [Fact]
        public void DoStep()
        {
            GameController.RegisterRenderRequest<IRRTest>();
            GameController.RegisterAvatarRenderRequest<IARRTest>(0);

            GameController.MakeStep();
            GameController.MakeStep();
        }
    }
}
