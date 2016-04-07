using System;
using System.IO;
using Game;
using GoodAI.ToyWorld.Control;
using Render.Tests.RRs;
using Xunit;

namespace ToyWorldTests.Game
{
    public class ControllerTests : IDisposable
    {
        protected GameControllerBase GameController;


        public ControllerTests()
        {
            var tmxMemoryStream = FileStreams.GetTmxMemoryStream();
            var tilesetTableMemoryStream = FileStreams.GetTilesetTableMemoryStream();

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
            Assert.ThrowsAny<RenderRequestNotImplementedException>(() => GameController.RegisterRenderRequest<INotImplementedARR>(0));

            // TODO: What to throw for an unknown aID? What should be an aID? How to get allowed aIDs?
            // var ac = gc.GetAvatarController(0);
        }

        [Fact]
        public void DoStep()
        {
            GameController.RegisterRenderRequest<IBasicTexRR>();
            GameController.RegisterRenderRequest<IBasicARR>(0);

            GameController.MakeStep();
            GameController.MakeStep();
        }
    }
}
