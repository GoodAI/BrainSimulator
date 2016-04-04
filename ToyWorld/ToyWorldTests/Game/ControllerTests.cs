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
        private IGameController m_gameController;


        public ControllerTests()
        {
            var tmxMemoryStream = TestingFiles.Files.GetTmxMemoryStream();
            var tilesetTableMemoryStream = TestingFiles.Files.GetTilesetTableMemoryStream();

            var tmxStreamReader = new StreamReader(tmxMemoryStream);
            var tilesetTableStreamReader = new StreamReader(tilesetTableMemoryStream);

            var gameSetup = new GameSetup(tmxStreamReader, tilesetTableStreamReader);

            m_gameController = ControllerFactory.GetController(gameSetup);

            m_gameController.Init();
        }

        private static void WriteToMemoryStream(MemoryStream memoryStream, string stringToWrite)
        {
            var stringBytes = System.Text.Encoding.UTF8.GetBytes(stringToWrite);
            memoryStream.Write(stringBytes, 0, stringBytes.Length);
        }

        public void Dispose()
        {
            m_gameController.Dispose();
            m_gameController = null;
        }


//        I think this doesn't need to be tesetd
//        [Fact]
//        public void Init()
//        {
//            Assert.NotNull(m_gameController);
//
//            m_gameController.RegisterRenderRequest<IRRTest>();
//            m_gameController.RegisterAvatarRenderRequest<IARRTest>(0);
//
//            m_gameController.GetAvatarController(0);
//
//            m_gameController.Reset();
//        }

        [Fact]
        public void ControllerNotImplementedThrows()
        {
            Assert.ThrowsAny<RenderRequestNotImplementedException>((Func<object>)m_gameController.RegisterRenderRequest<INotImplementedRR>);
            Assert.ThrowsAny<RenderRequestNotImplementedException>(() => m_gameController.RegisterAvatarRenderRequest<INotImplementedARR>(0));

            // TODO: What to throw for an unknown aID? What should be an aID? How to get allowed aIDs?
            // var ac = gc.GetAvatarController(0);
        }

        [Fact]
        public void RenderNotNull()
        {
            var gcBase = m_gameController as GameControllerBase;
            Assert.NotNull(gcBase);
            Assert.NotNull(gcBase.Renderer);
        }

        [Fact]
        public void GameNotNull()
        {
            // TODO: test world stuff for existence
        }

        [Fact]
        public void DoStep()
        {
            m_gameController.RegisterRenderRequest<IRRTest>();
            m_gameController.RegisterAvatarRenderRequest<IARRTest>(0);

            m_gameController.MakeStep();
            m_gameController.MakeStep();
        }
    }
}
