using System;
using System.Collections.Generic;
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
        private bool m_disposed;

        protected GameControllerBase GameController;


        public GameControllerTestBase()
        {
            var tmxMemoryStream = FileStreams.SmallTmx();
            var tilesetTableMemoryStream = FileStreams.TilesetTableStream();

            var tilesetTableStreamReader = new StreamReader(tilesetTableMemoryStream);

            var gameSetup = new GameSetup(tmxMemoryStream, tilesetTableStreamReader);

            GameController = GetController(gameSetup);

            GameController.Init();
        }

        ~GameControllerTestBase()
        {
            Dispose();
        }

        public void Dispose()
        {
            if (m_disposed)
                return;

            GameController.Dispose();
            // Test repeated Dispose()
            GameController.Dispose();
            GameController = null;

            m_disposed = true;
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
            Assert.ThrowsAny<RenderRequestNotImplementedException>(() => GameController.RegisterRenderRequest<INotImplementedAvatarRR>(1));

            Assert.ThrowsAny<RenderRequestNotImplementedException>((Func<object>)GameController.RegisterRenderRequest<IRenderRequest>);
            Assert.ThrowsAny<RenderRequestNotImplementedException>(() => GameController.RegisterRenderRequest<IAvatarRenderRequest>(1));

            // TODO: What to throw for an unknown aID? What should be an aID? How to get allowed aIDs?
            Assert.ThrowsAny<KeyNotFoundException>(() => GameController.GetAvatarController(-1));
        }

        [Fact]
        public void DoStep()
        {
            GameController.RegisterRenderRequest<IFullMapRR>();
            GameController.RegisterRenderRequest<IFovAvatarRR>(1);

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
