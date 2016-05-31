using Game;
using RenderingBase.Renderer;
using World.ToyWorldCore;

namespace GoodAI.ToyWorld.Control
{
    public static class ControllerFactory
    {
        public static GameControllerBase GetController(GameSetup gameSetup)
        {
            return new BasicGameController(new GLRenderer<World.ToyWorldCore.ToyWorld>(), gameSetup);
        }

        public static GameControllerBase GetThreadSafeController(GameSetup gameSetup)
        {
            return new ThreadSafeGameController(new GLRenderer<World.ToyWorldCore.ToyWorld>(), gameSetup);
        }

        public static int GetSignalCount()
        {
            return TWConfig.Instance.SignalCount;
        }
    }
}
