using Game;
using Render.Renderer;

namespace GoodAI.ToyWorld.Control
{
    public static class ControllerFactory
    {
        public static GameControllerBase GetController(GameSetup gameSetup)
        {
            return new BasicGameController(new GLRenderer(), gameSetup);
        }

        public static GameControllerBase GetThreadSafeController(GameSetup gameSetup)
        {
            return new ThreadSafeGameController(new GLRenderer(), gameSetup);
        }

        public static int GetSignalCount()
        {
            return World.ToyWorldCore.ToyWorld.SignalCount;
        }
    }
}
