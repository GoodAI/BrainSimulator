using GoodAI.ToyWorld.Control;

namespace ToyWorldFactory
{
    public static class GameFactory
    {
        public static IGameController GetGameController(GameSetup gameSetup)
        {
            return ControllerFactory.GetController(gameSetup);
        }

        public static IGameController GetThreadSafeGameController(GameSetup gameSetup)
        {
            return ControllerFactory.GetThreadSafeController(gameSetup);
        }

        public static int GetSignalCount()
        {
            return ControllerFactory.GetSignalCount();
        }
    }
}
