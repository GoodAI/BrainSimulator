using GoodAI.ToyWorld.Control;

namespace ToyWorldFactory
{
    public static class GameFactory
    {
        public static IGameController GetGameController(GameSetup gameSetup)
        {
            return ControllerFactory.GetController(gameSetup);
        }
    }
}
