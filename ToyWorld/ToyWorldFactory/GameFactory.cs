using GoodAI.ToyWorld.Control;

namespace ToyWorldFactory
{
    public static class GameFactory
    {
        public static IGameController GetGameController()
        {
            return ControllerFactory.GetController();
        }
    }
}
