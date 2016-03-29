using Game;

namespace GoodAI.ToyWorld.Control
{
    public static class ControllerFactory
    {
        public static IGameController GetController()
        {
            //return new TestGameController();
            return new BasicGameController();
        }
    }
}
