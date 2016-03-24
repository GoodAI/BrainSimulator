using Game;

namespace GoodAI.ToyWorld.Control
{
    public static class GameControlFactory
    {
        public static IGameController GetController()
        {
            return new TestGameController();
        }
    }
}
