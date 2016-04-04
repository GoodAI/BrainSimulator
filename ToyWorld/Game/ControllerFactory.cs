using Game;
using Render.Renderer;

namespace GoodAI.ToyWorld.Control
{
    public static class ControllerFactory
    {
        public static IGameController GetController(GameSetup gameSetup)
        {
            return new BasicGameController(new GLRenderer(), gameSetup);
        }
    }
}
