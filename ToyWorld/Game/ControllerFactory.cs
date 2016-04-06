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
    }
}
