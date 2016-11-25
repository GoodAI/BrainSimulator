using GoodAI.ToyWorld.Control;
using Render.Renderer;

namespace Game
{
    public class BasicGameController : GameControllerBase
    {
        public BasicGameController(ToyWorldRenderer renderer, GameSetup gameSetup)
            : base(renderer, gameSetup)
        { }
    }
}
