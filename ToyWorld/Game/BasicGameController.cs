using GoodAI.ToyWorld.Control;
using Render.Renderer;

namespace Game
{
    public class BasicGameController : GameControllerBase
    {
        public BasicGameController(RendererBase renderer, GameSetup gameSetup)
            : base(renderer, gameSetup)
        { }
    }
}
