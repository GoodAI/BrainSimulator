using GoodAI.ToyWorld.Control;
using RenderingBase.Renderer;
using World.ToyWorldCore;

namespace Game
{
    public class BasicGameController : GameControllerBase
    {
        public BasicGameController(RendererBase<ToyWorld> renderer, GameSetup gameSetup)
            : base(renderer, gameSetup)
        { }
    }
}
