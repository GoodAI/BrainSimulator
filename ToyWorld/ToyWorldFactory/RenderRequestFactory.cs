using GoodAI.ToyWorld.Render;
using GoodAI.ToyWorld.Render.RenderRequests;

namespace Render.RenderRequests.Setup
{
    public static class RenderRequestFactory
    {
        public static T CreateRenderRequest<T>()
            where T : class, IRenderRequest // unf cannot constrain T to be an interface, only a class
        {
        }

        public static T CreateAgentRenderRequest<T>()
            where T : class, IAgentRenderRequest // unf cannot constrain T to be an interface, only a class
        {
        }
    }
}
