using GoodAI.ToyWorld.Render;
using GoodAI.ToyWorld.Render.RenderRequests;
using GoodAI.TypeMapping;

namespace Render.RenderRequests.Setup
{
    public static class RenderRequestFactory
    {
        static RenderRequestFactory()
        {
            TypeMap.InitializeConfiguration<RenderFactoryContainerConfiguration>();
            TypeMap.Verify();
        }

        public static T CreateRenderRequest<T>()
            where T : class, IRenderRequest // unf cannot constrain T to be an interface, only a class
        {
            return TypeMap.GetInstance<T>();
            // TODO: does it throw when requesting a type that is not registered?
        }

        public static T CreateAgentRenderRequest<T>()
            where T : class, IAgentRenderRequest // unf cannot constrain T to be an interface, only a class
        {
            return TypeMap.GetInstance<T>();
        }
    }
}
