using GoodAI.ToyWorld.Control;
using Render.Renderer;

namespace Render.RenderRequests
{
    public abstract class RenderRequest
    {
        public virtual void Init(IRenderer renderer)
        { }
        
        public abstract void Draw(GLRenderer renderer);
    }
}
