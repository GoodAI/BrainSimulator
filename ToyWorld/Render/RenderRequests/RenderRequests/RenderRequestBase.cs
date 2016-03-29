using GoodAI.ToyWorld.Control;
using Render.Renderer;

namespace Render.RenderRequests
{
    internal abstract class RenderRequestBase : IRenderRequest
    {
        public virtual float Size { get; set; }
        public virtual float Position { get; set; }
        public virtual float Resolution { get; set; }

        public abstract void Draw(GLRenderer renderer);
    }
}
