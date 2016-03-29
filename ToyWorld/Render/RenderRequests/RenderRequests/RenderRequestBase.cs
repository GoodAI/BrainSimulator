using GoodAI.ToyWorld.Control;

namespace Render.RenderRequests.RenderRequests
{
    public abstract class RenderRequestBase : RenderRequest, IRenderRequest
    {
        #region IRenderRequest overrides

        public virtual float Size { get; set; }
        public virtual float Position { get; set; }
        public virtual float Resolution { get; set; }

        #endregion
    }
}
