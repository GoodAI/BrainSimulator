using GoodAI.ToyWorld.Control;
using Render.Renderer;

namespace Render.RenderRequests
{
    internal abstract class AgentRenderRequestBase : IAgentRenderRequest
    {
        public float AgentID { get; protected set; }

        public virtual float Size { get; protected set; }
        public virtual float Position { get; set; }
        public virtual float Resolution { get; set; }

        public virtual float MemAddress { get; set; }


        public AgentRenderRequestBase(int agentID)
        {
            AgentID = agentID;
        }

        public abstract void Draw(GLRenderer renderer);
    }
}
