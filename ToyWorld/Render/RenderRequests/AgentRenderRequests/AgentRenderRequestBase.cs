using GoodAI.ToyWorld.Control;

namespace Render.RenderRequests.AgentRenderRequests
{
    internal abstract class AgentRenderRequestBase : RenderRequest, IAgentRenderRequest
    {
        protected AgentRenderRequestBase(int agentID)
        {
            AgentID = agentID;
        }


        #region IAgentRenderRequest overrides

        public float AgentID { get; protected set; }

        public virtual float Size { get; protected set; }
        public virtual float Position { get; set; }
        public virtual float Resolution { get; set; }

        public virtual float MemAddress { get; set; }

        #endregion
    }
}
