using GoodAI.ToyWorld.Control;

namespace Render.RenderRequests.AgentRenderRequests
{
    internal abstract class AvatarRenderRequestBase : RenderRequest, IAvatarRenderRequest
    {
        protected AvatarRenderRequestBase(int agentID)
        {
            AgentID = agentID;
        }


        #region IAgentRenderRequest overrides

        public float AgentID { get; protected set; }

        public virtual float Size { get; protected set; }
        public virtual float Position { get; set; }
        public virtual float Resolution { get; set; }

        #endregion
    }
}
