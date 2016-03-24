using GoodAI.ToyWorld.Render.RenderRequests;

namespace GoodAI.ToyWorld.Render
{
    /// <summary>
    /// 
    /// </summary>
    public interface IAgentRenderRequest
    {
        /// <summary>
        /// 
        /// </summary>
        float AgentID { get; }
        /// <summary>
        /// 
        /// </summary>
        float Size { get; }
        /// <summary>
        /// 
        /// </summary>
        float Position { get; }
        /// <summary>
        /// 
        /// </summary>
        float Resolution { get; }
        /// <summary>
        /// 
        /// </summary>
        float MemAddress { get; }
    }
}
