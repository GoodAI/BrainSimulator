using System;

namespace GoodAI.ToyWorld.Control
{
    /// <summary>
    /// 
    /// </summary>
    public interface IAvatarRenderRequest
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
        float Resolution { get; set; }
    }

    /// <summary>
    /// 
    /// </summary>
    public interface IRenderRequestFoV : IAvatarRenderRequest
    {

    }
}
