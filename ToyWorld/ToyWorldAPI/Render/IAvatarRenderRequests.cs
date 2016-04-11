using System;
using System.Drawing;

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
        float AvatarID { get; }
        /// <summary>
        /// 
        /// </summary>
        float Size { get; set; }
        /// <summary>
        /// 
        /// </summary>
        float Position { get; }
        /// <summary>
        /// Relative position of the center of the view to the agent's position.
        /// </summary>
        float RelativePosition { get; set; }
        /// <summary>
        /// 
        /// </summary>
        Point Resolution { get; set; }
    }

    /// <summary>
    /// 
    /// </summary>
    public interface IFovAvatarRR : IAvatarRenderRequest
    {
        /// <summary>
        /// 
        /// </summary>
        uint[] Image { get; }
    }
}
