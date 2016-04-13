using System;
using System.Drawing;

namespace GoodAI.ToyWorld.Control
{
    /// <summary>
    /// 
    /// </summary>
    public interface IRenderRequest
    {
        /// <summary>
        /// 
        /// </summary>
        Size Size { get; }
        /// <summary>
        /// 
        /// </summary>
        PointF PositionCenter { get; }
        /// <summary>
        /// 
        /// </summary>
        Size Resolution { get; set; }
    }

    /// <summary>
    /// 
    /// </summary>
    public interface IFullMapRR : IRenderRequest
    {
        float Rotation { get; set; }
    }
}
