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
        RectangleF View { get; }


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
        /// <summary>
        /// 
        /// </summary>
        PointF Rotation { get; set; }
    }

    /// <summary>
    /// 
    /// </summary>
    public interface IFreeMapRR : IFullMapRR
    {
        /// <summary>
        /// 
        /// </summary>
        new PointF PositionCenter { get; set; }
    }
}
