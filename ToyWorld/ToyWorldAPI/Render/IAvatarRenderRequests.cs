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
        Size Size { get; }

        /// <summary>
        /// 
        /// </summary>
        PointF PositionCenter { get; }

        /// <summary>
        /// Relative position of PositionCenter to avatar's position
        /// </summary>
        PointF RelativePosition { get; set; }

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
    public interface IFovAvatarRR : IAvatarRenderRequest
    {
        /// <summary>
        /// 
        /// </summary>
        uint[] Image { get; }
    }
}
