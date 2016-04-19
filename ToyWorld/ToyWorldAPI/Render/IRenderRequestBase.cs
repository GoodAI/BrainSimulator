using System.Drawing;

namespace GoodAI.ToyWorld.Control
{
    /// <summary>
    /// 
    /// </summary>
    public interface IRenderRequestBase
    {
        /// <summary>
        /// 
        /// </summary>
        PointF PositionCenter { get; }

        /// <summary>
        /// 
        /// </summary>
        SizeF Size { get; }

        /// <summary>
        /// 
        /// </summary>
        RectangleF View { get; }

        /// <summary>
        /// 
        /// </summary>
        Size Resolution { get; set; }
    }
}
