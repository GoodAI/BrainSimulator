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
        SizeF Size { get; set; }

        /// <summary>
        /// Determined by PositionCenter and Size.
        /// </summary>
        RectangleF View { get; }


        /// <summary>
        /// 
        /// </summary>
        Size Resolution { get; set; }


        /// <summary>
        /// 
        /// </summary>
        bool GatherImage { get; set; }

        /// <summary>
        /// 
        /// </summary>
        uint[] Image { get; }
    }
}
