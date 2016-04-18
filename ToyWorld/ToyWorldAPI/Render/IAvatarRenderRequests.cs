using System.Drawing;

namespace GoodAI.ToyWorld.Control
{
    /// <summary>
    /// 
    /// </summary>
    public interface IAvatarRenderRequest : IRenderRequestBase
    {
        /// <summary>
        /// 
        /// </summary>
        int AvatarID { get; }


        /// <summary>
        /// Relative position of PositionCenter to avatar's position
        /// </summary>
        PointF RelativePosition { get; set; }

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
