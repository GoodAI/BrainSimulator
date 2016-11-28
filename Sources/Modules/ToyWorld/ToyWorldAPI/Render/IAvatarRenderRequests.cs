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

        /// <summary>
        /// If true, rotates the map around the agent (which stays fixed looking upwards).
        /// </summary>
        bool RotateMap { get; set; }
    }

    /// <summary>
    /// 
    /// </summary>
    public interface IFovAvatarRR : IAvatarRenderRequest
    { }

    /// <summary>
    /// 
    /// </summary>
    public interface IFofAvatarRR : IAvatarRenderRequest
    {
        /// <summary>
        /// 
        /// </summary>
        IFovAvatarRR FovAvatarRenderRequest { get; set; }
    }

    /// <summary>
    /// 
    /// </summary>
    public interface IToolAvatarRR : IAvatarRenderRequest
    {
        /// <summary>
        /// This property is not used and its value is undefined.
        /// </summary>
        new PointF RelativePosition { get; }

        /// <summary>
        /// This property is not used and its value is undefined.
        /// </summary>
        new bool RotateMap { get; }
    }
}
