using System.Drawing;

namespace GoodAI.ToyWorld.Control
{
    /// <summary>
    /// 
    /// </summary>
    public interface IRenderRequest : IRenderRequestBase
    { }

    /// <summary>
    /// 
    /// </summary>
    public interface IFullMapRR : IRenderRequest
    { }

    /// <summary>
    /// 
    /// </summary>
    public interface IFreeMapRR : IFullMapRR
    {
        /// <summary>
        /// 
        /// </summary>
        new PointF PositionCenter { get; set; }

        /// <summary>
        /// Equivalent to setting the PositionCenter property (except for the z value).
        /// </summary>
        void SetPositionCenter(float x, float y, float z = 0);
    }
}
