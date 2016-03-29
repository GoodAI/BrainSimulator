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
        float Size { get; }
        /// <summary>
        /// 
        /// </summary>
        float Position { get; }
        /// <summary>
        /// 
        /// </summary>
        float Resolution { get; }
    }

    /// <summary>
    /// 
    /// </summary>
    public interface IFreeRenderRequest : IRenderRequest
    {
        
    }
}
