namespace GoodAI.ToyWorld.Control
{
    /// <summary>
    /// 
    /// </summary>
    public interface IAvatarController
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="actions">Action for avatar.</param>
        void SetActions(AvatarControls actions);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        IStats GetStats();

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        string GetComment();
    }
}
