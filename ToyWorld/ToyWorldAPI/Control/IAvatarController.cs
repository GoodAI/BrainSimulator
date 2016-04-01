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
        /// <param name="action">Action for avatar.</param>
        void SetAction(AvatarAction<object> action);

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
