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
        /// <typeparam name="T"></typeparam>
        /// <param name="action"></param>
        void SetAction<T>(AvatarAction<T> action);

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
