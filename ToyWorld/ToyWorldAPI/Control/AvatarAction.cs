namespace GoodAI.ToyWorld.Control
{
    /// <summary>
    /// 
    /// </summary>
    public enum AvatarActionEnum
    {
        /// <summary>
        /// 
        /// </summary>
        Rotation,
        /// <summary>
        /// 
        /// </summary>
        Acceleration,
        /// <summary>
        /// 
        /// </summary>
        Use,
        /// <summary>
        /// 
        /// </summary>
        Interact,
        /// <summary>
        /// 
        /// </summary>
        Pick
    }

    /// <summary>
    /// 
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class AvatarAction<T>
    {
        private AvatarActionEnum m_actionId;
        private T m_value;
        private int m_priority;
    }
}
