namespace GoodAI.ToyWorld.Control
{
    /// <summary>
    /// 
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public struct AvatarAction<T>
    {
        /// <summary>
        /// Value of action.
        /// </summary>
        public T Value { get; private set; }

        /// <summary>
        /// If two or more controllers are connected to one agent, action with highest priority will be performed.
        /// </summary>
        public int Priority { get; private set; }

        /// <summary></summary>
        /// <param name="value">Value of action.</param>
        /// <param name="priority">If two or more controllers are connected to one agent, action with highest priority will be performed.</param>
        public AvatarAction(T value, int priority = 5) : this()
        {
            Value = value;
            Priority = priority;
        }

        public static implicit operator AvatarAction<T>(T value)
        {
            return new AvatarAction<T>(value);
        }

        public static AvatarAction<T> operator +(AvatarAction<T> value1, AvatarAction<T> value2)
        {
            if (value2.Priority <= value1.Priority)
            {
                value1.Priority = value2.Priority;
                value1.Value = value2.Value;
            }
            return value1;
        }
    }
}