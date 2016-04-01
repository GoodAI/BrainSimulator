namespace GoodAI.ToyWorld.Control
{
    public class AvatarAction<T>
    {
        /// <summary>
        /// Action to preform.
        /// </summary>
        public AvatarActionEnum ActionId { get; set; }
        /// <summary>
        /// Value of action.
        /// </summary>
        public T Value { get; set; }
        /// <summary>
        /// If two or more controllers are connected to one agent, action with highest priority will be performed.
        /// </summary>
        public int Priority { get; set; }

        public AvatarAction(AvatarActionEnum actionId, T value, int priority = 5)
        {
            ActionId = actionId;
            Value = value;
            Priority = priority;
        }
    }
}