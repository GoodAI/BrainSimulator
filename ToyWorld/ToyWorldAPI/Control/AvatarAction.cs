namespace GoodAI.ToyWorld.Control
{
    public class AvatarAction<T>
    {
        /// <summary>
        /// Action to preform.
        /// </summary>
        public readonly AvatarActionEnum ActionId;

        /// <summary>
        /// Value of action.
        /// </summary>
        public readonly T Value;

        /// <summary>
        /// If two or more controllers are connected to one agent, action with highest priority will be performed.
        /// </summary>
        public readonly int Priority;

        /// <summary></summary>
        /// <param name="actionId">Action to preform.</param>
        /// <param name="value">Value of action.</param>
        /// <param name="priority">If two or more controllers are connected to one agent, action with highest priority will be performed.</param>
        public AvatarAction(AvatarActionEnum actionId, T value, int priority = 5)
        {
            ActionId = actionId;
            Value = value;
            Priority = priority;
        }
    }
}