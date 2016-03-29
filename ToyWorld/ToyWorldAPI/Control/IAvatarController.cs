namespace GoodAI.ToyWorld.Control
{
    /// <summary>
    /// 
    /// </summary>
    public enum ActionPriority
    {
        One,
        Two,
        Three,
        Four,
        Five,
        Six,
        Seven,
        Eight,
        Nine,
        Ten,
    }


    /// <summary>
    /// 
    /// </summary>
    public interface IAvatarController
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="priority">Specifies the priority with which the actions will be set on the returned object.
        /// For every action, only the highest-priority setting will be kept.</param>
        /// <returns>The object through which to set the Avatar's action.</returns>
        IAvatarAction GetActions(ActionPriority priority = ActionPriority.Five);

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
