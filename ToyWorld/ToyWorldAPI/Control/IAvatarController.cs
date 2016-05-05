using GoodAI.ToyWorldAPI;

namespace GoodAI.ToyWorld.Control
{
    /// <summary>
    ///
    /// </summary>
    public interface IAvatarController : IMessageSender
    {
        string MessageIn { get; set; }
        string MessageOut { get; set; }

        /// <summary>
        ///
        /// </summary>
        /// <param name="actions">Action for avatar.</param>
        void SetActions(IAvatarControls actions);

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

        /// <summary>
        /// Sets initial values to controls. Agent should not move.
        /// </summary>
        void ResetControls();
    }
}
