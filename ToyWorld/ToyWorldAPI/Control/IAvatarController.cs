using GoodAI.ToyWorldAPI;

namespace GoodAI.ToyWorld.Control
{
    /// <summary>
    ///
    /// </summary>
    public interface IAvatarController : IMessageSender
    {
        string Message { get; }

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

        /// <summary>
        /// Sends text message to Avatar
        /// </summary>
        void SendMessage(string message);

        /// <summary>
        /// Sets the agent message to null
        /// </summary>
        void ClearMessage();
    }
}
