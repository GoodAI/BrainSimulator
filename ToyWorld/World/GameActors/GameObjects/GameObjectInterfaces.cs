namespace GoodAI.ToyWorld.Control
{
    /// <summary>
    /// Object which implement this interface can be controlled by AvatarController.
    /// </summary>
    public interface IAvatarControllable
    {
        /// <summary>
        /// To interact with object in front.
        /// </summary>
        bool Interact { get; set; }

        /// <summary>
        /// To use tool in hand / punch.
        /// </summary>
        bool Use { get; set; }

        /// <summary>
        /// Pick up or put down tool in hand.
        /// </summary>
        bool PickUp { get; set; }

        /// <summary>
        /// Set controls to default position.
        /// </summary>
        void ResetControls();

        /// <summary>
        /// Value is clamped to (-1,1). Negative values mean move backwards, positive are for forward movement.
        /// </summary>
        float DesiredSpeed { get; set; }

        /// <summary>
        /// Value is clamped to (-1,1). Negative values mean rotate left, positive are for rotation to the right.
        /// </summary>
        float DesiredRotation { get; set; }
    }
}