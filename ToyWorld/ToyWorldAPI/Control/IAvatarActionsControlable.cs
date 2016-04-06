namespace GoodAI.ToyWorld.Control
{
    /// <summary>
    /// 
    /// </summary>
    public interface IControlable
    {
        /// <summary>
        /// Value is clamped to (-1,1). Negative values mean move backwards, positive are for forward movement.
        /// </summary>
        float Acceleration { get; set; }
        /// <summary>
        /// Value is clamped to (-1,1). Negative values mean rotate left, positive are for rotation to the right.
        /// </summary>
        float Rotation { get; set; }
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
    }
}
