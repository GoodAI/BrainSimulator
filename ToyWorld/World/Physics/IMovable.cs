namespace GoodAI.ToyWorld.Control
{
    /// <summary>
    /// For game objects that can move.
    /// </summary>
    public interface IForwardMovable
    {
        /// <summary>
        /// Speed has no limit. It is limited by physics engine.
        /// </summary>
        float ForwardSpeed { get; set; }

        /// <summary>
        /// Rotation speed has no limit. It is limited by physics engine.
        /// </summary>
        float RotationSpeed { get; set; }
    }

    public interface IGridMovable
    {
        /// <summary>
        /// Speed left right. (-1,1)
        /// </summary>
        float XSpeed { get; set; }

        /// <summary>
        /// Speed upside down. (-1,1)
        /// </summary>
        float YSpeed { get; set; }
    }

    /// <summary>
    /// For GameObjects with direction
    /// </summary>
    public interface IDirectable
    {
        /// <summary>
        /// Direction in degrees. 0 means right, 90 down, 180 left, 270 up.
        /// </summary>
        float Direction { get; set; }
    }
}