namespace World.Physics
{
    /// <summary>
    /// For game objects that can move.
    /// </summary>
    public interface IForwardMovable : IDirectable
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
    /// For GameObjects with direction. Game Object Moves towards direction, but must not be rotated in the way.
    /// </summary>
    public interface IDirectable
    {
        /// <summary>
        /// Direction in radians. 0 means up, -Pi/2 right.
        /// </summary>
        float Direction { get; set; }
    }

    public interface IRotatable
    {
        /// <summary>
        /// Direction in radians. 0 means up, -Pi/2 right.
        /// </summary>
        float Rotation { get; set; }
    }
}