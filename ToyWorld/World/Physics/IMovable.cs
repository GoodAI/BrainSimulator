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

        /// <summary>
        /// Move according to internal speed and direction.
        /// </summary>
        void Move();

        /// <summary>
        /// Move according to given speed and direction.
        /// </summary>
        /// <param name="forwardSpeed"></param>
        /// <param name="direction"></param>
        void Move(float forwardSpeed, float direction);
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