using World.GameActors.GameObjects;

namespace World.Physics
{
    interface IAvatarMover
    {
        /// <summary>
        /// Maximum speed in meters (length of a side of a tile) per simulation step.
        /// </summary>
        float MaximumSpeed { get; }

        /// <summary>
        /// Maximum rotation speed in degrees per simulation step.
        /// </summary>
        float MaximumRotationSpeed { get; }

        /// <summary>
        /// Sets Avatars PhysicalEntity according to his controls.
        /// </summary>
        /// <param name="avatar"></param>
        void SetAvatarMotion(IAvatar avatar);
    }
}
