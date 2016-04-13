using System.Diagnostics;
using World.GameActors.GameObjects;

namespace World.Physics
{
    public class BasicAvatarMover : IAvatarMover
    {
        public const float MAXIMUM_SPEED = 0.2f;
        public float MaximumSpeed { get { return MAXIMUM_SPEED; } }

        public const float MAXIMUM_ROTATION_SPEED = 27.5f;
        public float MaximumRotationSpeed { get { return MAXIMUM_ROTATION_SPEED; } }

        public void SetAvatarMotion(IAvatar avatar)
        {
            IForwardMovablePhysicalEntity physicalEntity = avatar.PhysicalEntity;
            Debug.Assert(physicalEntity != null, "physicalEntity != null");
            physicalEntity.ForwardSpeed = avatar.DesiredSpeed * MaximumSpeed;
            physicalEntity.RotationSpeed = avatar.DesiredRotation * MaximumRotationSpeed;
        }
    }
}
