using System.Diagnostics;
using VRageMath;
using World.GameActors.GameObjects;

namespace World.Physics
{
    public class BasicAvatarMover : IAvatarMover
    {
        private const float MAXIMUM_SPEED = 0.2f;
        public float MaximumSpeed
        {
            get { return MAXIMUM_SPEED; }
        }

        private const float MAXIMUM_ROTATION_SPEED_DEG = 11.25f;
        public float MaximumRotationSpeed
        {
            get { return MAXIMUM_ROTATION_SPEED_DEG; }
        }

        public void SetAvatarMotion(IAvatar avatar)
        {
            IForwardMovablePhysicalEntity physicalEntity = avatar.PhysicalEntity;
            Debug.Assert(physicalEntity != null, "physicalEntity != null");
            physicalEntity.ForwardSpeed = avatar.DesiredSpeed * MaximumSpeed;
            physicalEntity.RotationSpeed = avatar.DesiredRotation * MathHelper.ToRadians(MaximumRotationSpeed);
        }
    }
}
