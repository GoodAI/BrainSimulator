using System.Diagnostics;
using World.GameActors.GameObjects;

namespace World.Physics
{
    class BasicAvatarMover : IAvatarMover
    {
        public void SetAvatarMotion(Avatar avatar)
        {
            var physicalEntity = avatar.PhysicalEntity as MovableDirectablePhysicalEntity;
            Debug.Assert(physicalEntity != null, "physicalEntity != null");
            physicalEntity.ForwardSpeed = avatar.DesiredSpeed;
            physicalEntity.RotationSpeed = avatar.DesiredRotation;
        }
    }
}
