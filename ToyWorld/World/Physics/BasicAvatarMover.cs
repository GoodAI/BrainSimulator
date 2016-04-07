using System.Diagnostics;
using World.GameActors.GameObjects;

namespace World.Physics
{
    public class BasicAvatarMover : IAvatarMover
    {
        public void SetAvatarMotion(IAvatar avatar)
        {
            IMovableDirectablePhysicalEntity physicalEntity = avatar.PhysicalEntity;
            Debug.Assert(physicalEntity != null, "physicalEntity != null");
            physicalEntity.ForwardSpeed = avatar.DesiredSpeed;
            physicalEntity.RotationSpeed = avatar.DesiredRotation;
        }
    }
}
