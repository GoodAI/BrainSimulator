using System.Diagnostics;
using World.GameActors.GameObjects;

namespace World.Physics
{
    public class BasicAvatarMover : IAvatarMover
    {

        public void SetAvatarMotion(IAvatar avatar)
        {
            IForwardMovablePhysicalEntity physicalEntity = avatar.PhysicalEntity;
            Debug.Assert(physicalEntity != null, "physicalEntity != null");
            physicalEntity.ForwardSpeed = avatar.DesiredSpeed / 6;
            physicalEntity.RotationSpeed = avatar.DesiredRotation * 11.25f;
        }

        public float MaximumSpeed
        {
            // TODO
            get { throw new System.NotImplementedException(); }
        }

        public float MaximumRotationSpeed
        {
            // TODO
            get { throw new System.NotImplementedException(); }
        }
    }
}
