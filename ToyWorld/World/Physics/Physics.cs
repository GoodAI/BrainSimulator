using System.Collections.Generic;
using GoodAI.ToyWorld.Control;
using World.GameActors.GameObjects;

namespace World.Physics
{
    public interface IPhysics
    {
        void TransofrmControlsToMotion(List<IAvatar> avatars);
        void MoveMovableDirectable(List<IForwardMovablePhysicalEntity> movableEntities);
    }

    public class Physics : IPhysics
    {
        private readonly IAvatarMover m_avatarMover;

        private readonly IMovementPhysics m_movementPhysics;

        public Physics()
        {
            m_movementPhysics = new MovementPhysics();
            m_avatarMover = new BasicAvatarMover();
        }


        public void TransofrmControlsToMotion(List<IAvatar> avatars)
        {
            foreach (var avatar in avatars)
            {
                m_avatarMover.SetAvatarMotion(avatar);
            }
        }

        public void MoveMovableDirectable(List<IForwardMovablePhysicalEntity> movableEntities)
        {
            foreach (var movableEntity in movableEntities)
            {
                m_movementPhysics.Move(movableEntity);
            }
        }
    }
}