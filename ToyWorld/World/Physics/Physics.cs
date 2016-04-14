using System.Collections.Generic;
using World.GameActors.GameObjects;

namespace World.Physics
{
    public interface IPhysics
    {
        /// <summary>
        /// Transform Avatar's desires (usually set by controller) to physical properties (not directly to motion).
        /// </summary>
        /// <param name="avatars"></param>
        void TransofrmControlsPhysicalProperties(List<IAvatar> avatars);

        /// <summary>
        /// Move with given IForwardMovablePhysicalEntities.
        /// </summary>
        /// <param name="movableEntities"></param>
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


        public void TransofrmControlsPhysicalProperties(List<IAvatar> avatars)
        {
            foreach (IAvatar avatar in avatars)
            {
                m_avatarMover.SetAvatarMotion(avatar);
            }
        }

        public void MoveMovableDirectable(List<IForwardMovablePhysicalEntity> movableEntities)
        {
            foreach (IForwardMovablePhysicalEntity movableEntity in movableEntities)
            {
                m_movementPhysics.Move(movableEntity);
            }
        }
    }
}