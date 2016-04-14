using GoodAI.ToyWorld.Control;
using World.Physics;

namespace World.GameActors.GameObjects
{
    public interface ICharacter : IGameObject, IForwardMovable
    {
    }

    public abstract class Character : GameObject, ICharacter
    {
        public new IForwardMovablePhysicalEntity PhysicalEntity
        {
            get
            {
                return (IForwardMovablePhysicalEntity)base.PhysicalEntity;
            }
            set
            {
                base.PhysicalEntity = value;
            }
        }

        public float Direction
        {
            get
            {
                return PhysicalEntity.Direction;
            }
            set
            {
                PhysicalEntity.Direction = value;
            }
        }

        public float ForwardSpeed { get; set; }

        public float RotationSpeed { get; set; }
    }
}
