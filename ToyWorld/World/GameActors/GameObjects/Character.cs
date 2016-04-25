using GoodAI.ToyWorld.Control;
using World.GameActors.GameObjects;
using World.Physics;

namespace World.Physics
{
    public interface ICharacter : IGameObject, IForwardMovable
    {
        new IForwardMovablePhysicalEntity PhysicalEntity { get; set; }
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

        public float ForwardSpeed
        {
            get
            {
                return this.PhysicalEntity.ForwardSpeed;
            }
            set
            {
                this.PhysicalEntity.ForwardSpeed = value;
            }
        }

        public float RotationSpeed
        {
            get
            {
                return this.PhysicalEntity.RotationSpeed;
            }
            set
            {
                this.PhysicalEntity.RotationSpeed = value;
            }
        }

        public Character(string tilesetName, int tileID)
            : base(tilesetName, tileID)
        {
        }
    }
}
