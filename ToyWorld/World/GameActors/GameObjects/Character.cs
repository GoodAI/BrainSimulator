using System;
using System.Reflection;
using VRageMath;
using World.Physics;

namespace World.GameActors.GameObjects
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

        public Character(
            string tilesetName,
            int tileId,
            string name,
            Vector2 position,
            Vector2 size,
            float direction,
            TileCollision tileCollision = TileCollision.Slide,
            Type shapeType = null
            )
            : base(tilesetName, tileId, name)
        {
            shapeType = shapeType ?? typeof(Circle);
            ConstructorInfo ctor = shapeType.GetConstructor(new[] {typeof(Vector2)});
            Shape shape = (Shape)ctor.Invoke(new object[] {size});
            PhysicalEntity = new ForwardMovablePhysicalEntity(position, shape, direction: direction, tileCollision: tileCollision);
        }
    }
}
