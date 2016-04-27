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
                return PhysicalEntity.ForwardSpeed;
            }
            set
            {
                PhysicalEntity.ForwardSpeed = value;
            }
        }

        public float RotationSpeed
        {
            get
            {
                return PhysicalEntity.RotationSpeed;
            }
            set
            {
                PhysicalEntity.RotationSpeed = value;
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
            ConstructorInfo ctor = shapeType.GetConstructor(new[] { typeof(Vector2) });

            if (ctor == null)
            {
                throw new Exception("Class " + shapeType.FullName + " has no constructor " + shapeType.Name + "(Vector2 v), " +
                                    "hence Character cannot create his PhysicalEntity Shape.");
            }

            Shape shape = (Shape)ctor.Invoke(new object[] {size});
            PhysicalEntity = new ForwardMovablePhysicalEntity(position, shape, direction: direction, tileCollision: tileCollision);
        }
    }
}
