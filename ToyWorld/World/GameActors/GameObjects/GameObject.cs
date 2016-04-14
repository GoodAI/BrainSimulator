using VRageMath;
using World.Physics;

namespace World.GameActors.GameObjects
{
    public interface IGameObject
    {
        string Name { get; }
        IPhysicalEntity PhysicalEntity { get; set; }
        Vector2 Size { get; set; }
        Vector2 Position { get; set; }
    }

    public abstract class GameObject : GameActor, IGameObject
    {
        public string Name { get; protected set; }
        public IPhysicalEntity PhysicalEntity { get; set; }

        Vector2 IGameObject.Size
        {
            get
            {
                return PhysicalEntity.Size;
            }
            set
            {
                PhysicalEntity.Size = value;
            }
        }

        Vector2 IGameObject.Position
        {
            get
            {
                return PhysicalEntity.Position;
            }
            set
            {
                PhysicalEntity.Position = value;
            }
        }
    }
}