using VRageMath;
using World.Physics;

namespace World.GameActors.GameObjects
{
    public interface IGameObject
    {
        string Name { get; }
        IPhysicalEntity PhysicalEntity { get; set; }
    }

    public abstract class GameObject : GameActor, IGameObject
    {
        public abstract string Name { get; protected set; }

        public IPhysicalEntity PhysicalEntity { get; set; }
    }
}