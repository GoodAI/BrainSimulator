using VRageMath;
using World.Physics;

namespace World.GameActors.GameObjects
{
    public abstract class GameObject : GameActor
    {
        public abstract string Name { get; protected set; }

        public IPhysicalEntity PhysicalEntity { get; set; }
    }
}