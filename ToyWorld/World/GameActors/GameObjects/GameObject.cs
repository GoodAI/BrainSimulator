using VRageMath;

namespace World.GameActors.GameObjects
{
    public abstract class GameObject : GameActor
    {
        public abstract string Name { get; protected set; }
        public Point Position { get; set; }
    }
}