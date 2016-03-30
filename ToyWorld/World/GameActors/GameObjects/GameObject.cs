using VRageMath;

namespace World.GameActors.GameObjects
{
    public abstract class GameObject : GameActor
    {
        public abstract string Name { get; }
        public Point Position { get; set; }
    }
}