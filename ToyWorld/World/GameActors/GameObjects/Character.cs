using GoodAI.ToyWorld.Control;

namespace World.GameActors.GameObjects
{
    public abstract class Character : GameObject, IDirection
    {
        public float Direction { get; set; }
    }
}
