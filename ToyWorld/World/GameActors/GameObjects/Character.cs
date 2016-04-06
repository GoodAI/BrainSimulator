using GoodAI.ToyWorld.Control;

namespace World.GameActors.GameObjects
{
    public abstract class Character : GameObject, IDirectable
    {
        public float Direction { get; set; }
    }
}
