using GoodAI.ToyWorld.Control;
using World.GameActors.GameObjects;

namespace World.Physics
{
    public interface ICharacter : IGameObject, IDirectable
    {
    }

    public abstract class Character : GameObject, ICharacter
    {
        public float Direction { get; set; }
    }
}
