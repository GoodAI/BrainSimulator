using GoodAI.ToyWorld.Control;

namespace World.GameActors.GameObjects
{
    public interface ICharacter : IGameObject, IDirectable
    {
    }

    public abstract class Character : GameObject, ICharacter
    {
        public float Direction { get; set; }
    }
}
