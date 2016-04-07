using GoodAI.ToyWorld.Control;
using VRageMath;
using World.Physics;

namespace World.GameActors.GameObjects
{
    public interface IAvatar : IAvatarControllable
    {
        IUsable Tool { get; set; }
        MovableDirectablePhysicalEntity PhysicalEntity { get; set; }
    }

    public class Avatar : Character, IAvatar
    {
        public readonly int Id;
        public sealed override string Name { get; protected set; }
        public IUsable Tool { get; set; }
        public MovableDirectablePhysicalEntity PhysicalEntity { get; set; }

        public float DesiredSpeed { get; set; }
        public float DesiredRotation { get; set; }
        public bool Interact { get; set; }
        public bool Use { get; set; }
        public bool PickUp { get; set; }

        public Avatar(string name, int id, Vector2 initialPosition)
        {
            Name = name;
            Id = id;
            PhysicalEntity = new MovableDirectablePhysicalEntity(initialPosition);
        }

        public void ResetControls()
        {
            DesiredSpeed = 0f;
            DesiredRotation = 0f;
            Interact = false;
            Use = false;
            PickUp = false;
        }
    }
}
