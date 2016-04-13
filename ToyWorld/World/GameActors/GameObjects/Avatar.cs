using GoodAI.ToyWorld.Control;
using System;
using VRageMath;
using World.Physics;

namespace World.GameActors.GameObjects
{
    public interface IAvatar : IAvatarControllable, ICharacter
    {
        int Id { get;}
        IUsable Tool { get; set; }
        new IForwardMovablePhysicalEntity PhysicalEntity { get; set; }
    }

    public class Avatar : Character, IAvatar
    {
        public int Id { get; private set; }
        public IUsable Tool { get; set; }
        public new IForwardMovablePhysicalEntity PhysicalEntity { get; set; }

        public float DesiredSpeed { get; set; }
        public float DesiredRotation { get; set; }
        public bool Interact { get; set; }
        public bool Use { get; set; }
        public bool PickUp { get; set; }

        public Avatar(string name, int id, Vector2 initialPosition, Vector2 size)
        {
            Name = name;
            Id = id;

            float circleRadius = size.Length() / 2f * (float) Math.Cos(Math.PI / 4);
            PhysicalEntity = new ForwardMovablePhysicalEntity(initialPosition, new Circle(circleRadius));
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
