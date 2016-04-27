using GoodAI.ToyWorld.Control;
using System.Drawing;
using VRageMath;
using World.Physics;

namespace World.GameActors.GameObjects
{
    public interface IAvatar : IAvatarControllable, ICharacter
    {
        int Id { get; }
        IUsable Tool { get; set; }
    }

    public class Avatar : Character, IAvatar
    {
        public int Id { get; private set; }
        public IUsable Tool { get; set; }

        public float DesiredSpeed { get; set; }
        public float DesiredRotation { get; set; }
        public bool Interact { get; set; }
        public bool Use { get; set; }
        public bool PickUp { get; set; }
        public PointF Fof { get; set; }

        public Avatar(
            string tilesetName,
            int tileId,
            string name,
            int id,
            Vector2 initialPosition,
            Vector2 size,
            float direction = 0
            )
            : base(tilesetName, tileId, name, initialPosition, size, direction, TileCollision.Slide, typeof(Circle))
        {
            Id = id;
        }

        public void ResetControls()
        {
            DesiredSpeed = 0f;
            DesiredRotation = 0f;
            Interact = false;
            Use = false;
            PickUp = false;
            Fof = default(PointF);
        }
    }
}
