using System.Collections.Generic;
using VRageMath;
using World.Atlas;
using World.Atlas.Layers;
using World.GameActors.GameObjects;
using World.Physics;

namespace World.GameActors.Tiles.OnGroundInteractable
{
    public class TrapCharged : DynamicTile, IDetectorTile
    {
        public bool RequiresCenterOfObject { get; set; }
        private const float ENERGY_FOR_STEP_ON_TRAP = 0.1f;

        public TrapCharged(Vector2I position) : base(position)
        {
            Init();
        }

        public TrapCharged(Vector2I position, int textureId) : base(position, textureId) { Init(); }

        public TrapCharged(Vector2I position, string textureName) : base(position, textureName)
        {
            Init();
        }

        private void Init()
        {
            RequiresCenterOfObject = true;
        }

        public void ObjectDetected(IGameObject gameObject, IAtlas atlas)
        {
            var avatar = gameObject as IAvatar;
            if (avatar != null)
            {
                avatar.Energy -= ENERGY_FOR_STEP_ON_TRAP;
            }

            var forwardMovable = gameObject as IForwardMovable;
            if (forwardMovable != null)
            {
                forwardMovable.ForwardSpeed = 0;
            }

            gameObject.Position = Vector2.Floor(gameObject.Position) + Vector2.One/2;

            var trapDischarged = new TrapDischarged(Position);
            atlas.RegisterToAutoupdate(trapDischarged);
            atlas.ReplaceWith(new GameActorPosition(this, new Vector2(Position), LayerType.OnGroundInteractable),
                trapDischarged);
        }
    }

    public class TrapDischarged : DynamicTile, IAutoupdateable
    {
        public int NextUpdateAfter { get; private set; }

        public TrapDischarged(Vector2I position) : base(position)
        {
            NextUpdateAfter = 1;
        }

        public TrapDischarged(Vector2I position, string textureName) : base(position, textureName)
        {
            NextUpdateAfter = 1;
        }

        public void Update(IAtlas atlas)
        {
            List<IGameObject> stayingOnTile = atlas.StayingOnTile(Position);
            if (stayingOnTile.Count != 0) return;
            atlas.ReplaceWith(new GameActorPosition(this, new Vector2(Position), LayerType.OnGroundInteractable),
                new TrapCharged(Position));
            NextUpdateAfter = 0;
        }
    }
}