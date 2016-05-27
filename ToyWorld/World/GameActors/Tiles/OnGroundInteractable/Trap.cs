using System.Collections.Generic;
using VRageMath;
using World.Atlas;
using World.Atlas.Layers;
using World.GameActors.GameObjects;
using World.Physics;
using World.ToyWorldCore;

namespace World.GameActors.Tiles.OnGroundInteractable
{
    public class TrapCharged : DynamicTile, IDetectorTile
    {
        public bool RequiresCenterOfObject { get; set; }
        private const float ENERGY_FOR_STEP_ON_TRAP = 0.1f;

        public TrapCharged(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable, position)
        {
            RequiresCenterOfObject = true;
        }

        public TrapCharged(int tileType, Vector2I position) : base(tileType, position)
        {
            RequiresCenterOfObject = true;
        }

        public void ObjectDetected(IGameObject gameObject, IAtlas atlas, ITilesetTable tilesetTable)
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

            var trapDischarged = new TrapDischarged(tilesetTable, Position);
            atlas.RegisterToAutoupdate(trapDischarged);
            atlas.ReplaceWith(new GameActorPosition(this, new Vector2(Position), LayerType.OnGroundInteractable),
                trapDischarged);
        }
    }

    public class TrapDischarged : DynamicTile, IAutoupdateableGameActor
    {
        public int NextUpdateAfter { get; private set; }

        public TrapDischarged(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable, position)
        {
            NextUpdateAfter = 1;
        }

        public TrapDischarged(int tileType, Vector2I position) : base(tileType, position)
        {
            NextUpdateAfter = 1;
        }

        public void Update(IAtlas atlas, ITilesetTable table)
        {
            List<IGameObject> stayingOnTile = atlas.StayingOnTile(Position);
            if (stayingOnTile.Count != 0) return;
            atlas.ReplaceWith(new GameActorPosition(this, new Vector2(Position), LayerType.OnGroundInteractable),
                new TrapCharged(table, Position));
            NextUpdateAfter = 0;
        }
    }
}