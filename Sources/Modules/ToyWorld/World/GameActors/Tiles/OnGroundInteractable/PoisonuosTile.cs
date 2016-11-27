using System.Collections.Generic;
using System.Linq;
using VRageMath;
using World.Atlas;
using World.GameActors.GameObjects;
using World.ToyWorldCore;

namespace World.GameActors.Tiles.OnGroundInteractable
{
    public class PoisonuosTile : DynamicTile, IAutoupdateableGameActor, IDetectorTile
    {
        private float ENERGY_FOR_STEP_OR_WAIT_A_SECOND_ON_POISON = 0.1f;
        public int NextUpdateAfter { get; private set; }
        public bool SomeoneOnTile { get; set; }
        public bool RequiresCenterOfObject { get; private set; }

        public PoisonuosTile(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable, position)
        {
            Ctor();
        }

        public PoisonuosTile(int tileType, Vector2I position) : base(tileType, position)
        {
            Ctor();
        }

        private void Ctor()
        {
            NextUpdateAfter = 0;
            RequiresCenterOfObject = true;
        }

        public void ObjectDetected(IGameObject gameObject, IAtlas atlas, ITilesetTable tilesetTable)
        {
            if(SomeoneOnTile) return;
            NextUpdateAfter = 60;
            atlas.RegisterToAutoupdate(this);
        }

        public void Update(IAtlas atlas, ITilesetTable table)
        {
            List<IGameObject> gameActorPositions = atlas.StayingOnTile(Position);

            SomeoneOnTile = gameActorPositions.Any();

            foreach (IGameObject gameActor in gameActorPositions)
            {
                var avatar = gameActor as IAvatar;
                if (avatar != null)
                {
                    avatar.Energy -= ENERGY_FOR_STEP_OR_WAIT_A_SECOND_ON_POISON;
                }
            }
        }
    }
}