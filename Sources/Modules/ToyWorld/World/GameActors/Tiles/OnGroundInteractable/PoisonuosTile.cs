using System.Collections.Generic;
using System.Linq;
using VRageMath;
using World.Atlas;
using World.GameActors.GameObjects;

namespace World.GameActors.Tiles.OnGroundInteractable
{
    public class PoisonuosTile : DynamicTile, IAutoupdateable, IDetectorTile
    {
        private float ENERGY_FOR_STEP_OR_WAIT_A_SECOND_ON_POISON = 0.1f;
        public int NextUpdateAfter { get; private set; }
        public bool SomeoneOnTile { get; set; }
        public bool RequiresCenterOfObject { get; private set; }

        public PoisonuosTile(Vector2I position) : base(position)
        {
            Ctor();
        }

        public PoisonuosTile(Vector2I position, int textureId) : base(position, textureId) { Ctor(); }

        public PoisonuosTile(Vector2I position, string textureName) : base(position, textureName)
        {
            Ctor();
        }

        private void Ctor()
        {
            NextUpdateAfter = 0;
            RequiresCenterOfObject = true;
        }

        public void ObjectDetected(IGameObject gameObject, IAtlas atlas)
        {
            if(SomeoneOnTile) return;
            NextUpdateAfter = 60;
            atlas.RegisterToAutoupdate(this);
        }

        public void Update(IAtlas atlas)
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