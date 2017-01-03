using System.Collections.Generic;
using System.Linq;
using VRageMath;
using World.Atlas;
using World.GameActors.GameObjects;

namespace World.GameActors.Tiles.OnGroundInteractable
{
    public class Recharger : DynamicTile, IAutoupdateable, IDetectorTile
    {
        private float ENERGY_RECHARGED = 0.1f;
        public int NextUpdateAfter { get; private set; }
        public bool SomeoneOnTile { get; set; }
        public bool RequiresCenterOfObject { get; private set; }

        public Recharger(Vector2I position) : base(position)
        {
            Init();
        }

        public Recharger(Vector2I position, int textureId) : base(position, textureId)
        {
            Init();
        }

        public Recharger(Vector2I position, string textureName) : base(position, textureName)
        {
            Init();
        }

        private void Init()
        {
            NextUpdateAfter = 0;
            RequiresCenterOfObject = false;
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
                    avatar.Energy += ENERGY_RECHARGED;
                }
            }
        }
    }
}