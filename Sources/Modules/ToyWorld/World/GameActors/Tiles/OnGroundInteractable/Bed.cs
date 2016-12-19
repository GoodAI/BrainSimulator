using System.Collections.Generic;
using System.Linq;
using VRageMath;
using World.Atlas;
using World.GameActors.GameObjects;
using World.ToyWorldCore;

namespace World.GameActors.Tiles.OnGroundInteractable
{
    public class Bed : DynamicTile, IAutoupdateableGameActor, IDetectorTile
    {
        public bool RequiresCenterOfObject { get { return true; } }
        public int NextUpdateAfter { get; private set; }
        public bool SomeoneOnTile { get; set; }

        public Bed(Vector2I position) : base(position) { Init(); }

        public Bed(Vector2I position, int textureId) : base(position, textureId) { Init(); }

        public Bed(Vector2I position, string textureName) : base(position, textureName) { Init(); }

        private void Init()
        {
            NextUpdateAfter = 0;
        }

        public void ObjectDetected(IGameObject gameObject, IAtlas atlas)
        {
            if (SomeoneOnTile) return;
            NextUpdateAfter = 60;
            atlas.RegisterToAutoupdate(this);
        }

        public void Update(IAtlas atlas)
        {
            List<IGameObject> gameActorPositions = atlas.StayingOnTile(Position);

            SomeoneOnTile = gameActorPositions.Any();

            foreach (IGameObject gameActor in gameActorPositions)
            {
                IAvatar avatar = gameActor as IAvatar;
                if (avatar != null)
                    avatar.Rested += TWConfig.Instance.BedRechargeRate;
            }
        }
    }
}
