using System.Collections.Generic;
using System.Linq;
using VRageMath;
using World.Atlas;
using World.Atlas.Layers;
using World.GameActions;
using World.GameActors.GameObjects;

namespace World.GameActors.Tiles.OnGroundInteractable
{
    public class KeyToDoor : DynamicTile, IPickableGameActor, IUsableGameActor, ISwitcherGameActor
    {
        public KeyToDoor(Vector2I position) : base(position) { } 

 		public KeyToDoor(Vector2I position, int textureId) : base(position, textureId) { }

        public KeyToDoor(Vector2I position, string textureName) : base(position, textureName)
        {
        }

        public void PickUp(IAtlas atlas, GameAction gameAction, Vector2 position)
        {
            gameAction.Resolve(new GameActorPosition(this, position, LayerType.OnGroundInteractable), atlas);
        }

        public void Use(GameActorPosition senderPosition, IAtlas atlas)
        {
            IEnumerable<GameActorPosition> actorsInFrontOf =
                atlas.ActorsInFrontOf(senderPosition.Actor as ICharacter).ToList();
            if (actorsInFrontOf.Select(x => x.Actor).Contains(Switchable as GameActor))
            {
                GameActorPosition switchable = actorsInFrontOf.First(x => x.Actor == Switchable);
                Switch(switchable, atlas);
            }
        }

        public void Switch(GameActorPosition gameActorPosition, IAtlas atlas)
        {
            Switchable = Switchable.Switch(null, atlas);
        }

        public ISwitchableGameActor Switchable { get; set; }
    }
}