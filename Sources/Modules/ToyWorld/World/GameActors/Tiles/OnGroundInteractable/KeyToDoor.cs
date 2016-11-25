using System.Collections.Generic;
using System.Linq;
using VRageMath;
using World.Atlas;
using World.Atlas.Layers;
using World.GameActions;
using World.GameActors.GameObjects;
using World.ToyWorldCore;

namespace World.GameActors.Tiles.OnGroundInteractable
{
    public class KeyToDoor : DynamicTile, IPickableGameActor, IUsableGameActor, ISwitcherGameActor
    {
        public KeyToDoor(ITilesetTable tilesetTable, Vector2I position)
            : base(tilesetTable, position)
        {
        }

        public KeyToDoor(int tileType, Vector2I position)
            : base(tileType, position)
        {
        }

        public void PickUp(IAtlas atlas, GameAction gameAction, Vector2 position, ITilesetTable tilesetTable)
        {
            gameAction.Resolve(new GameActorPosition(this, position, LayerType.OnGroundInteractable), atlas, tilesetTable);
        }

        public void Use(GameActorPosition senderPosition, IAtlas atlas, ITilesetTable tilesetTable)
        {
            IEnumerable<GameActorPosition> actorsInFrontOf =
                atlas.ActorsInFrontOf(senderPosition.Actor as ICharacter).ToList();
            if (actorsInFrontOf.Select(x => x.Actor).Contains(Switchable as GameActor))
            {
                GameActorPosition switchable = actorsInFrontOf.First(x => x.Actor == Switchable);
                Switch(switchable, atlas, tilesetTable);
            }
        }

        public void Switch(GameActorPosition gameActorPosition, IAtlas atlas, ITilesetTable table)
        {
            Switchable = Switchable.Switch(null, atlas, table);
        }

        public ISwitchableGameActor Switchable { get; set; }
    }
}