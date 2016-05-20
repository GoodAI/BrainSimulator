using System.Collections.Generic;
using System.Linq;
using VRageMath;
using World.GameActions;
using World.GameActors.GameObjects;
using World.Physics;
using World.ToyWorldCore;

namespace World.GameActors.Tiles.OnGroundInteractable
{
    public class KeyToDoor : DynamicTile, IPickable, IUsable, ISwitcher
    {
        public KeyToDoor(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable, position)
        {
        }

        public KeyToDoor(int tileType, Vector2I position) : base(tileType, position)
        {
        }

        public void ApplyGameAction(IAtlas atlas, GameAction gameAction, Vector2 position, ITilesetTable tilesetTable)
        {
            gameAction.Resolve(new GameActorPosition(this, position, LayerType.OnGroundInteractable), atlas, tilesetTable);
        }

        public void Use(GameActorPosition senderPosition, IAtlas atlas, ITilesetTable tilesetTable)
        {
            IEnumerable<GameActorPosition> actorsInFrontOf = atlas.ActorsInFrontOf(senderPosition.Actor as ICharacter);
            if (actorsInFrontOf.Select(x => x.Actor).Contains(Switchable as GameActor))
            {
                Switch(atlas, tilesetTable);
            }
        }

        public void Switch(IAtlas atlas, ITilesetTable table)
        {
            Switchable = Switchable.Switch(atlas, table);
        }

        public ISwitchable Switchable { get; set; }
    }
}