using VRageMath;
using World.Atlas;
using World.Atlas.Layers;
using World.GameActions;
using World.ToyWorldCore;

namespace World.GameActors.Tiles.ObstacleInteractable
{
    public class LeverSwitchOn : DynamicTile, ISwitcherGameActor, IInteractableGameActor
    {
        public ISwitchableGameActor Switchable { get; set; }

        public LeverSwitchOn(ITilesetTable tilesetTable, Vector2I position)
            : base(tilesetTable, position)
        {
        }

        public LeverSwitchOn(int tileType, Vector2I position)
            : base(tileType, position)
        {
        }

        public void Switch(GameActorPosition gameActorPosition, IAtlas atlas, ITilesetTable table)
        {
            var leverOff = new LeverSwitchOff(table, Position);
            atlas.ReplaceWith(new GameActorPosition(this, (Vector2)Position, LayerType.ObstacleInteractable), leverOff);
            if (Switchable == null) return;
            leverOff.Switchable = Switchable.Switch(null, atlas, table) as ISwitchableGameActor;
        }

        public void ApplyGameAction(IAtlas atlas, GameAction gameAction, Vector2 position, ITilesetTable tilesetTable)
        {
            gameAction.Resolve(new GameActorPosition(this, Vector2.PositiveInfinity, LayerType.ObstacleInteractable), atlas, tilesetTable);
        }
    }
}