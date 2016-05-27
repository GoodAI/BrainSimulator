using VRageMath;
using World.GameActions;
using World.ToyWorldCore;

namespace World.GameActors.Tiles.ObstacleInteractable
{
    public class LeverSwitchOff : DynamicTile, ISwitcherGameActor, IInteractable
    {
        public ISwitchableGameActor Switchable { get; set; }

        public LeverSwitchOff(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable, position)
        {
        }

        public LeverSwitchOff(int tileType, Vector2I position) : base(tileType, position)
        {
        }

        public void Switch(GameActorPosition gameActorPosition, IAtlas atlas, ITilesetTable table)
        {
            var leverOn = new LeverSwitchOn(table, Position);
            atlas.ReplaceWith(new GameActorPosition(this, (Vector2)Position, LayerType.ObstacleInteractable), leverOn);
            if (Switchable == null) return;
            leverOn.Switchable = Switchable.Switch(null, atlas, table);
        }

        public void ApplyGameAction(IAtlas atlas, GameAction gameAction, Vector2 position, ITilesetTable tilesetTable)
        {
            gameAction.Resolve(new GameActorPosition(this, Vector2.PositiveInfinity, LayerType.ObstacleInteractable), atlas, tilesetTable);
        }
    }
}