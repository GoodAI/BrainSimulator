using VRageMath;
using World.GameActions;
using World.ToyWorldCore;

namespace World.GameActors.Tiles.ObstacleInteractable
{
    public class LeverSwitchOff : DynamicTile, ISwitcher, IInteractable
    {
        public ISwitchable Switchable { get; set; }

        public LeverSwitchOff(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable, position)
        {
        }

        public LeverSwitchOff(int tileType, Vector2I position) : base(tileType, position)
        {
        }

        public void Switch(IAtlas atlas, ITilesetTable table)
        {
            LeverSwitchOn leverOn = new LeverSwitchOn(table, Position);
            atlas.ReplaceWith(new GameActorPosition(this, (Vector2) Position), leverOn);
            if (Switchable == null) return;
            leverOn.Switchable = Switchable.Switch(atlas, table) as ISwitchable;
        }

        public void ApplyGameAction(IAtlas atlas, GameAction gameAction, Vector2 position, ITilesetTable tilesetTable)
        {
            gameAction.Resolve(new GameActorPosition(this, Vector2.PositiveInfinity), atlas, tilesetTable);
        }
    }
}