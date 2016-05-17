using VRageMath;
using World.GameActions;
using World.ToyWorldCore;

namespace World.GameActors.Tiles.ObstacleInteractable
{
    public class LeverSwitchOn : DynamicTile, ISwitcher, IInteractable
    {
        public ISwitchable Switchable { get; set; }

        public LeverSwitchOn(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable, position)
        {
        }

        public LeverSwitchOn(int tileType, Vector2I position) : base(tileType, position)
        {
        }

        public void Switch(IAtlas atlas, ITilesetTable table)
        {
            LeverSwitchOff leverOff = new LeverSwitchOff(table, Position);
            atlas.ReplaceWith(new GameActorPosition(this, (Vector2)Position), leverOff);
            if (Switchable == null) return;
            leverOff.Switchable = Switchable.Switch(atlas, table) as ISwitchable;
        }

        public void ApplyGameAction(IAtlas atlas, GameAction gameAction, Vector2 position, ITilesetTable tilesetTable)
        {
            gameAction.Resolve(new GameActorPosition(this, Vector2.PositiveInfinity), atlas, tilesetTable);
        }
    }
}