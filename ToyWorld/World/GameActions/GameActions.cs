using World.GameActors.Tiles;

namespace World.GameActions
{
    public abstract class GameAction
    {
        public readonly TilesetTable TilesetTable;

        public GameAction(TilesetTable tilesetTable)
        {
            TilesetTable = tilesetTable;
        }
    }

    public class ToUsePickaxe : GameAction
    {
        public float Damage { get; set; }

        public ToUsePickaxe(TilesetTable tilesetTable) : base(tilesetTable)
        {
        }
    }
}