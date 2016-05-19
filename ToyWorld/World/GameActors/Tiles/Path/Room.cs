namespace World.GameActors.Tiles.Path
{
    public class Room : StaticTile
    {
        public Room(ITilesetTable tilesetTable) : base(tilesetTable)
        {
        }

        public Room(int tileType) : base(tileType)
        {
        }
    }
}