namespace World.GameActors.Tiles.Path
{
    public class RoomTile : StaticTile
    {
        public RoomTile(ITilesetTable tilesetTable) : base(tilesetTable)
        {
        }

        public RoomTile(int tileType) : base(tileType)
        {
        }
    }
}