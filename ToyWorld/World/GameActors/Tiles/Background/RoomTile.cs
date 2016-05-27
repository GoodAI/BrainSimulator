namespace World.GameActors.Tiles.Background
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