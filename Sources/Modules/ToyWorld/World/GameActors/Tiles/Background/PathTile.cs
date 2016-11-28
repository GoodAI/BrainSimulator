namespace World.GameActors.Tiles.Background
{
    public class PathTile : StaticTile
    {
        public PathTile(ITilesetTable tileset)
            : base(tileset)
        {
        }

        public PathTile(int tileType)
            : base(tileType)
        {
        }
    }
}
