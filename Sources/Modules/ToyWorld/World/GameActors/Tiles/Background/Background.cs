namespace World.GameActors.Tiles.Background
{
    public class Background : StaticTile
    {
        public Background(ITilesetTable tileset)
            : base(tileset)
        {
        }

        public Background(int tileType)
            : base(tileType)
        {
        }
    }
}
