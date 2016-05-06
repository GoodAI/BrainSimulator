namespace World.GameActors.Tiles
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
