namespace World.GameActors.Tiles
{
    public class Obstacle : StaticTile
    {
        public Obstacle(ITilesetTable tilesetTable)
            : base(tilesetTable)
        {
        }

        public Obstacle(int tileType)
            : base(tileType)
        {
        }
    }
}
