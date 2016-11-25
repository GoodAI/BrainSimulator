namespace World.GameActors.Tiles.Obstacle
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
