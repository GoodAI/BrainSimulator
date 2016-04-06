namespace World.GameActors.Tiles
{
    class Obstacle : StaticTile
    {
        public Obstacle(ITilesetTable tilesetTable) : base(tilesetTable)
        {
        }

        public Obstacle(int tileType) : base(tileType)
        {
        }
    }
}
