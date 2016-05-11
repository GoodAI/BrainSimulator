using World.GameActors.Tiles.ObstacleInteractable;

namespace World.GameActors.Tiles.Obstacle
{
    public class Tree<T> : StaticTile where T : Fruit
    {
        protected Tree(ITilesetTable tilesetTable) : base(tilesetTable) { }

        protected Tree(int tileType) : base(tileType) { }
    }

    public class AppleTree : Tree<Apple>
    {
        public AppleTree(ITilesetTable tilesetTable) : base(tilesetTable) { }

        public AppleTree(int tileType) : base(tileType) { }
    }

    public class PearTree : Tree<Apple>
    {
        public PearTree(ITilesetTable tilesetTable) : base(tilesetTable) { }

        public PearTree(int tileType) : base(tileType) { }
    }

    public class Pine : Tree<Pinecone>
    {
        public Pine(ITilesetTable tilesetTable) : base(tilesetTable) { }

        public Pine(int tileType) : base(tileType) { }
    }
}
