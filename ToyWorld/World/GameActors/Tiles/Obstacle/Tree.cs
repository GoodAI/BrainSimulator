using VRageMath;
using World.GameActors.Tiles.ObstacleInteractable;
using World.ToyWorldCore;

namespace World.GameActors.Tiles.Obstacle
{
    public class Tree<T> : DynamicTile, IAutoupdateable where T : Fruit
    {
        public int NextUpdateAfter { get; private set; }

        protected Tree(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable, position) { }
        protected Tree(int tileType, Vector2I position) : base(tileType, position) { }

        public void Update(IAtlas atlas, ITilesetTable table)
        {
            throw new System.NotImplementedException();
        }
    }

    public class AppleTree : Tree<Apple>
    {
        public AppleTree(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable, position) { }
        public AppleTree(int tileType, Vector2I position) : base(tileType, position) { }
    }

    public class PearTree : Tree<Apple>
    {
        public PearTree(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable, position) { }
        public PearTree(int tileType, Vector2I position) : base(tileType, position) { }
    }

    public class Pine : Tree<Pinecone>
    {
        public Pine(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable, position) { }
        public Pine(int tileType, Vector2I position) : base(tileType, position) { }
    }
}
