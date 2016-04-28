using World.GameActions;
using World.ToyWorldCore;

namespace World.GameActors.Tiles
{
    public class Fruit : StaticTile, IPickable
    {
        protected Fruit(ITilesetTable tilesetTable) : base(tilesetTable) { }

        protected Fruit(int tileType) : base(tileType) { }

        public void ApplyGameAction(IAtlas atlas, GameAction gameAction, TilesetTable tilesetTable = null)
        {
            gameAction.Resolve(this);
        }
    }

    public class Apple : Fruit
    {
        public Apple(ITilesetTable tilesetTable) : base(tilesetTable) { }

        public Apple(int tileType) : base(tileType) { }
    }

    public class Pear : Fruit
    {
        public Pear(ITilesetTable tilesetTable) : base(tilesetTable) { }

        public Pear(int tileType) : base(tileType) { }
    }

    public class Pinecone : Fruit
    {
        public Pinecone(ITilesetTable tilesetTable) : base(tilesetTable) { }

        public Pinecone(int tileType) : base(tileType) { }
    }
}
