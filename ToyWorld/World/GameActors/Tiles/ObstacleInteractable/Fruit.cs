using VRageMath;
using World.GameActions;
using World.ToyWorldCore;

namespace World.GameActors.Tiles.ObstacleInteractable
{
    public class Fruit : DynamicTile, IPickable
    {
        public Fruit(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable, position)
        {
        }

        public Fruit(int tileType, Vector2I position) : base(tileType, position)
        {
        }

        public void ApplyGameAction(IAtlas atlas, GameAction gameAction, Vector2 position, ITilesetTable tilesetTable)
        {
            gameAction.Resolve(new GameActorPosition(this, position), atlas, tilesetTable);
        }
    }

    public interface IEatable
    {
    }

    public class Apple : Fruit, IEatable
    {
        public Apple(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable, position)
        {
        }

        public Apple(int tileType, Vector2I position) : base(tileType, position)
        {
        }
    }

    public class Pear : Fruit, IEatable
    {
        public Pear(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable, position)
        {
        }

        public Pear(int tileType, Vector2I position) : base(tileType, position)
        {
        }
    }

    public class Pinecone : Fruit
    {
        public Pinecone(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable, position)
        {
        }

        public Pinecone(int tileType, Vector2I position) : base(tileType, position)
        {
        }
    }
}
