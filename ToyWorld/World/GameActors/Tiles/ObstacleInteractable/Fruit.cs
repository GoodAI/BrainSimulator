using VRageMath;
using World.GameActions;
using World.ToyWorldCore;

namespace World.GameActors.Tiles.ObstacleInteractable
{
    public class Fruit : DynamicTile, IPickable, ICombustible, IInteractable
    {
        public Fruit(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable, position) { }

        public Fruit(int tileType, Vector2I position) : base(tileType, position) { }

        public void PickUp(IAtlas atlas, GameAction gameAction, Vector2 position, ITilesetTable tilesetTable)
        {
            gameAction.Resolve(new GameActorPosition(this, position, LayerType.ObstacleInteractable), atlas, tilesetTable);
        }

        public void Burn(GameActorPosition gameActorPosition, IAtlas atlas, ITilesetTable table)
        {
            atlas.Remove(gameActorPosition);
        }

        public void ApplyGameAction(IAtlas atlas, GameAction gameAction, Vector2 position, ITilesetTable tilesetTable)
        {
            gameAction.Resolve(new GameActorPosition(this, position, LayerType.ObstacleInteractable), atlas, tilesetTable);
        }
    }

    public interface IEatable : IAutoupdateable { }

    public class Apple : Fruit, IEatable
    {
        public int NextUpdateAfter { get; private set; }

        public Apple(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable, position) { Init(); }

        public Apple(int tileType, Vector2I position) : base(tileType, position) { Init(); }

        private void Init()
        {
            NextUpdateAfter = TWConfig.Instance.FruitRotAfter; // time to rot
        }

        public void Update(IAtlas atlas, ITilesetTable table)
        {
            atlas.Remove(new GameActorPosition(this, (Vector2)Position, LayerType.ObstacleInteractable));
        }
    }

    public class Pear : Fruit, IEatable
    {
        public int NextUpdateAfter { get; private set; }

        public Pear(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable, position) { Init(); }

        public Pear(int tileType, Vector2I position) : base(tileType, position) { Init(); }

        private void Init()
        {
            NextUpdateAfter = TWConfig.Instance.FruitRotAfter; // time to rot
        }

        public void Update(IAtlas atlas, ITilesetTable table)
        {
            atlas.Remove(new GameActorPosition(this, (Vector2)Position, LayerType.ObstacleInteractable));
        }
    }

    public class Pinecone : Fruit
    {
        public Pinecone(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable, position) { }

        public Pinecone(int tileType, Vector2I position) : base(tileType, position) { }
    }
}
