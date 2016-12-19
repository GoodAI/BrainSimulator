using VRageMath;
using World.Atlas;
using World.Atlas.Layers;
using World.GameActions;
using World.ToyWorldCore;

namespace World.GameActors.Tiles.ObstacleInteractable
{
    public class Fruit : DynamicTile, IPickableGameActor, ICombustibleGameActor, IInteractableGameActor
    {
        public Fruit(Vector2I position) : base(position) { }

        public Fruit(Vector2I position, int textureId) : base(position, textureId) { }

        public Fruit(Vector2I position, string textureName) : base(position, textureName) { }

        public void PickUp(IAtlas atlas, GameAction gameAction, Vector2 position)
        {
            gameAction.Resolve(new GameActorPosition(this, position, LayerType.ObstacleInteractable), atlas);
        }

        public void Burn(GameActorPosition gameActorPosition, IAtlas atlas)
        {
            atlas.Remove(gameActorPosition);
        }

        public void ApplyGameAction(IAtlas atlas, GameAction gameAction, Vector2 position)
        {
            gameAction.Resolve(new GameActorPosition(this, position, LayerType.ObstacleInteractable), atlas);
        }
    }

    public interface IEatable : IAutoupdateableGameActor { }

    public class Apple : Fruit, IEatable
    {
        public int NextUpdateAfter { get; private set; }

        public Apple(Vector2I position) : base(position) { Init(); }

        public Apple(Vector2I position, int textureId) : base(position, textureId) { Init(); }

        public Apple(Vector2I position, string textureName) : base(position, textureName) { Init(); }

        private void Init()
        {
            NextUpdateAfter = TWConfig.Instance.FruitRotAfter; // time to rot
        }

        public void Update(IAtlas atlas)
        {
            atlas.Remove(new GameActorPosition(this, (Vector2)Position, LayerType.ObstacleInteractable));
        }
    }

    public class Pear : Fruit, IEatable
    {
        public int NextUpdateAfter { get; private set; }

        public Pear(Vector2I position) : base(position) { Init(); }

        public Pear(Vector2I position, int textureId) : base(position, textureId) { Init(); }

        public Pear(Vector2I position, string textureName) : base(position, textureName) { Init(); }

        private void Init()
        {
            NextUpdateAfter = TWConfig.Instance.FruitRotAfter; // time to rot
        }

        public void Update(IAtlas atlas)
        {
            atlas.Remove(new GameActorPosition(this, (Vector2)Position, LayerType.ObstacleInteractable));
        }
    }

    public class Pinecone : Fruit
    {
        public Pinecone(Vector2I position) : base(position) { }

        public Pinecone(Vector2I position, int textureId) : base(position, textureId) { }

        public Pinecone(Vector2I position, string textureName) : base(position, textureName) { }
    }
}
