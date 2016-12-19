using System;
using System.Collections.Generic;
using System.Linq;
using VRageMath;
using World.Atlas;
using World.Atlas.Layers;
using World.GameActors.Tiles.Background;
using World.GameActors.Tiles.ObstacleInteractable;
using World.ToyWorldCore;

namespace World.GameActors.Tiles.Obstacle
{
    public abstract class Tree<T> : DynamicTile, IAutoupdateableGameActor where T : Fruit
    {
        public int NextUpdateAfter { get; private set; }

        private int m_firstUpdate;
        private int m_updatePeriod;
        private static readonly Random m_rng = new Random();

        protected Tree(Vector2I position) : base(position) { Init(); }
        protected Tree(Vector2I position, int textureId) : base(position, textureId) { Init(); }
        protected Tree(Vector2I position, string textureName) : base(position, textureName) { Init(); }

        private void Init()
        {
            m_firstUpdate = TWConfig.Instance.FruitFirstSpawn + m_rng.Next(-TWConfig.Instance.FruitFirstSpawnRange, TWConfig.Instance.FruitFirstSpawnRange);
            m_updatePeriod = TWConfig.Instance.FruitSpawnPeriod + m_rng.Next(-TWConfig.Instance.FruitSpawnRange, TWConfig.Instance.FruitSpawnRange);
            NextUpdateAfter = m_firstUpdate;
        }

        public void Update(IAtlas atlas)
        {
            List<Vector2I> free = atlas.FreePositionsAround(Position, LayerType.Obstacle | LayerType.Object).ToList();
            if (free.Count == 0) return;

            // filter out position with PathTiles
            free = free.Where(x => atlas.ActorsAt((Vector2)x, LayerType.Background).First().Actor.GetType() != typeof(PathTile)).ToList();
            if (free.Count == 0) return;

            Vector2I targetPosition = free[m_rng.Next(free.Count)];
            object[] args = { targetPosition };
            Fruit fruit = (Fruit)Activator.CreateInstance(typeof(T), args);
            GameActorPosition fruitPosition = new GameActorPosition(fruit, new Vector2(targetPosition), LayerType.ObstacleInteractable);
            atlas.Add(fruitPosition);

            NextUpdateAfter = m_updatePeriod;
        }
    }

    public class AppleTree : Tree<Apple>
    {
        public AppleTree(Vector2I position) : base(position) { }
        public AppleTree(Vector2I position, int textureId) : base(position, textureId) { }
        public AppleTree(Vector2I position, string textureName) : base(position, textureName) { }
    }

    public class PearTree : Tree<Apple>
    {
        public PearTree(Vector2I position) : base(position) { }
        public PearTree(Vector2I position, int textureId) : base(position, textureId) { }
        public PearTree(Vector2I position, string textureName) : base(position, textureName) { }
    }

    public class Pine : Tree<Pinecone>
    {
        public Pine(Vector2I position) : base(position) { }
        public Pine(Vector2I position, int textureId) : base(position, textureId) { }
        public Pine(Vector2I position, string textureName) : base(position, textureName) { }
    }
}
