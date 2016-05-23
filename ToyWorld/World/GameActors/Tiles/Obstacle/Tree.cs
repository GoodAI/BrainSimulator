using System;
using System.Collections.Generic;
using System.Linq;
using VRageMath;
using World.GameActors.Tiles.ObstacleInteractable;
using World.ToyWorldCore;

namespace World.GameActors.Tiles.Obstacle
{
    public abstract class Tree<T> : DynamicTile, IAutoupdateable where T : Fruit
    {
        public int NextUpdateAfter { get; private set; }

        private int m_firstUpdate;
        private int m_updatePeriod;
        private static readonly Random m_rng = new Random();

        protected Tree(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable, position) { Init(); }
        protected Tree(int tileType, Vector2I position) : base(tileType, position) { Init(); }

        private void Init()
        {
            m_firstUpdate = 500 + m_rng.Next(-200, 200);
            m_updatePeriod = 5000 + m_rng.Next(-1000, 1000);
            NextUpdateAfter = m_firstUpdate;
        }

        public void Update(IAtlas atlas, ITilesetTable table)
        {
            List<Vector2I> free = atlas.FreePositionsAround(Position, LayerType.Obstacle).ToList();
            if (free.Count == 0) return;

            Random rng = new Random();
            Vector2I targetPosition = free[rng.Next(free.Count)];
            object[] args = { table, targetPosition };
            Fruit fruit = (Fruit)Activator.CreateInstance(typeof(T), args);
            GameActorPosition fruitPosition = new GameActorPosition(fruit, new Vector2(targetPosition), LayerType.ObstacleInteractable);
            atlas.Add(fruitPosition);

            NextUpdateAfter = m_updatePeriod;
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
