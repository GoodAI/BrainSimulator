using System;
using System.Collections.Generic;
using System.Linq;
using VRageMath;
using World.GameActors.Tiles.ObstacleInteractable;
using World.ToyWorldCore;

namespace World.GameActors.Tiles.Obstacle
{
    public class Tree<T> : DynamicTile, IAutoupdateable where T : Fruit, new()
    {
        public int NextUpdateAfter { get; private set; }

        protected Tree(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable, position) { }
        protected Tree(int tileType, Vector2I position) : base(tileType, position) { }

        public void Update(IAtlas atlas, ITilesetTable table)
        {
            List<Vector2I> free = atlas.FreePositionsAround(Position, LayerType.Obstacle).ToList();
            if (free.Count == 0) return;

            Random rng = new Random();
            Vector2I targetPosition = free[rng.Next(free.Count)];
            GameActorPosition fruitPosition = new GameActorPosition(new T(), new Vector2(targetPosition), LayerType.Obstacle);
            atlas.Add(fruitPosition);
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
