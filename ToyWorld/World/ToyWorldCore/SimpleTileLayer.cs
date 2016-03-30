using System;
using System.Collections.Generic;
using World.GameActors.Tiles;

namespace World.ToyWorldCore
{
    public class SimpleTileLayer : ITileLayer
    {
        public SimpleTileLayer(LayerType layerType, int width, int height)
        {
            LayerType = layerType;
            Tiles = new Tile[width][];
            for (var i = 0; i < Tiles.Length; i++)
            {
                Tiles[i] = new Tile[height];
            }
        }

        public Tile[][] Tiles { get; private set; }

        public LayerType LayerType { get; set; }

        public Tile GetTile(int x, int y)
        {
            return Tiles[x][y];
        }
    }
}
