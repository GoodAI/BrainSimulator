using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using World.GameActors.Tiles;
using World.ToyWorld;

namespace World
{
    public class SimpleTileLayer : ITileLayer
    {
        public List<List<Tile>> Tiles { get; set; }
    
        public Tile GetTile(int x, int y)
        {
            throw new NotImplementedException();
        }
    }
}
