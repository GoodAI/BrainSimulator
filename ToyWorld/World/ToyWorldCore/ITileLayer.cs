using World.GameActors.Tiles;
using World.Tiles;

namespace World.ToyWorld
{
    public interface ITileLayer
    {
        Tile GetTile(int x, int y);
    }
}