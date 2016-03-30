using World.GameActors.Tiles;

namespace World.ToyWorldCore
{
    public interface ITileLayer
    {
        LayerType LayerType { get; set; }

        Tile GetTile(int x, int y);
    }
}