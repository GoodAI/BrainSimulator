using VRageMath;

namespace World.GameActors.Tiles.OnGroundInteractable
{
    class Bed : DynamicTile
    {
        public Bed(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable, position) { }
        public Bed(int tileType, Vector2I position) : base(tileType, position) { }
    }
}
