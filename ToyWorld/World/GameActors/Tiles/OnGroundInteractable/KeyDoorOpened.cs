using VRageMath;
using World.GameActors.Tiles.ObstacleInteractable;
using World.ToyWorldCore;

namespace World.GameActors.Tiles.OnGroundInteractable
{
    public class KeyDoorOpened : DynamicTile, ISwitchable
    {
        public KeyDoorOpened(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable, position)
        {
        }

        public KeyDoorOpened(int tileType, Vector2I position) : base(tileType, position)
        {
        }

        public ISwitchable Switch(IAtlas atlas, ITilesetTable table)
        {
            KeyDoorClosed closedDoor = new KeyDoorClosed(table, Position);
            bool added = atlas.Add(new GameActorPosition(closedDoor, (Vector2)Position, LayerType.ObstacleInteractable), true);
            if (!added) return this;
            atlas.Remove(new GameActorPosition(this, (Vector2)Position, LayerType.OnGroundInteractable));
            return closedDoor;
        }
    }
}