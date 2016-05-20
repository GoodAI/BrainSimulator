using VRageMath;
using World.GameActors.Tiles.OnGroundInteractable;
using World.ToyWorldCore;

namespace World.GameActors.Tiles.ObstacleInteractable
{
    public class KeyDoorClosed : DynamicTile, ISwitchable
    {
        public KeyDoorClosed(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable, position)
        {
        }

        public KeyDoorClosed(int tileType, Vector2I position) : base(tileType, position)
        {
        }

        public ISwitchable Switch(IAtlas atlas, ITilesetTable table)
        {
            KeyDoorOpened openedDoor = new KeyDoorOpened(table, Position);
            bool added = atlas.Add(new GameActorPosition(openedDoor, (Vector2)Position, LayerType.OnGroundInteractable));
            if (!added) return this;
            atlas.Remove(new GameActorPosition(this, (Vector2)Position, LayerType.ObstacleInteractable));
            return openedDoor;
        }
    }
}