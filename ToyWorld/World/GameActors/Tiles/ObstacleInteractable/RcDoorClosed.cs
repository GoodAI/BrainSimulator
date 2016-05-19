using VRageMath;
using World.GameActors.Tiles.OnGroundInteractable;
using World.ToyWorldCore;

namespace World.GameActors.Tiles.Obstacle
{
    public class RcDoorClosed : DynamicTile, ISwitchable
    {
        public RcDoorClosed(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable, position)
        {
        }

        public RcDoorClosed(int tileType, Vector2I position) : base(tileType, position)
        {
        }

        public GameActor Switch(IAtlas atlas, ITilesetTable table)
        {
            atlas.Remove(new GameActorPosition(this, (Vector2)Position, LayerType.ObstacleInteractable));
            RcDoorOpened openedDoor = new RcDoorOpened(table, Position);
            atlas.Add(new GameActorPosition(openedDoor, (Vector2)Position, LayerType.OnGroundInteractable));
            return openedDoor;
        }
    }
}