using VRageMath;
using World.GameActors.Tiles.Obstacle;
using World.ToyWorldCore;

namespace World.GameActors.Tiles.OnGroundInteractable
{
    public class RcDoorOpened : DynamicTile, ISwitchable
    {

        public RcDoorOpened(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable, position)
        {
        }

        public RcDoorOpened(int tileType, Vector2I position) : base(tileType, position)
        {
        }

        public GameActor Switch(IAtlas atlas, ITilesetTable table)
        {
            atlas.Remove(new GameActorPosition(this, (Vector2)Position, LayerType.ObstacleInteractable));
            RcDoorClosed closedDoor = new RcDoorClosed(table, Position);
            atlas.Add(new GameActorPosition(closedDoor, (Vector2)Position, LayerType.ObstacleInteractable));
            return closedDoor;
        }
    }
}