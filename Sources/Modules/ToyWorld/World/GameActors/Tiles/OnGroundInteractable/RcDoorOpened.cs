using VRageMath;
using World.Atlas;
using World.Atlas.Layers;
using World.GameActors.Tiles.ObstacleInteractable;
using World.ToyWorldCore;

namespace World.GameActors.Tiles.OnGroundInteractable
{
    public class RcDoorOpened : DynamicTile, ISwitchableGameActor
    {
        public RcDoorOpened(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable, position)
        {
        }

        public RcDoorOpened(int tileType, Vector2I position) : base(tileType, position)
        {
        }

        public ISwitchableGameActor Switch(GameActorPosition gameActorPosition, IAtlas atlas, ITilesetTable table)
        {
            RcDoorClosed closedDoor = new RcDoorClosed(table, Position);
            bool added = atlas.Add(new GameActorPosition(closedDoor, (Vector2)Position, LayerType.ObstacleInteractable), true);
            if (!added) return this;
            atlas.Remove(new GameActorPosition(this, (Vector2)Position, LayerType.OnGroundInteractable));
            return closedDoor;
        }
    }
}