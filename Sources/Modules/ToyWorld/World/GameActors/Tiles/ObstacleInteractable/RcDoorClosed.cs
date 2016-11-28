using VRageMath;
using World.Atlas;
using World.Atlas.Layers;
using World.GameActors.Tiles.OnGroundInteractable;
using World.ToyWorldCore;

namespace World.GameActors.Tiles.ObstacleInteractable
{
    public class RcDoorClosed : DynamicTile, ISwitchableGameActor
    {
        public RcDoorClosed(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable, position)
        {
        }

        public RcDoorClosed(int tileType, Vector2I position) : base(tileType, position)
        {
        }

        public ISwitchableGameActor Switch(GameActorPosition gameActorPosition, IAtlas atlas, ITilesetTable table)
        {
            RcDoorOpened openedDoor = new RcDoorOpened(table, Position);
            bool added = atlas.Add(new GameActorPosition(openedDoor, (Vector2)Position, LayerType.OnGroundInteractable));
            if (!added) return this;
            atlas.Remove(new GameActorPosition(this, (Vector2)Position, LayerType.ObstacleInteractable));
            return openedDoor;
        }
    }
}