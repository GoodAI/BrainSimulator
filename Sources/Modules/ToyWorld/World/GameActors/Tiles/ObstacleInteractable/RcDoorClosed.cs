using VRageMath;
using World.Atlas;
using World.Atlas.Layers;
using World.GameActors.Tiles.OnGroundInteractable;
using World.ToyWorldCore;

namespace World.GameActors.Tiles.ObstacleInteractable
{
    public class RcDoorClosed : DynamicTile, ISwitchableGameActor
    {
        private bool m_close = false;

        public RcDoorClosed(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable, position)
        {
        }

        public RcDoorClosed(int tileType, Vector2I position) : base(tileType, position)
        {
        }

        public ISwitchableGameActor Switch(GameActorPosition gameActorPosition, IAtlas atlas, ITilesetTable table)
        {
            if (m_close)
            {
                SwitchOn(gameActorPosition, atlas, table);
            }
            else
            {
                SwitchOff(gameActorPosition, atlas, table);
            }
            m_close = !m_close;
            return this;
        }

        public ISwitchableGameActor SwitchOn(GameActorPosition gameActorPosition, IAtlas atlas, ITilesetTable table)
        {
            TilesetId = table.TileNumber("RcDoorOpened");
            atlas.MoveToOtherLayer(new GameActorPosition(this, new Vector2(Position), LayerType.ObstacleInteractable), LayerType.OnGroundInteractable);
            return this;
        }

        public ISwitchableGameActor SwitchOff(GameActorPosition gameActorPosition, IAtlas atlas, ITilesetTable table)
        {
            TilesetId = table.TileNumber("RcDoorClosed");
            atlas.MoveToOtherLayer(new GameActorPosition(this, new Vector2(Position), LayerType.OnGroundInteractable), LayerType.ObstacleInteractable);
            return this;
        }
    }
}