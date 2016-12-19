using VRageMath;
using World.Atlas;
using World.Atlas.Layers;

namespace World.GameActors.Tiles.ObstacleInteractable
{
    public class RcDoor : DynamicTile, ISwitchableGameActor
    {
        private bool m_close = false;

        public RcDoor(Vector2I position) : base(position) { } 

 		public RcDoor(Vector2I position, int textureId) : base(position, textureId) { }

        public RcDoor(Vector2I position, string textureName) : base(position, textureName)
        {
        }

        public ISwitchableGameActor Switch(GameActorPosition gameActorPosition, IAtlas atlas)
        {
            if (m_close)
            {
                SwitchOn(gameActorPosition, atlas);
            }
            else
            {
                SwitchOff(gameActorPosition, atlas);
            }
            m_close = !m_close;
            return this;
        }

        public ISwitchableGameActor SwitchOn(GameActorPosition gameActorPosition, IAtlas atlas)
        {
            TilesetId = AlternativeTextures.Id("Closed");
            atlas.MoveToOtherLayer(new GameActorPosition(this, new Vector2(Position), LayerType.ObstacleInteractable), LayerType.OnGroundInteractable);
            return this;
        }

        public ISwitchableGameActor SwitchOff(GameActorPosition gameActorPosition, IAtlas atlas)
        {
            TilesetId = AlternativeTextures.Id("Open");
            atlas.MoveToOtherLayer(new GameActorPosition(this, new Vector2(Position), LayerType.OnGroundInteractable), LayerType.ObstacleInteractable);
            return this;
        }
    }
}