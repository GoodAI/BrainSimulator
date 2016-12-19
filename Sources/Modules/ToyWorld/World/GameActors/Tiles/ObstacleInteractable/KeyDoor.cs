using VRageMath;
using World.Atlas;
using World.Atlas.Layers;

namespace World.GameActors.Tiles.ObstacleInteractable
{
    public class KeyDoor : DynamicTile, ISwitchableGameActor
    {
        private bool m_closed = true;

        public KeyDoor(Vector2I position) : base(position)
        {
            Init();
        }

        public KeyDoor(Vector2I position, int textureId) : base(position, textureId)
        {
            Init();
        }

        public KeyDoor(Vector2I position, string textureName) : base(position, textureName)
        {
            Init();
        }

        private void Init()
        {
            if (AlternativeTextures != null)
                m_closed = TilesetId != AlternativeTextures.Id("Open");
        }

        public ISwitchableGameActor Switch(GameActorPosition gameActorPosition, IAtlas atlas)
        {
            if (m_closed)
            {
                SwitchOn(gameActorPosition, atlas);
            }
            else
            {
                SwitchOff(gameActorPosition, atlas);
            }
            return this;
        }

        public ISwitchableGameActor SwitchOn(GameActorPosition gameActorPosition, IAtlas atlas)
        {
            TilesetId = AlternativeTextures.Id("Open");
            atlas.MoveToOtherLayer(new GameActorPosition(this, (Vector2)Position, LayerType.ObstacleInteractable), LayerType.OnGroundInteractable);
            m_closed = true;
            return this;
        }

        public ISwitchableGameActor SwitchOff(GameActorPosition gameActorPosition, IAtlas atlas)
        {
            TilesetId = AlternativeTextures.Id("Closed");
            atlas.MoveToOtherLayer(new GameActorPosition(this, (Vector2)Position, LayerType.OnGroundInteractable), LayerType.ObstacleInteractable);
            m_closed = false;
            return this;
        }
    }
}