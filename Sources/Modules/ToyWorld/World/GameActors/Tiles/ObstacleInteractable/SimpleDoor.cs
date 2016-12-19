using System.Diagnostics;
using VRageMath;
using World.Atlas;
using World.Atlas.Layers;
using World.GameActions;
using World.GameActors.Tiles.OnGroundInteractable;

namespace World.GameActors.Tiles.ObstacleInteractable
{
    public class SimpleDoor : StaticTile, IInteractableGameActor
    {
        private bool m_closed = true;

        public SimpleDoor() : base()
        {
            Init();
        }

        public SimpleDoor(int textureId) : base(textureId)
        {
            Init();
        }

        public SimpleDoor(string textureName)
            : base(textureName)
        {
            Init();
        }

        private void Init()
        {
            if (AlternativeTextures != null)
                m_closed = TilesetId != AlternativeTextures.Id("Open");
        }

        public void ApplyGameAction(IAtlas atlas, GameAction gameAction, Vector2 position)
        {
            if (m_closed)
            {
                TilesetId = AlternativeTextures.Id("Open");
                atlas.MoveToOtherLayer(new GameActorPosition(this, position, LayerType.ObstacleInteractable), LayerType.OnGroundInteractable);
            }
            else
            {
                TilesetId = AlternativeTextures.Id("Closed");
                atlas.MoveToOtherLayer(new GameActorPosition(this, position, LayerType.OnGroundInteractable), LayerType.ObstacleInteractable);
            }
            m_closed = !m_closed;
        }
    }
}