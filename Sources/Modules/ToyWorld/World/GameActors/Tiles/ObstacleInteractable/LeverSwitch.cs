using VRageMath;
using World.Atlas;
using World.Atlas.Layers;
using World.GameActions;

namespace World.GameActors.Tiles.ObstacleInteractable
{
    public class LeverSwitch : DynamicTile, ISwitcherGameActor, IInteractableGameActor
    {
        public ISwitchableGameActor Switchable { get; set; }

        private bool m_On;

        public LeverSwitch(Vector2I position) : base(position) { Init(); } 

        public LeverSwitch(Vector2I position, int textureId) : base(position, textureId) { Init(); }

        public LeverSwitch(Vector2I position, string textureName) : base(position, textureName)
        {
            Init();
        }

        private void Init()
        {
            if(AlternativeTextures == null) { return; }
            if (TilesetId == AlternativeTextures.Id("On"))
            {
                m_On = true;
            }
            else
            {
                m_On = false;
            }
        }

        public void Switch(GameActorPosition gameActorPosition, IAtlas atlas)
        {
            if (m_On)
            {
                SwitchOff(gameActorPosition, atlas);
            }
            else
            {
                SwitchOn(gameActorPosition, atlas);
            }
        }

        private void SwitchOn(GameActorPosition gameActorPosition, IAtlas atlas)
        {
            TilesetId = AlternativeTextures.Id("On");
            Switchable = Switchable?.Switch(gameActorPosition, atlas);
            m_On = true;
        }

        private void SwitchOff(GameActorPosition gameActorPosition, IAtlas atlas)
        {
            TilesetId = AlternativeTextures.Id("Off");
            Switchable = Switchable?.Switch(gameActorPosition, atlas);
            m_On = false;
        }

        public void ApplyGameAction(IAtlas atlas, GameAction gameAction, Vector2 position)
        {
            gameAction.Resolve(new GameActorPosition(this, Vector2.PositiveInfinity, LayerType.ObstacleInteractable), atlas);
        }
    }
}