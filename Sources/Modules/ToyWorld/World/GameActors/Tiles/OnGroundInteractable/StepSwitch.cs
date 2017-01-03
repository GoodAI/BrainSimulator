using VRageMath;
using World.Atlas;
using World.GameActors.GameObjects;

namespace World.GameActors.Tiles.OnGroundInteractable
{
    public class StepSwitch : DynamicTile, ISwitcherGameActor, IDetectorTile, IAutoupdateable
    {
        public ISwitchableGameActor Switchable { get; set; }

        public StepSwitch(Vector2I position) : base(position) { } 

 		public StepSwitch(Vector2I position, int textureId) : base(position, textureId) { }

        public StepSwitch(Vector2I position, string textureName) : base(position, textureName)
        {
        }

        public void Switch(GameActorPosition gameActorPosition, IAtlas atlas)
        {
            Switchable = Switchable?.SwitchOn(null, atlas);
        }

        public bool RequiresCenterOfObject => false;


        private bool m_wasActive = false;
        private bool m_isActive = false;
        public void ObjectDetected(IGameObject gameObject, IAtlas atlas)
        {
            if (m_wasActive)
            {
                m_isActive = true;
                return;
            }
            m_isActive = true;
            NextUpdateAfter = 1;
            atlas.RegisterToAutoupdate(this);
        }

        public int NextUpdateAfter { get; private set; } = 0;

        private const int DELAY = 8;

        public void Update(IAtlas atlas)
        {
            if (m_isActive)
            {
                Switchable?.SwitchOn(null, atlas);
                m_isActive = false;
                m_wasActive = true;
                NextUpdateAfter = DELAY;
                return;
            }
            Switchable?.SwitchOff(null, atlas);
            NextUpdateAfter = 0;
            m_wasActive = false;
        }
    }
}