using VRageMath;
using World.Atlas;
using World.GameActors.GameObjects;
using World.ToyWorldCore;

namespace World.GameActors.Tiles.OnGroundInteractable
{
    public class StepSwitch : DynamicTile, ISwitcherGameActor, IDetectorTile, IAutoupdateableGameActor
    {
        public ISwitchableGameActor Switchable { get; set; }

        public StepSwitch(ITilesetTable tilesetTable, Vector2I position)
            : base(tilesetTable, position)
        {
        }

        public StepSwitch(int tileType, Vector2I position)
            : base(tileType, position)
        {
        }

        public void Switch(GameActorPosition gameActorPosition, IAtlas atlas, ITilesetTable table)
        {
            Switchable = Switchable?.SwitchOn(null, atlas, table);
        }

        public bool RequiresCenterOfObject => false;


        private bool m_wasActive = false;
        private bool m_isActive = false;
        public void ObjectDetected(IGameObject gameObject, IAtlas atlas, ITilesetTable tilesetTable)
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

        public void Update(IAtlas atlas, ITilesetTable table)
        {
            if (m_isActive)
            {
                Switchable?.SwitchOn(null, atlas, table);
                m_isActive = false;
                m_wasActive = true;
                NextUpdateAfter = DELAY;
                return;
            }
            Switchable?.SwitchOff(null, atlas, table);
            NextUpdateAfter = 0;
            m_wasActive = false;
        }
    }
}