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

        public void ObjectDetected(IGameObject gameObject, IAtlas atlas, ITilesetTable tilesetTable)
        {
            Switchable = Switchable?.SwitchOn(null, atlas, tilesetTable);
            NextUpdateAfter = 1;
        }

        public int NextUpdateAfter { get; private set; } = 1;

        public void Update(IAtlas atlas, ITilesetTable table)
        {
            Switchable = Switchable?.SwitchOff(null, atlas, table);
            NextUpdateAfter = 0;
        }
    }
}