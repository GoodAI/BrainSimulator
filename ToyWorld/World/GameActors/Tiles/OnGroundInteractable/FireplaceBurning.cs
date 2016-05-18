using VRageMath;
using World.ToyWorldCore;

namespace World.GameActors.Tiles.OnGroundInteractable
{
    class FireplaceBurning : DynamicTile, IHeatSource, IAutoupdateable
    {
        public float Heat { get; private set; }
        public int NextUpdateAfter { get; private set; }

        public FireplaceBurning(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable, position)
        {
            Init();
        }

        public FireplaceBurning(int tileType, Vector2I position) : base(tileType, position)
        {
            Init();
        }

        private void Init()
        {
            NextUpdateAfter = 1;
            Heat = -1f;
        }

        public void Update(IAtlas atlas, ITilesetTable table)
        {
            if (Heat < 0)
            {
                Heat = 0;
                atlas.RegisterHeatSource(this);
                NextUpdateAfter = 60;
            }
            if (Heat >= 1)
            {
                Heat = 0;
                NextUpdateAfter = 0;
                var fireplace = new Fireplace(table, Position);
                atlas.UnregisterHeatSource(this);
                atlas.ReplaceWith(ThisGameActorPosition(LayerType.OnGroundInteractable), fireplace);
                return;
            }
            if (Heat < 1)
            {
                Heat += 0.1f;
            }
            if(Heat >= 1)
            {
                NextUpdateAfter = 1000;
            }
        }
    }
}
