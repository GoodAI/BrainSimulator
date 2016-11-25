using System.Collections.Generic;
using VRageMath;
using World.Atlas;
using World.Atlas.Layers;
using World.ToyWorldCore;

namespace World.GameActors.Tiles.OnGroundInteractable
{
    class FireplaceBurning : DynamicTile, IHeatSource, IAutoupdateableGameActor
    {
        private int m_counter;
        private const float MAX_HEAT = 4;
        public float Heat { get; private set; }
        public float MaxDistance { get; private set; }
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
            MaxDistance = 1.5f;
            m_counter = 0;
        }

        public void Update(IAtlas atlas, ITilesetTable table)
        {
            if (Heat < 0)
            {
                // first update - fire starts
                Heat = 0.2f;
                atlas.RegisterHeatSource(this);
                NextUpdateAfter = 60;
            }
            if (Heat >= MAX_HEAT && m_counter > 1000)
            {
                // fourth update - fire is extinguished.
                Heat = 0;
                NextUpdateAfter = 0;
                var fireplace = new Fireplace(table, Position);
                atlas.UnregisterHeatSource(this);
                atlas.ReplaceWith(ThisGameActorPosition(LayerType.OnGroundInteractable), fireplace);
                return;
            }
            if (Heat < MAX_HEAT)
            {
                // second update - fire is growing
                Heat += 0.4f;
            }
            if(Heat >= MAX_HEAT)
            {
                // third update - fire is stable
                IEnumerable<GameActorPosition> gameActorPositions = atlas.ActorsAt((Vector2) Position, LayerType.All);
                foreach (GameActorPosition gameActorPosition in gameActorPositions)
                {
                    ICombustibleGameActor combustible = gameActorPosition.Actor as ICombustibleGameActor;
                    if (combustible != null)
                    {
                        combustible.Burn(gameActorPosition, atlas, table);
                    }
                }
                m_counter++;
            }
        }
    }
}
