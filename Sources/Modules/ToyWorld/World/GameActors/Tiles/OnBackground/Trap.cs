using System.Collections.Generic;
using System.Linq;
using VRageMath;
using World.GameActors.GameObjects;
using World.ToyWorldCore;

namespace World.GameActors.Tiles.OnBackground
{
    public class Trap : DynamicTile, IAutoupdateable
    {
        private const float ENERGY_FOR_STEP_ON_TRAP = 0.1f;

        private bool Charged { get; set; }

        public int NextUpdateAfter
        {
            get { return 1; }
        }

        public Trap(ITilesetTable tilesetTable, Vector2I position) : base(tilesetTable, position)
        {
        }

        public Trap(int tileType, Vector2I position) : base(tileType, position)
        {
        }


        public void Update(IAtlas atlas)
        {
            List<IGameObject> stayingOnTile = atlas.StayingOnTile(Position);
            if (stayingOnTile.Count == 0)
            {
                Charged = true;
            }
            if (Charged)
            {
                var avatars = stayingOnTile.OfType<IAvatar>();
                avatars.ForEach(x => x.Energy -= ENERGY_FOR_STEP_ON_TRAP);
                Charged = false;
            }
        }
    }
}