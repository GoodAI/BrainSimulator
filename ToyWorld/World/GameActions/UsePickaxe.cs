using World.GameActors;
using World.GameActors.Tiles;
using World.ToyWorldCore;

namespace World.GameActions
{
    public class UsePickaxe : GameAction
    {
        public float Damage { get; set; }

        public UsePickaxe(GameActor sender) : base(sender) { }

        public override void Resolve(GameActorPosition target, IAtlas atlas, ITilesetTable table)
        {
            throw new System.NotImplementedException();
        }
    }
}