using World.GameActors;

namespace World.GameActions
{
    public class UsePickaxe : GameAction
    {
        public float Damage { get; set; }

        public UsePickaxe(GameActor sender) : base(sender) { }
    }
}