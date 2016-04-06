using World.GameActors.Tiles;

namespace World.GameActions
{
    public abstract class GameAction
    {
        protected GameAction()
        {
        }
    }

    public class ToUsePickaxe : GameAction
    {
        public float Damage { get; set; }
    }
}