namespace World.GameActions
{
    public abstract class GameAction
    {
    }

    public class ToUsePickaxe : GameAction
    {
        public float Damage { get; set; }
    }
}
