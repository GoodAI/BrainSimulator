namespace GoodAI.ToyWorldAPI.Tiles
{
    /// <summary>
    /// Objects which interacts with Pickaxe should implement this
    /// </summary>
    public interface IBreakableWithPickaxe
    {
        void BreakItUp(float damage);
    }
}