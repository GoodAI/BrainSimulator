namespace GoodAI.ToyWorld.Control
{
    /// <summary>
    /// 
    /// </summary>
    public interface IAvatarActions
    {
        float Forward { get; set; }
        float Back { get; set; }

        bool DoPickup { get; set; }
    }
}
