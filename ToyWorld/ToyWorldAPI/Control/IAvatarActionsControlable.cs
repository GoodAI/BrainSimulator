namespace GoodAI.ToyWorld.Control
{
    /// <summary>
    /// 
    /// </summary>
    public interface IControlable
    {
        float Acceleration { get; set; }
        float Rotation { get; set; }
        bool Interact { get; set; }
        bool Use { get; set; }
        bool PickUp { get; set; }
    }
}
