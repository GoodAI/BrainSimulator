namespace GoodAI.ToyWorld.Control
{
    /// <summary>
    /// 
    /// </summary>
    public interface IAvatarPriorityActions : IAvatarActions
    {
        new AvatarAction<float> Forward { get; set; }
        new AvatarAction<float> Back { get; set; }

        new AvatarAction<bool> DoPickup { get; set; }
    }
}
