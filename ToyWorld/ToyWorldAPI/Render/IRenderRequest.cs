namespace GoodAI.ToyWorld.Render
{
    public interface IRenderRequest
    {
        float Size { get; }
        float Position { get; }
        float Resolution { get; }
        float MemAddress { get; }
    }
}
