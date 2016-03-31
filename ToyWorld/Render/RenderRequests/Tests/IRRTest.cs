using GoodAI.ToyWorld.Control;
using OpenTK.Input;

namespace Render.RenderRequests.Tests
{
    public interface IRRTest : IRenderRequest
    {
        float MemAddress { get; set; }
    }
}
