using GoodAI.ToyWorld.Control;
using OpenTK.Input;

namespace Render.RenderRequests.Tests
{
    public interface IARRTest : IAvatarRenderRequest
    {
        float MemAddress { get; set; }
    }
}
