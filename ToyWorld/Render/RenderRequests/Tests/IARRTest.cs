using GoodAI.ToyWorld.Control;
using OpenTK.Input;

namespace Render.RenderRequests.Tests
{
    public interface IARRTest : IAvatarRenderRequest
    {
        Key WindowKeypressResult { get; }
        float MemAddress { get; set; }
    }
}
