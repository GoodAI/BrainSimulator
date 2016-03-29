using GoodAI.ToyWorld.Control;
using OpenTK.Input;

namespace Render.RenderRequests
{
    public interface IRRTest : IRenderRequest
    {
        Key WindowKeypressResult { get; }
        float MemAddress { get; set; }
    }
}
