using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.ToyWorld.Render;
using GoodAI.ToyWorld.Render.RenderRequests;

namespace Render.Renderer
{
    public interface IRenderRequestTest : IRenderRequest
    {
        float MemAddress { get; set; }
    }
}
