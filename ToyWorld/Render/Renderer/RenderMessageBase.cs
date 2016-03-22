using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.ToyWorld.Render;

namespace Render.Renderer
{
    internal abstract class RenderMessageBase : IRenderRequest
    {
        public abstract float Size { get; set; }
        public abstract float Position { get; set; }
        public abstract float Resolution { get; set; }
        public abstract float MemAddress { get; set; }
    }
}
