using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using VRage.Collections;

namespace Render.Renderer
{
    internal class RenderMessageQueue : MyQueue<RenderMessageBase>
    {
        public RenderMessageQueue(int capacity)
            : base(capacity)
        { }

        public RenderMessageQueue(IEnumerable<RenderMessageBase> collection)
            : base(collection)
        { }
    }
}
