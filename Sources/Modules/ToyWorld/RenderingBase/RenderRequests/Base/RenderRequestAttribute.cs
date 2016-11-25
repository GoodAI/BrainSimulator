using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.ToyWorld.Control;

namespace RenderingBase.RenderRequests
{
    [AttributeUsage(AttributeTargets.Class)]
    class RenderRequestAttribute : Attribute
    {
        public readonly Type BaseRenderRequestType;

        public RenderRequestAttribute(Type baseRenderRequestType)
        {
            BaseRenderRequestType = baseRenderRequestType;
        }
    }
}
