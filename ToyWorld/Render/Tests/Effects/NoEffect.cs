using Render.RenderObjects.Shaders;

namespace Render.Tests.Effects
{
    internal class NoEffect : EffectBase
    {
        public NoEffect()
            : base("Basic.vert", "Basic.frag")
        { }
    }
}
