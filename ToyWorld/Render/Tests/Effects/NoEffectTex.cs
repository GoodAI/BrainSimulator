using Render.RenderObjects.Shaders;

namespace Render.Tests.Effects
{
    internal class NoEffectTex : EffectBase
    {
        public NoEffectTex()
            : base("BasicTex.vert", "BasicTex.frag")
        { }
    }
}
