using System;

namespace Render.RenderObjects.Effects
{
    internal class NoEffectOffset : EffectBase
    {
        private enum Uniforms
        {
            tex,
            texSizeCount,
            tileSizeMargin,
            mvp,
        }


        public NoEffectOffset()
            : base(typeof(Uniforms), "BasicOffset.vert", "BasicOffset.frag")
        {
        }


        public void TexUniform(float val)
        {

        }
    }
}
