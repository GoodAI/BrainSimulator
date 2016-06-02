using RenderingBase.RenderObjects.Effects;

namespace Render.RenderObjects.Effects
{
    internal class NoEffectTex : NoEffect
    {
        private enum Uniforms
        {
            tex,
        }

        public NoEffectTex()
            : base(typeof(Uniforms), "BasicTex.vert", "BasicTex.frag")
        { }


        public void TextureUniform(int val)
        {
            SetUniform1(base[Uniforms.tex], val);
        }
    }
}
