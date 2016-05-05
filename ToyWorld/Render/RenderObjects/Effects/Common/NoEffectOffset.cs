using System;
using VRageMath;

namespace Render.RenderObjects.Effects
{
    internal class NoEffectOffset : EffectBase
    {
        private enum Uniforms
        {
            // Names must correspond to names as defined in the shaders
            texSizeCount,
            tileSizeMargin,
            tileBorder,
            mvp,

            tex,
        }


        public NoEffectOffset()
            : base(typeof(Uniforms), "BasicOffset.vert", "BasicOffset.frag")
        { }


        public void TexSizeCountUniform(Vector3I val)
        {
            SetUniform3(base[Uniforms.texSizeCount], val);
        }

        public void TileSizeMarginUniform(Vector4I val)
        {
            SetUniform4(base[Uniforms.tileSizeMargin], val);
        }

        public void TileBorderUniform(Vector2I val)
        {
            SetUniform2(base[Uniforms.tileBorder], val);
        }

        public void ModelViewProjectionUniform(ref Matrix val)
        {
            SetUniformMatrix4(base[Uniforms.mvp], val);
        }

        
        public void TextureUniform(int val)
        {
            SetUniform1(base[Uniforms.tex], val);
        }
    }
}
