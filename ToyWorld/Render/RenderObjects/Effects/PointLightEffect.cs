using System.IO;
using System.Reflection;
using RenderingBase.RenderObjects.Effects;
using VRageMath;

namespace Render.RenderObjects.Effects
{
    public class PointLightEffect : NoEffect
    {
        private enum Uniforms
        {
            // Names must correspond to names as defined in the shaders
            mw,
            mvp,

            colorIntensity,
            decay,
            lightPos,
        }


        public PointLightEffect()
            : base(typeof(Uniforms), "Post.Smoke.vert", "Light.PointLight.frag")
        { }


        public void ModelWorldUniform(ref Matrix val)
        {
            SetUniformMatrix4(base[Uniforms.mw], val);
        }

        public void ModelViewProjectionUniform(ref Matrix val)
        {
            SetUniformMatrix4(base[Uniforms.mvp], val);
        }


        public void ColorIntensityUniform(Vector4 val)
        {
            SetUniform4(base[Uniforms.colorIntensity], val);
        }

        public void DecayUniform(float val)
        {
            SetUniform1(base[Uniforms.decay], val);
        }

        public void LightPosUniform(Vector3 val)
        {
            SetUniform3(base[Uniforms.lightPos], val);
        }
    }
}
