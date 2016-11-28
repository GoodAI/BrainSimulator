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

            color,
            intensityDecay,
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


        public void ColorUniform(Vector4 val)
        {
            SetUniform4(base[Uniforms.color], val);
        }

        public void IntensityDecayUniform(Vector2 val)
        {
            SetUniform2(base[Uniforms.intensityDecay], val);
        }

        public void LightPosUniform(Vector3 val)
        {
            SetUniform3(base[Uniforms.lightPos], val);
        }
    }
}
