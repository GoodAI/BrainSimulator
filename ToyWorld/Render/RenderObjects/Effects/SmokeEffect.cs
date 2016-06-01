using System.IO;
using System.Reflection;
using RenderingBase.RenderObjects.Effects;
using VRageMath;

namespace Render.RenderObjects.Effects
{
    public class SmokeEffect : NoEffect
    {
        private enum Uniforms
        {
            // Names must correspond to names as defined in the shaders
            mw,
            mvp,

            smokeColor,
            timeStep,
            meanScale,
        }


        public SmokeEffect()
            : base(typeof(Uniforms), "Post.Smoke.vert", "Post.Smoke.frag", fragAddendum: "Noise.simplexNoise3D.glsl")
        { }


        public void ModelWorldUniform(ref Matrix val)
        {
            SetUniformMatrix4(base[Uniforms.mw], val);
        }

        public void ModelViewProjectionUniform(ref Matrix val)
        {
            SetUniformMatrix4(base[Uniforms.mvp], val);
        }


        public void SmokeColorUniform(Vector4 val)
        {
            SetUniform4(base[Uniforms.smokeColor], val);
        }

        public void TimeStepUniform(Vector2 val)
        {
            SetUniform2(base[Uniforms.timeStep], val);
        }

        public void MeanScaleUniform(Vector2 val)
        {
            SetUniform2(base[Uniforms.meanScale], val);
        }
    }
}
