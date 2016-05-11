using System.IO;
using System.Reflection;
using VRageMath;

namespace Render.RenderObjects.Effects
{
    internal class SmokeEffect : EffectBase
    {
        private enum Uniforms
        {
            // Names must correspond to names as defined in the shaders
            mw,
            mvp,

            smokeColor,
            timeMeanScale,
        }


        public SmokeEffect()
            : base(typeof(Uniforms), "Post.Smoke.vert", "Post.Smoke.frag", fragAddendum: GetSmokeSrcStream("Noise.simplexNoise3D.glsl"))
        { }

        private static Stream GetSmokeSrcStream(string path)
        {
            return Assembly.GetExecutingAssembly().GetManifestResourceStream(ShaderPathBase + path);
        }


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

        public void TimeMeanUniform(Vector4 val)
        {
            SetUniform4(base[Uniforms.timeMeanScale], val);
        }
    }
}
