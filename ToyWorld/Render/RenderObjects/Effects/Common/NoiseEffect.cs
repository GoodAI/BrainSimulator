using System.IO;
using System.Reflection;
using VRageMath;

namespace Render.RenderObjects.Effects
{
    internal class NoiseEffect : EffectBase
    {
        private enum Uniforms
        {
            // Names must correspond to names as defined in the shaders
            mw,
            mvp,

            noiseColor,
            timeMean,
        }


        public NoiseEffect()
            : base(typeof(Uniforms), "Noise.vert", "Noise.frag", fragAddendum: GetNoiseSrcStream("Noise.perlinNoise3D.glsl"))
        { }

        private static Stream GetNoiseSrcStream(string path)
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


        public void NoiseColorUniform(Vector4 val)
        {
            SetUniform4(base[Uniforms.noiseColor], val);
        }

        public void TimeMeanUniform(Vector4 val)
        {
            SetUniform4(base[Uniforms.timeMean], val);
        }
    }
}
