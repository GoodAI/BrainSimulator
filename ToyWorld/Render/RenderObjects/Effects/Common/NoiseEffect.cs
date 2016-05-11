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

            sceneTexture,
            timeMean,
        }


        public NoiseEffect()
            : base(typeof(Uniforms), "Post.Noise.vert", "Post.Noise.frag", fragAddendum: GetSmokeSrcStream("Noise.perlinNoise3D.glsl"))
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


        public void SceneTextureUniform(int val)
        {
            SetUniform1(base[Uniforms.sceneTexture], val);
        }

        public void TimeMeanUniform(Vector4 val)
        {
            SetUniform4(base[Uniforms.timeMean], val);
        }
    }
}
