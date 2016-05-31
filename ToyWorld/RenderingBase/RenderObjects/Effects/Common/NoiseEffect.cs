using System.IO;
using System.Reflection;
using VRageMath;

namespace RenderingBase.RenderObjects.Effects
{
    public class NoiseEffect : EffectBase
    {
        private enum Uniforms
        {
            // Names must correspond to names as defined in the shaders
            mw,
            mvp,

            sceneTexture,
            viewportSize,
            timeStep,
            variance,
        }


        public NoiseEffect()
            : base(typeof(Uniforms), "Post.Noise.vert", "Post.Noise.frag", fragAddendum: GetAddendumSrcStream("Noise.random.glsl"))
        { }

        private static Stream GetAddendumSrcStream(string path)
        {
            return Assembly.GetExecutingAssembly().GetManifestResourceStream(ShaderPathBase + path);
        }


        public void SceneTextureUniform(int val)
        {
            SetUniform1(base[Uniforms.sceneTexture], val);
        }

        public void ViewportSizeUniform(Vector2I val)
        {
            SetUniform2(base[Uniforms.viewportSize], val);
        }

        public void TimeStepUniform(Vector2 val)
        {
            SetUniform2(base[Uniforms.timeStep], val);
        }

        public void VarianceUniform(float val)
        {
            SetUniform1(base[Uniforms.variance], val);
        }
    }
}
