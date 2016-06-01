using System.IO;
using System.Reflection;
using RenderingBase.RenderObjects.Effects;
using VRageMath;

namespace Render.RenderObjects.Effects
{
    public class NoiseEffect : NoEffect
    {
        private enum Uniforms
        {
            // Names must correspond to names as defined in the shaders

            sceneTexture,
            viewportSize,
            timeStep,
            variance,
        }


        public NoiseEffect()
            : base(typeof(Uniforms), "Post.Noise.vert", "Post.Noise.frag", fragAddendum: "Noise.random.glsl")
        { }


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
