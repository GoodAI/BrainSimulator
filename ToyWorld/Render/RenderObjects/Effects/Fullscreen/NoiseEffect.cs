using System.IO;
using System.Reflection;

namespace Render.RenderObjects.Effects
{
    internal class NoiseEffect : EffectBase
    {
        public NoiseEffect()
            : base("Noise.vert", "Noise.frag", fragAddendum: GetNoiseSrcStream("Noise.perlinNoise3D.glsl"))
        { }


        private static Stream GetNoiseSrcStream(string path)
        {
            return Assembly.GetExecutingAssembly().GetManifestResourceStream(ShaderPathBase + path);
        }
    }
}
