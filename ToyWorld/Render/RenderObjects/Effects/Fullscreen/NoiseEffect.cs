using System.IO;
using System.Reflection;

namespace Render.RenderObjects.Effects
{
    internal class NoiseEffect : EffectBase
    {
        public NoiseEffect()
            : base("Basic.vert", "Noise.frag", fragAddendum: GetNoiseSrcStream("noise3D.glsl"))
        { }


        private static Stream GetNoiseSrcStream(string path)
        {
            return Assembly.GetExecutingAssembly().GetManifestResourceStream(ShaderPathBase + path);
        }
    }
}
