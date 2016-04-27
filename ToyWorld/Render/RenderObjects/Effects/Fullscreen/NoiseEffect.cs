using System.IO;
using System.Reflection;

namespace Render.RenderObjects.Effects
{
    internal class NoiseEffect : EffectBase
    {
        enum MyEnum
        {

        }
        public NoiseEffect()
            : base(typeof(MyEnum), "Noise.vert", "Noise.frag", fragAddendum: GetNoiseSrcStream("Noise.perlinNoise3D.glsl"))
        { }


        private static Stream GetNoiseSrcStream(string path)
        {
            return Assembly.GetExecutingAssembly().GetManifestResourceStream(ShaderPathBase + path);
        }
    }
}
