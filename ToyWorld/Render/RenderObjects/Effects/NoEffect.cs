using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using RenderingBase.RenderObjects.Effects;
using VRageMath;

namespace Render.RenderObjects.Effects
{
    public class NoEffect : EffectBase
    {
        private const string ShaderPathBase = "Render.RenderObjects.Effects.Src.";


        protected NoEffect(Type uniformNamesEnumType, string vertPath, string fragPath, string vertAddendum = null, string fragAddendum = null)
            : base(uniformNamesEnumType, GetSrcStream(vertPath), GetSrcStream(fragPath), GetSrcStream(vertAddendum), GetSrcStream(fragAddendum))
        { }


        private static Stream GetSrcStream(string path)
        {
            if (string.IsNullOrEmpty(path))
                return null;

            Stream sourceStream = Assembly.GetExecutingAssembly().GetManifestResourceStream(ShaderPathBase + path);
            Debug.Assert(sourceStream != null);
            return sourceStream;
        }
    }
}
