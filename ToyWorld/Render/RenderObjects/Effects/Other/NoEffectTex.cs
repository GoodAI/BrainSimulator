namespace Render.RenderObjects.Effects
{
    internal class NoEffectTex : EffectBase
    {
        enum MyEnum
        {

        }
        public NoEffectTex()
            : base(typeof(MyEnum), "BasicTex.vert", "BasicTex.frag")
        { }
    }
}
