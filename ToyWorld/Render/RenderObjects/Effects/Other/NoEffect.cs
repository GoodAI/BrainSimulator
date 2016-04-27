namespace Render.RenderObjects.Effects
{
    internal class NoEffect : EffectBase
    {
        enum MyEnum
        {

        }
        public NoEffect()
            : base(typeof(MyEnum), "BasicColor.vert", "BasicColor.frag")
        { }
    }
}
