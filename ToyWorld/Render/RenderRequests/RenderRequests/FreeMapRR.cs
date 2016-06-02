using System.Drawing;
using GoodAI.ToyWorld.Control;
using RenderingBase.Renderer;
using VRageMath;
using World.ToyWorldCore;

namespace Render.RenderRequests
{
    internal class FreeMapRR : RRBase, IFreeMapRR
    {
        #region Genesis
        #endregion

        #region IFreeMapRR overrides

        // Enable setting of base position (that has only public getter)
        public new PointF PositionCenter { get { return base.PositionCenter; } set { base.PositionCenter = value; } }

        public void SetPositionCenter(float x, float y, float z = 0)
        {
            PositionCenterV = new Vector3(x, y, z);
        }

        #endregion

        #region RenderRequestBase overrides

        public override void Init(RendererBase<ToyWorld> renderer, ToyWorld world)
        {
            SizeV = (Vector2)world.Size;
            PositionCenterV2 = SizeV * 0.5f;

            base.Init(renderer, world);
        }

        public override void Draw(RendererBase<ToyWorld> renderer, ToyWorld world)
        {
            // TODO: setup camera

            base.Draw(renderer, world);
        }

        #endregion
    }
}
