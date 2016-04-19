using System.Drawing;
using GoodAI.ToyWorld.Control;
using Render.Renderer;
using VRageMath;
using World.ToyWorldCore;

namespace Render.RenderRequests
{
    internal class FreeMapRR : RenderRequestBase, IFreeMapRR
    {
        #region Genesis

        #endregion

        #region IFreeMapRR overrides

        public new PointF PositionCenter { get { return base.PositionCenter; } set { base.PositionCenter = value; } }

        #endregion

        #region RenderRequestBase overrides

        public override void Init(RendererBase renderer, ToyWorld world)
        {
            SizeV = (Vector2)world.Size;
            SizeV = Vector2.One * 4;
            PositionCenterV = new Vector3(SizeV, 0) * 0.5f;

            base.Init(renderer, world);
        }

        public override void Draw(RendererBase renderer, ToyWorld world)
        {
            // TODO: setup camera

            base.Draw(renderer, world);
        }

        #endregion
    }
}
