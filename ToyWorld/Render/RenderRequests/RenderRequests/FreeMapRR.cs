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

        public override void Init()
        {
            SizeV = (Vector2)World.Size;
            PositionCenterV2 = SizeV * 0.5f;

            base.Init();
        }

        public override void Update()
        {
            // TODO: setup camera

            base.Update();
        }

        public override void Draw()
        {
            base.Draw();
        }

        #endregion
    }
}
