using System;
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

        protected override Matrix Get3DViewMatrix(Vector3 cameraPos, Vector3? cameraDirection = null, Vector3? up = null)
        {
            return base.Get3DViewMatrix(cameraPos, cameraDirection, Vector3.Up);
        }

        public override void Update()
        {
            float sideSize = SizeV.X - 1;
            PositionZ = (float)Math.Sqrt(sideSize * sideSize * 7 / 4);
            base.Update();
        }

        public override void Draw()
        {
            base.Draw();
        }

        #endregion
    }
}
