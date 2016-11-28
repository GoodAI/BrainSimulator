using System.Drawing;
using GoodAI.ToyWorld.Control;
using OpenTK.Graphics.OpenGL;
using RenderingBase.RenderObjects.Geometries;
using RenderingBase.Renderer;
using VRageMath;
using World.ToyWorldCore;
using RectangleF = VRageMath.RectangleF;

namespace Render.RenderRequests
{
    internal class FullMapRR : RRBase, IFullMapRR
    {
        #region Genesis
        #endregion

        #region IFullMapRR overrides
        #endregion

        #region RenderRequestBase overrides

        public override void Init()
        {
            SizeV = (Vector2)World.Size;
            PositionCenterV2 = SizeV * 0.5f;

            base.Init();
        }

        #endregion
    }
}
