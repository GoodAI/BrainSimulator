using System.Diagnostics;
using System.Drawing;
using GoodAI.ToyWorld.Control;
using VRageMath;

namespace Render.RenderRequests
{
    internal class ToolAvatarRR : AvatarRRBase, IToolAvatarRR
    {
        #region Genesis

        public ToolAvatarRR(int avatarID)
            : base(avatarID)
        {
            MultisampleLevel = RenderRequestMultisampleLevel.None;
        }

        #endregion

        #region IToolAvatarRR overrides
        #endregion

        #region RenderRequestBase overrides

        public override void Init()
        {
            var objects = GameObjects;
            objects.Use3D = false;
            objects.EnabledEffects = RenderRequestGameObject.None;
            GameObjects = objects;

            var effects = Effects;
            effects.EnabledEffects = RenderRequestEffect.None;
            Effects = effects;

            var post = Postprocessing;
            post.EnabledPostprocessing = RenderRequestPostprocessing.None;
            Postprocessing = post;

            SizeV = new Vector2(0.9f);
            var overlay = Overlay;
            overlay.EnabledOverlays |= RenderRequestOverlay.InventoryTool;
            overlay.ToolSize = new PointF(SizeV.X, SizeV.Y);
            overlay.ToolPosition = new PointF();
            Overlay = overlay;

            Debug.Assert(!Image.CopyDepth, "Depth information for inventory overlay seems useless...");

            base.Init();
        }

        protected override Matrix Get2DViewMatrix(Vector3 cameraPos, Vector3? up = null)
        {
            return Matrix.Identity;
        }

        protected override Matrix Get3DViewMatrix(Vector3 cameraPos, Vector3? cameraDirection = null, Vector3? up = null)
        {
            return Matrix.Identity;
        }

        #endregion
    }
}
