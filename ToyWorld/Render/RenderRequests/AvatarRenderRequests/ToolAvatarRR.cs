using System.Drawing;
using GoodAI.ToyWorld.Control;
using RenderingBase.Renderer;
using VRageMath;
using World.ToyWorldCore;

namespace Render.RenderRequests
{
    internal class ToolAvatarRR : AvatarRRBase, IToolAvatarRR
    {
        #region Genesis

        public ToolAvatarRR(int avatarID)
            : base(avatarID)
        { }

        #endregion

        #region IToolAvatarRR overrides
        #endregion

        #region RenderRequestBase overrides

        public override void Init()
        {
            SizeV = new Vector2(0.9f);

            MultisampleLevel = RenderRequestMultisampleLevel.None;

            if (EffectRenderer.Settings != null)
                EffectRenderer.Settings.EnabledEffects = RenderRequestEffect.None;
            if (PostprocessRenderer.Settings != null)
                PostprocessRenderer.Settings.EnabledPostprocessing = RenderRequestPostprocessing.None;

            OverlayRenderer.Settings = new AvatarRROverlaySettings
            {
                EnabledOverlays = AvatarRenderRequestOverlay.InventoryTool,
                ToolSize = new PointF(SizeV.X, SizeV.Y),
                ToolPosition = new PointF(),
            };

            base.Init();
        }

        public override void Draw()
        {
            base.Draw();
        }

        protected override Matrix GetViewMatrix(Vector3 cameraPos, Vector3? cameraDirection = null, Vector3? up = null)
        {
            return Matrix.Identity;
        }

        protected override void DrawTileLayers(ToyWorld world)
        { }

        protected override void DrawObjectLayers(ToyWorld world)
        { }

        #endregion
    }
}
