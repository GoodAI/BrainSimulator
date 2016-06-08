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

        public override void Init(RendererBase<ToyWorld> renderer, ToyWorld world)
        {
            SizeV = new Vector2(0.9f);

            MultisampleLevel = RenderRequestMultisampleLevel.None;
            DrawOverlay = true;
            EnableDayAndNightCycle = false;

            base.Init(renderer, world);
        }

        public override void Draw(RendererBase<ToyWorld> renderer, ToyWorld world)
        {
            DirtyParams &= DirtyParam.Size | DirtyParam.Resolution | DirtyParam.Image | DirtyParam.Overlay;

            base.Draw(renderer, world);
        }

        protected override Matrix GetViewMatrix(Vector3 cameraPos, Vector3? cameraDirection = null, Vector3? up = null)
        {
            return Matrix.Identity;
        }

        protected override void DrawTileLayers(ToyWorld world)
        { }

        protected override void DrawObjectLayers(ToyWorld world)
        { }

        protected override void DrawEffects(RendererBase<ToyWorld> renderer, ToyWorld world)
        { }

        protected override void ApplyPostProcessingEffects(RendererBase<ToyWorld> renderer)
        { }

        protected override void DrawOverlays(RendererBase<ToyWorld> renderer, ToyWorld world)
        {
            var avatar = world.GetAvatar(AvatarID);


            DrawAvatarTool(renderer, avatar, SizeV, Vector2.Zero, ToolBackgroundType);
        }

        #endregion
    }
}
