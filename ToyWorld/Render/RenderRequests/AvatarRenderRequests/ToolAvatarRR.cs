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
        }

        public override void Draw(RendererBase<ToyWorld> renderer, ToyWorld world)
        {
            var avatar = world.GetAvatar(AvatarID);

            // Compute transform of the center of the inventory
            const float margin = 0.05f;
            Vector2 position = Vector2.One - (new Vector2(margin) + SizeV * 0.5f);

            DrawAvatarTool(renderer, avatar, SizeV, position, ToolBackgroundType);
        }

        #endregion
    }
}
