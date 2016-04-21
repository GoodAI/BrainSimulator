using System;
using GoodAI.ToyWorld.Control;
using Render.Renderer;
using VRageMath;
using World.ToyWorldCore;

namespace Render.RenderRequests
{
    internal class FofAvatarRR : AvatarRRBase, IFofAvatarRR
    {
        private IFovAvatarRR m_fovAvatarRenderRequest;

        #region Genesis

        public FofAvatarRR(int avatarID)
            : base(avatarID)
        { }

        #endregion

        #region IFofAvatarRR overrides

        public IFovAvatarRR FovAvatarRenderRequest
        {
            get { return m_fovAvatarRenderRequest; }
            set
            {
                if (value == null)
                    throw new ArgumentNullException("value", "The supplied IFovAvatarRR cannot be null.");
                if (value.AvatarID != AvatarID)
                    throw new ArgumentException("The supplied IFovAvatarRR is tied to a different avatarID. Fof/Fov ID: " + AvatarID + '/' + value.AvatarID);

                m_fovAvatarRenderRequest = value;
            }
        }

        #endregion

        #region RenderRequestBase overrides

        public override void Init(RendererBase renderer, ToyWorld world)
        {
            SizeV = new Vector2(3, 3);

            base.Init(renderer, world);
        }

        public override void Draw(RendererBase renderer, ToyWorld world)
        {
            if (FovAvatarRenderRequest == null)
                throw new MissingFieldException("Missing the IFovAvatarRR. Please specify one before using this render request.");

            // Setup params
            var avatar = world.GetAvatar(AvatarID);
            PositionCenterV2 =
                avatar.Position
                // Offset so that the FofOffset interval (-1,1) spans the entire Fov view and doesn't reach outside of it
            + ((Vector2)FovAvatarRenderRequest.Size - SizeV) / 2
            * (Vector2)avatar.Fof;

            base.Draw(renderer, world);
        }

        #endregion

    }
}
