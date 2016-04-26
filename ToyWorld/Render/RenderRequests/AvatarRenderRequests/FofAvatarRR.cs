using System;
using System.Drawing;
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

        public override SizeF Size
        {
            get { return base.Size; }
            set
            {
                base.Size = new SizeF(Math.Min(value.Width, FovAvatarRenderRequest.Size.Width), Math.Min(value.Height, FovAvatarRenderRequest.Size.Height));
            }
        }

        public IFovAvatarRR FovAvatarRenderRequest
        {
            get { return m_fovAvatarRenderRequest; }
            set
            {
                if (value == null)
                    throw new ArgumentNullException("value", "The supplied IFovAvatarRR cannot be null.");
                if (value.AvatarID != AvatarID)
                    throw new ArgumentException("The supplied IFovAvatarRR is tied to a different avatarID. Fof/Fov ID: " + AvatarID + '/' + value.AvatarID);
                if (value.Size.Width < Size.Width || value.Size.Height < Size.Height)
                    throw new ArgumentException("The supplied IFovAvatarRR's view size cannot be smaller than this view size. Fof/Fov sizes: " + Size + '/' + value.Size);

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
