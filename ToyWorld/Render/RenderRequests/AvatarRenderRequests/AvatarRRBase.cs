using System;
using GoodAI.ToyWorld.Control;
using Render.Renderer;
using VRageMath;
using World.ToyWorldCore;

namespace Render.RenderRequests
{
    public abstract class AvatarRRBase : RenderRequest, IAvatarRenderRequest
    {
        private Vector3? m_avatarDirection;

        protected Vector2 RelativePositionV { get; set; }


        protected AvatarRRBase(int avatarID)
        {
            AvatarID = avatarID;
        }


        #region IAvatarRenderRequest overrides

        public int AvatarID { get; protected set; }

        public System.Drawing.PointF RelativePosition
        {
            get { return new System.Drawing.PointF(RelativePositionV.X, RelativePositionV.Y); }
            set { RelativePositionV = (Vector2)value; }
        }

        private bool m_rotateMap;
        public bool RotateMap
        {
            get { return m_rotateMap; }
            set
            {
                m_rotateMap = value;
                m_dirtyParams |= DirtyParams.Size;
            }
        }

        protected override RectangleF ViewV
        {
            get
            {
                RectangleF tmp = base.ViewV;

                if (RotateMap)
                {
                    // Use an extended grid for displaying
                    double diag = tmp.Size.Length();
                    float gridSize = (float)Math.Ceiling(diag);
                    tmp = new RectangleF(Vector2.Zero, new Vector2(gridSize)) { Center = new Vector2(PositionCenterV) };
                }

                return tmp;
            }
        }

        #endregion


        protected override Matrix GetViewMatrix(Vector3 cameraPos, Vector3? cameraDirection = null, Vector3? up = null)
        {
            return base.GetViewMatrix(cameraPos, cameraDirection, RotateMap ? m_avatarDirection : null);
        }


        public override void Init(RendererBase renderer, ToyWorld world)
        {
            base.Init(renderer, world);
        }

        public override void Draw(RendererBase renderer, ToyWorld world)
        {
            PositionCenterV2 += RelativePositionV;

            if (RotateMap)
            {
                var avatar = world.GetAvatar(AvatarID);
                Vector3 dir = Vector3.Up;
                dir.RotateZ(avatar.Rotation);
                m_avatarDirection = dir;
            }

            base.Draw(renderer, world);
        }
    }
}
