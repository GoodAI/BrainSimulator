using System;
using GoodAI.ToyWorld.Control;
using RenderingBase.Renderer;
using VRageMath;
using World.ToyWorldCore;

namespace Render.RenderRequests
{
    public abstract class AvatarRRBase
        : RenderRequest, IAvatarRenderRequest
    {
        // Internal class that replaces the base renderer, adding new features
        internal class ARROverlayRenderer
            : OverlayRenderer
        {
            internal new AvatarRRBase Owner { get { return (AvatarRRBase)base.Owner; } }
            internal new AvatarRROverlaySettings Settings { get { return (AvatarRROverlaySettings)base.Settings; } set { base.Settings = value; } }

            public ARROverlayRenderer(RenderRequest owner)
                : base(owner)
            { }

            public override void Draw(RendererBase<ToyWorld> renderer, ToyWorld world)
            {
                base.Draw(renderer, world);

                DrawAvatarTool(
                    renderer, world.GetAvatar(Owner.AvatarID),
                    (Vector2)Settings.ToolSize, (Vector2)Settings.ToolPosition,
                    Settings.ToolBackground);
            }
        }

        internal new ARROverlayRenderer OverlayRenderer
        {
            get { return (ARROverlayRenderer)base.OverlayRenderer; }
            set { base.OverlayRenderer = value; }
        }


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
                DirtyParams |= DirtyParam.Size;
            }
        }

        protected internal override RectangleF ViewV
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


        public new AvatarRROverlaySettings Overlay { get; set; }

        #endregion


        protected override Matrix GetViewMatrix(Vector3 cameraPos, Vector3? cameraDirection = null, Vector3? up = null)
        {
            return base.GetViewMatrix(cameraPos, cameraDirection, RotateMap ? m_avatarDirection : null);
        }


        public override void Init()
        {
            base.Init();

            OverlayRenderer = new ARROverlayRenderer(this);
        }

        public override void Draw()
        {
            PositionCenterV2 += RelativePositionV;

            if (RotateMap)
            {
                var avatar = World.GetAvatar(AvatarID);
                Vector3 dir = Vector3.Up;
                dir.RotateZ(avatar.Rotation);
                m_avatarDirection = dir;
            }

            base.Draw();
        }
    }
}
