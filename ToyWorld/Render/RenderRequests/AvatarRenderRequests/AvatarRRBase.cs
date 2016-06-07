using System;
using GoodAI.ToyWorld.Control;
using RenderingBase.Renderer;
using RenderingBase.RenderRequests;
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

        public InventoryBackgroundType InventoryBackgroundType { get; set; }

        #endregion


        protected override Matrix GetViewMatrix(Vector3 cameraPos, Vector3? cameraDirection = null, Vector3? up = null)
        {
            return base.GetViewMatrix(cameraPos, cameraDirection, RotateMap ? m_avatarDirection : null);
        }


        public override void Init(RendererBase<ToyWorld> renderer, ToyWorld world)
        {
            InventoryBackgroundType = InventoryBackgroundType.BrownBorder;
            
            base.Init(renderer, world);
        }

        public override void Draw(RendererBase<ToyWorld> renderer, ToyWorld world)
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

        protected override void DrawOverlays(RendererBase<ToyWorld> renderer, ToyWorld world)
        {
            base.DrawOverlays(renderer, world);

            // Compute transform of the center of the inventory
            const float margin = 0.05f;
            Vector2 size = new Vector2(0.08f);
            Vector2 position = Vector2.One - (new Vector2(margin) + size * 0.5f);

            DrawAvatarInventoryTool(renderer, world.GetAvatar(AvatarID), size, position, InventoryBackgroundType);
        }
    }
}
