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
        #region Internal overlay class

        // Internal class that replaces the base renderer, adding new features
        internal class ARROverlayRenderer
            : OverlayRenderer
        {
            internal new AvatarRRBase Owner { get { return (AvatarRRBase)base.Owner; } }

            public ARROverlayRenderer(RenderRequest owner)
                : base(owner)
            { }

            public override void Init(RendererBase<ToyWorld> renderer, ToyWorld world, OverlaySettings settings)
            {
                base.Init(renderer, world, settings);
            }

            public override void Draw(RendererBase<ToyWorld> renderer, ToyWorld world)
            {
                if (Settings.EnabledOverlays == RenderRequestOverlay.None)
                    return;

                base.Draw(renderer, world);

                if (Settings.EnabledOverlays.HasFlag(RenderRequestOverlay.InventoryTool))
                    DrawAvatarTool(
                        renderer, world.GetAvatar(Owner.AvatarID),
                        (Vector2)Settings.ToolSize, (Vector2)Settings.ToolPosition,
                        Settings.ToolBackground);
            }
        }

        #endregion


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
            OverlayRenderer = new ARROverlayRenderer(this);
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

        #endregion


        public override void Init()
        {
            // Must use rotateMap when using 3D
            RotateMap = GameObjects.Use3D || RotateMap;

            base.Init();
        }

        protected override Matrix Get2DViewMatrix(Vector3 cameraPos, Vector3? up = null)
        {
            return base.Get2DViewMatrix(cameraPos, RotateMap ? m_avatarDirection : up);
        }

        protected override Matrix Get3DViewMatrix(Vector3 cameraPos, Vector3? cameraDirection = null, Vector3? up = null)
        {
            return base.Get3DViewMatrix(cameraPos, RotateMap ? m_avatarDirection : cameraDirection, Vector3.Backward);
        }

        public override void Update()
        {
            PositionCenterV2 += RelativePositionV;

            var avatar = World.GetAvatar(AvatarID);

            if (RotateMap)
            {
                //Vector3 dir = Vector3.Up;
                Vector3 dir = new Vector3(0, 5.5f, -1);
                dir.RotateZ(avatar.Rotation);
                m_avatarDirection = dir;
            }

            base.Update(); // Sets up view projection -- we need to change grid position AFTER that

            // Apply grid offset to increase view distance
            if (GameObjects.Use3D && m_avatarDirection.HasValue)
            {
                Vector2 gridOffsetDir = new Vector2(m_avatarDirection.Value);
                gridOffsetDir.Normalize();
                PositionCenterV2 += gridOffsetDir * (Math.Min(SizeV.X, SizeV.Y) - 5);
            }

            if (GameObjects.Use3D)
                GameObjectRenderer.IgnoredGameObjects.Add(avatar);
        }
    }
}
