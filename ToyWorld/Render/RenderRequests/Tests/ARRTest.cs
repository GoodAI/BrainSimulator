using OpenTK;
using OpenTK.Graphics.OpenGL;
using OpenTK.Input;
using Render.Renderer;
using Render.RenderObjects.Geometries;
using Render.RenderRequests.AgentRenderRequests;

namespace Render.RenderRequests.Tests
{
    class ARRTest : AvatarRenderRequestBase, IARRTest
    {
        private readonly Square m_sq = new Square();

        private bool odd;


        public ARRTest(int avatarID)
            : base(avatarID)
        {
            WindowKeypressResult = default(Key);
        }


        #region IRRTest overrides

        public Key WindowKeypressResult { get; private set; }

        public float MemAddress { get; set; }

        #endregion

        #region RenderRequestBase overrides

        public override void Init(IRenderer renderer)
        {
            m_sq.Init();

            renderer.Window.KeyDown += (sender, args) => WindowKeypressResult = args.Key;
            renderer.Window.Visible = true;
        }

        public override void Draw(GLRenderer renderer)
        {
            DrawInternal(renderer);
            HandleWindow(renderer);
        }

        void DrawInternal(GLRenderer renderer)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

            renderer.Context.SwapBuffers();
        }

        void HandleWindow(GLRenderer renderer)
        {
            renderer.Window.ProcessEvents();
        }

        #endregion
    }
}
