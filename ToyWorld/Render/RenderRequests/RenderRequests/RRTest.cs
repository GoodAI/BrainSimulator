using OpenTK;
using OpenTK.Graphics.OpenGL;
using OpenTK.Input;
using Render.Renderer;
using Render.RenderObjects.Geometries;

namespace Render.RenderRequests.RenderRequests
{
    class RRTest : RenderRequestBase, IRRTest
    {
        private readonly Square m_sq = new Square();

        private bool odd;


        public RRTest()
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

            GL.MatrixMode(MatrixMode.Modelview);
            Matrix4 m;

            if (odd)
                m = Matrix4.CreateScale(0.5f);
            else
                m = Matrix4.CreateScale(0.1f);


            GL.LoadMatrix(ref m);
            odd = !odd;

            m_sq.Draw();

            renderer.Context.SwapBuffers();
        }

        void HandleWindow(GLRenderer renderer)
        {
            renderer.ProcessRequests();
            renderer.Window.ProcessEvents();
        }

        #endregion
    }
}
