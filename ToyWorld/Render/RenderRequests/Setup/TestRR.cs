using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Remoting.Contexts;
using System.Text;
using System.Threading.Tasks;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using Render.Renderer;
using Render.RenderObjects.Geometries;
using VRageMath;

namespace Render.RenderRequests
{
    class RenderRequestTest : RenderRequestBase, IRenderRequestTest
    {
        public override float Size { get; set; }
        public override float Position { get; set; }
        public override float Resolution { get; set; }
        public override float MemAddress { get; set; }


        private readonly Square m_sq = new Square();

        private bool odd;


        public RenderRequestTest()
        {
            m_sq.Init();
        }


        public override void Draw(GLRenderer renderer)
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
    }
}
