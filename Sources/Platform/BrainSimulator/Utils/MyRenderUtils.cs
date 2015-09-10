using OpenTK.Graphics.OpenGL;
using System.Drawing;

namespace GoodAI.BrainSimulator.Utils
{
    public class MyRenderUtils
    {
        public static readonly Color GRID_GRAY = Color.FromArgb(160, 160, 160);

        public static void RenderAxes(bool renderZ)
        {
            float size = 0.5f;

            GL.Begin(PrimitiveType.Lines);

            GL.Color3(Color.Red);
            GL.Vertex3(0, 0, 0);
            GL.Vertex3(size, 0, 0);

            GL.Color3(Color.LimeGreen);
            GL.Vertex3(0, 0, 0);
            GL.Vertex3(0, size, 0);

            if (renderZ)
            {
                GL.Color3(Color.Blue);
                GL.Vertex3(0, 0, 0);
                GL.Vertex3(0, 0, size);
            }

            GL.End();
        }

        public static void RenderGrid(bool inXZPlane)
        {
            int num = 10;
            float size = 5;
            float sizeHalf = size * 0.5f;

            GL.Begin(PrimitiveType.Lines);
            GL.Color3(GRID_GRAY);

            if (inXZPlane)
            {
                for (int i = 0; i <= num; i++)
                {
                    float t = (float)i / num;
                    GL.Vertex3(size * (t - 0.5f), 0, -sizeHalf);
                    GL.Vertex3(size * (t - 0.5f), 0, sizeHalf);

                    GL.Vertex3(-sizeHalf, 0, size * (t - 0.5f));
                    GL.Vertex3(sizeHalf, 0, size * (t - 0.5f));
                }
            }
            else
            {
                for (int i = 0; i <= num; i++)
                {
                    float t = (float)i / num;
                    GL.Vertex2(size * (t - 0.5f), -sizeHalf);
                    GL.Vertex2(size * (t - 0.5f), sizeHalf);

                    GL.Vertex2(-sizeHalf, size * (t - 0.5f));
                    GL.Vertex2(sizeHalf, size * (t - 0.5f));
                }
            }

            GL.End();
        }    
    }
}
