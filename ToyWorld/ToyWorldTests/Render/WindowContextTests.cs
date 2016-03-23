using OpenTK.Graphics.OpenGL;
using Render.Renderer;
using World.Tiles;
using Xunit;

namespace ToyWorldTests.Render
{
    public class WindowContextTests
    {
        [Fact]
        public void CreateWindow()
        {
            var r = new GLRenderer();

            r.CreateWindow(1024, 768);
            r.CreateContext();

            GL.Begin(BeginMode.Lines);

        }
    }
}
