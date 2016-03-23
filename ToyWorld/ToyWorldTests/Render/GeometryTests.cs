using System;
using System.Threading;
using System.Threading.Tasks;
using OpenTK.Graphics.OpenGL;
using OpenTK.Input;
using Render.Renderer;
using Utils;
using World.Tiles;
using Xunit;

namespace ToyWorldTests.Render
{
    public class GeometryTests : RenderingTestBase
    {
        [Fact]
        public void BuildSimpleGeometry()
        {
            
        }

        [Fact]
        public void BuildGeometries()
        {
            
        }

        [Fact(Skip = "Manual input needed")]
        public void RenderSimpleShape()
        {
            ManualDebugDraw(RenderSimpleShapeInternal, "CreateRenderWindowAndContext");
        }

        void RenderSimpleShapeInternal(IRenderer r)
        {

        }

        [Fact(Skip = "Manual input needed")]
        public void CreateWindow()
        {
            Thread.Sleep(5000);

        }
    }
}
