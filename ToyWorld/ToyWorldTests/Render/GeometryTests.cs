using System;
using System.Threading;
using System.Threading.Tasks;
using OpenTK.Graphics.OpenGL;
using OpenTK.Input;
using Render.Renderer;
using Render.RenderObjects.Geometries;
using Utils;
using World.Tiles;
using Xunit;

namespace ToyWorldTests.Render
{
    public class GeometryTests : RenderingTestBase
    {
        [Fact]
        public void InitGeometry()
        {
            ManualDebugDraw("CreateRenderWindowAndContext");
        }

        void BuildSimpleGeometryInternal(GLRenderer r)
        {
        }

        [Fact]
        public void BuildGeometries()
        {
            
        }

        [Fact(Skip = "Manual input needed")]
        public void RenderSimpleShape()
        {
            ManualDebugDraw("CreateRenderWindowAndContext");
        }

        void RenderSimpleShapeInternal(GLRenderer r)
        {

        }

        [Fact(Skip = "Manual input needed")]
        public void CreateWindow()
        {
            Thread.Sleep(5000);

        }
    }
}
