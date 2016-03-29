using System;
using System.Threading;
using System.Threading.Tasks;
using GoodAI.ToyWorld.Control;
using OpenTK.Graphics.OpenGL;
using OpenTK.Input;
using Render.RenderRequests;
using Utils;
using World.Tiles;
using Xunit;

namespace ToyWorldTests.Render
{
    public class GeometryTests
    {
        [Fact]
        public void BuildGeometries()
        {

        }

        [Fact(Skip = "Manual input needed")]
        public void DrawBasicGeometry()
        {
            var gc = ControllerFactory.GetController();
            gc.InitWorld(null);

            var RRTest = gc.RegisterRenderRequest<IRRTest>();


            while (RRTest.WindowKeypressResult == default(Key))
            {
                gc.MakeStep();
            }


            Assert.True(RRTest.WindowKeypressResult == Key.A);
        }

    }
}
