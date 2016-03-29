using System;
using System.Threading;
using System.Threading.Tasks;
using GoodAI.ToyWorld.Control;
using OpenTK.Input;
using Render.RenderRequests;
using Render.RenderRequests.Tests;
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
            gc.Init(null);

            var RRTest = gc.RegisterRenderRequest<IRRTest>();


            while (RRTest.WindowKeypressResult == default(Key))
            {
                gc.MakeStep();
            }


            Assert.True(RRTest.WindowKeypressResult == Key.A);
        }
    }
}
