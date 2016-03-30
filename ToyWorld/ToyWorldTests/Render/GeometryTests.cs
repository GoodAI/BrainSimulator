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
        //[Fact]
        public void DrawBasicGeometry()
        {
            var gc = ControllerFactory.GetController();
            gc.Init(null);

            var RRTest = gc.RegisterRenderRequest<IRRTest>();

            try
            {
                while (RRTest.WindowKeypressResult == default(Key))
                {
                    gc.MakeStep();
                }
            }
            catch (Exception)
            {
                Assert.False(true);
            }
            finally
            {
                gc.Dispose();
            }


            Assert.Equal(RRTest.WindowKeypressResult, Key.A);
        }
    }
}
