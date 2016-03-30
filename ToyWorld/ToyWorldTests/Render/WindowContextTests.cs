using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using Game;
using GoodAI.ToyWorld.Control;
using OpenTK.Graphics.OpenGL;
using OpenTK.Input;
using Render.Renderer;
using Render.RenderRequests;
using Utils;
using World.Tiles;
using Xunit;

namespace ToyWorldTests.Render
{
    public class WindowContextTests : RenderingTestBase
    {
        [Fact]
        public void CheckRRInits()
        {
            var c = ControllerFactory.GetController() as GameControllerBase;
            Assert.NotNull(c);
            c.Init(null);

            foreach (var rr in RenderRequestFactory.RRs)
            {
                var r = rr as RenderRequest;
                Assert.NotNull(r);
                r.Init(c.Renderer);
            }

            foreach (var rr in RenderRequestFactory.ARRs)
            {
                var r = rr as RenderRequest;
                Assert.NotNull(r);
                r.Init(c.Renderer);
            }
        }

        [Fact(Skip = "Manual input needed")]
        //[Fact]
        public void CreateRenderWindowAndContext()
        {
            var res = ManualDebugDraw("CreateRenderWindowAndContext");

            var k = res;
        }
    }
}
