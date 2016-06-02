using System;
using System.Threading;
using System.Threading.Tasks;
using Game;
using GoodAI.ToyWorld.Control;
using OpenTK.Input;
using RenderingBase.Renderer;
using RenderingBase.RenderRequests;
using Xunit;

namespace ToyWorldTests.Render
{
    [Collection("Renderer")]
    // TODO: GL-independent geometry tests (use rendererBase only)
    public class GLGeometryTests : GLRendererTests
    {
        [Fact]
        public void BuildGeometries()
        {

        }

        [Fact]
        public void StaticVboThrows()
        {
            
        }
    }
}
