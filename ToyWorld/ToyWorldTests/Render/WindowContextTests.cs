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
    public class WindowContextTests : RenderingTestBase
    {
        [Fact(Skip = "Manual input needed")]
        public void CreateRenderWindowAndContext()
        {
            ManualDebugDraw("CreateRenderWindowAndContext");
        }

    }
}
