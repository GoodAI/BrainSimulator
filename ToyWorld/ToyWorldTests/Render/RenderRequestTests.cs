using System;
using System.IO;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using Game;
using GoodAI.ToyWorld.Control;
using OpenTK.Graphics.OpenGL;
using OpenTK.Input;
using Render.Renderer;
using Render.RenderRequests;
using Render.Tests.RRs;
using ToyWorldTests.Game;
using Xunit;

namespace ToyWorldTests.Render
{
    [Collection("Renderer")]
    public class RenderRequestTests : ControllerTests
    {
        [Fact]
        public void RRInits()
        {
            Assert.NotNull(GameController.Renderer);
            GameController.Renderer.MakeContextCurrent();

            foreach (var rr in RenderRequestFactory.RRs)
            {
                var r = rr as RenderRequest;
                Assert.NotNull(r);
                r.Init(GameController.Renderer);
                GameController.Renderer.CheckError();
            }

            foreach (var rr in RenderRequestFactory.ARRs)
            {
                var r = rr as RenderRequest;
                Assert.NotNull(r);
                r.Init(GameController.Renderer);
                GameController.Renderer.CheckError();
            }

            GameController.Renderer.MakeContextNotCurrent();
        }


        [Fact]
        public void AvatarFoV()
        {
            var RRTest = GameController.RegisterRenderRequest<IFovAvatarRenderRequest>(0);

            Assert.NotEmpty(RRTest.Image);
            Assert.Equal(RRTest.Size, RRTest.Image.Length);

            GameController.MakeStep();
            Assert.Contains(RRTest.Image, u => (u & 0xFFFFFF00) != 0);
            GameController.MakeStep();
            Assert.Contains(RRTest.Image, u => (u & 0xFFFFFF00) != 0);
        }
    }
}
