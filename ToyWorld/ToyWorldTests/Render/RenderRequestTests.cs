using System;
using System.IO;
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
using Render.Tests.RRs;
using ToyWorldTests.Game;
using Xunit;

namespace ToyWorldTests.Render
{
    [Collection("Renderer")]
    public class RenderRequestTests : GameControllerTestBase
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
                r.Init(GameController.Renderer, GameController.World);
                GameController.Renderer.CheckError();
            }

            foreach (var rr in RenderRequestFactory.AvatarRRs)
            {
                var r = rr as RenderRequest;
                Assert.NotNull(r);
                r.Init(GameController.Renderer, GameController.World);
                GameController.Renderer.CheckError();
            }

            GameController.Renderer.MakeContextNotCurrent();
        }

        private void TestStepsImage(IRenderRequestBase renderRequest)
        {
            Assert.True(renderRequest.Image == null || renderRequest.Image.Length == 0);

            GameController.MakeStep();
            Assert.True(
                renderRequest.Image == null
                || renderRequest.Image.Length == 0
                || renderRequest.Image.All(u => (u & 0xFFFFFFFF) == 0));

            renderRequest.GatherImage = true;
            Assert.Equal(renderRequest.GatherImage, true);
            GameController.MakeStep();
            Assert.Equal(renderRequest.GatherImage, true);
            Assert.True(renderRequest.Image.Length >= renderRequest.Resolution.Width * renderRequest.Resolution.Height); // Allocation needn't be immediate
            Assert.Contains(renderRequest.Image, u => (u & 0xFFFFFF00) != 0);
        }

        #region RRs

        [Fact]
        public void FullMapRR()
        {
            var RRTest = GameController.RegisterRenderRequest<IFullMapRR>();
            TestStepsImage(RRTest);
        }

        [Fact]
        public void FreeMapRR()
        {
            var RRTest = GameController.RegisterRenderRequest<IFreeMapRR>();
            TestStepsImage(RRTest);
        }

        #endregion

        #region AvatarRRs

        [Fact]
        public void FoVAvatarRR()
        {
            var RRTest = GameController.RegisterRenderRequest<IFovAvatarRR>(1);
            TestStepsImage(RRTest);
        }

        [Fact]
        public void FoFAvatarRR()
        {
            var RR = GameController.RegisterRenderRequest<IFovAvatarRR>(1);
            var RRTest = GameController.RegisterRenderRequest<IFofAvatarRR>(1);
            RRTest.FovAvatarRenderRequest = RR;
            TestStepsImage(RRTest);
        }
        [Fact]
        public void FoFAvatarRRThrows()
        {
            var RRTest = GameController.RegisterRenderRequest<IFofAvatarRR>(1);
            Assert.ThrowsAny<MissingFieldException>((Action)GameController.MakeStep);
            Assert.ThrowsAny<ArgumentException>(() => RRTest.FovAvatarRenderRequest = null);

            //var differentRR = GameController.RegisterRenderRequest<IFovAvatarRR>(0);
            //Assert.ThrowsAny<ArgumentException>(() => RRTest.FovAvatarRenderRequest = differentRR); // TODO: need at least two avatars for this test
        }

        #endregion
    }
}
