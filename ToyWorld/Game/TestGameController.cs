using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.ToyWorld.Control;
using Render.Renderer;
using Render.RenderRequests;
using Render.RenderRequests.Setup;

namespace Game
{
    internal class TestGameController : BasicGameController
    {
        private readonly IRenderer m_renderer = new GLRenderer();


        #region IGameController overrides


        #endregion
    }
}
