using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.ToyWorld.Control;
using Render.Renderer;
using Render.RenderRequests;

namespace Game
{
    internal class TestGameController : GameControllerBase
    {
        #region GameControllerBase overrides

        public TestGameController()
            : base(new GLRenderer())
        { }

        #endregion
    }
}
