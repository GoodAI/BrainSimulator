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
    public class BasicGameController : GameControllerBase
    {
        #region IGameController overrides

        public BasicGameController()
            : base(new GLRenderer())
        { }

        #endregion

    }
}
