using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.ToyWorld.Control;

namespace GoodAI.ToyWorldAPI.Control
{
    public static class GameControlFactory
    {
        public static IGameController GetController()
        {
            return RenderRequestFactory.GetTestRenderRequest();
        }
    }
}
