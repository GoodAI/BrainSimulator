using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenTK;
using OpenTK.Graphics;

namespace Render
{
    public static class Test
    {
        public static void A()
        {
            //var a = new OpenTK.GameWindow(0, 0, GraphicsMode.Default, "");
            var a = new OpenTK.NativeWindow(0, 0, "", GameWindowFlags.Default, GraphicsMode.Default, DisplayDevice.Default);
        }
    }
}
