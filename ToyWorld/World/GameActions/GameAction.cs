using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace World.Tiles
{
    public abstract class GameAction
    {
    }

    public class UsePickaxe : GameAction
    {
        public float Damage { get; set; }
    }
}
