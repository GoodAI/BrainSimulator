using System.Collections.Generic;
using World.GameActors;

namespace World.ToyWorldCore
{
    public interface ILayer<out T> where T : GameActor
    {
        LayerType LayerType { get; set; }
    }
}
