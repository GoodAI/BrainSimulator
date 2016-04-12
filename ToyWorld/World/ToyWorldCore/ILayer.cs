using System.Collections.Generic;
using World.GameActors;

namespace World.ToyWorldCore
{
    public interface ILayer<T> where T : GameActor
    {
        LayerType LayerType { get; set; }

        List<T> GetAllObjects();
    }
}
