using VRageMath;
using World.GameActors.Tiles;

namespace World.Atlas
{
    public interface IAtmosphere
    {
        /// <summary>
        /// Temperature at given position.
        /// </summary>
        /// <param name="position"></param>
        /// <returns></returns>
        float Temperature(Vector2 position);

        /// <summary>
        /// Call before Temperature() call or every step.
        /// </summary>
        void Update();

        void RegisterHeatSource(IHeatSource heatSource);

        void UnregisterHeatSource(IHeatSource heatSource);
    }
}