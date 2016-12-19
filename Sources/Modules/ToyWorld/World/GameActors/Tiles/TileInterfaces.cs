using World.Atlas;
using World.GameActors.GameObjects;

namespace World.GameActors.Tiles
{
    /// <summary>
    /// Tile which want to detect object colliding or standing on him.
    /// </summary>
    public interface IDetectorTile
    {
        /// <summary>
        /// If this is true, object is not detected until center of object is within tile.
        /// </summary>
        bool RequiresCenterOfObject { get; }

        /// <summary>
        /// When any GameObject was detected, it is passed as parameter.
        /// Can be called multiple times per step.
        /// </summary>
        /// <param name="gameObject"></param>
        /// <param name="atlas"></param>
        void ObjectDetected(IGameObject gameObject, IAtlas atlas);
    }

    /// <summary>
    /// 
    /// </summary>
    public interface IHeatSource : IDynamicTile
    {
        /// <summary>
        /// Adds temperature equal of heat at center.
        /// Then decreasing polynomially to the MaxDistance
        /// </summary>
        float Heat { get; }

        /// <summary>
        /// Distance from center of source where temperature change is 0;
        /// </summary>
        float MaxDistance { get; }
    }
}