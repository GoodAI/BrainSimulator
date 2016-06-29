using System;
using System.Collections.Generic;
using VRageMath;
using World.Atlas.Layers;
using World.GameActors;
using World.GameActors.GameObjects;
using World.GameActors.Tiles;
using World.Physics;

namespace World.Atlas
{
    public interface IAtlas
    {
        /// <summary>
        /// Adds avatar to Atlas or returns false.
        /// </summary>
        /// <param name="avatar"></param>
        /// <returns></returns>
        bool AddAvatar(IAvatar avatar);

        /// <summary>
        /// Dictionary of all registered avatars, where key is ID of Avatar and Value is IAvatar.
        /// </summary>
        Dictionary<int, IAvatar> Avatars { get; }

        /// <summary>
        /// Returns List of all Avatars.
        /// </summary>
        /// <returns></returns>
        List<IAvatar> GetAvatars();

        /// <summary>
        /// Returns IObjectLayer or ITileLayer for given LayerType
        /// </summary>
        /// <param name="layerType"></param>
        /// <returns></returns>
        ILayer<GameActor> GetLayer(LayerType layerType);

        List<ICharacter> Characters { get; }

        /// <summary>
        /// Container for all ObjectLayers
        /// </summary>
        List<IObjectLayer> ObjectLayers { get; }

        /// <summary>
        /// Dictionary of static tiles.
        /// </summary>
        Dictionary<int, StaticTile> StaticTilesContainer { get; }

        /// <summary>
        /// Container for all TileLayers
        /// </summary>
        List<ITileLayer> TileLayers { get; }

        /// <summary>
        /// Checks whether on given coordinates is colliding tile.
        /// </summary>
        /// <param name="coordinates"></param>
        /// <returns></returns>
        bool ContainsCollidingTile(Vector2I coordinates);

        /// <summary>
        ///
        /// </summary>
        /// <param name="tilePosition"></param>
        /// <param name="type"></param>
        /// <param name="width"></param>
        /// <returns>All game actors from all given layers colliding with given tile position.</returns>
        IEnumerable<GameActorPosition> ActorsAt(Vector2 tilePosition, LayerType type = LayerType.All, float width = 1);

        /// <summary>
        ///
        /// </summary>
        /// <typeparam name="T">IDirectable and IGameObject</typeparam>
        /// <param name="sender"></param>
        /// <param name="type"></param>
        /// <param name="distance"> Distance to center of observation.</param>
        /// <param name="width">If searching through Object layers, width of searching circle.</param>
        /// <returns>GameActors in front.</returns>
        IEnumerable<GameActorPosition> ActorsInFrontOf<T>(T sender, LayerType type = LayerType.All, float distance = 1,
            float width = 1) where T : class, IRotatable, IGameObject;

        /// <summary>
        /// Returns list of free cells around given position, in given layer
        /// </summary>
        /// <param name="position"></param>
        /// <param name="type"></param>
        /// <returns></returns>
        IEnumerable<Vector2I> FreePositionsAround(Vector2I position, LayerType type = LayerType.All);

        /// <summary>
        ///
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="sender"></param>
        /// <param name="distance">Distance from center of given object.</param>
        /// <returns>Coordinates of position in front of given sender.</returns>
        Vector2 PositionInFrontOf<T>(T sender, float distance) where T : class, IRotatable, IGameObject;

        /// <summary>
        /// Removes given GameActor from Layer specified in GameActorPosition.
        /// </summary>
        /// <param name="target"></param>
        void Remove(GameActorPosition target);

        /// <summary>
        /// Adds given GameActor to certain position. If position is not free, returns false.
        /// </summary>
        /// <param name="gameActorPosition"></param>
        /// <param name="collidesWithObstacles"></param>
        /// <returns>True if operation were successful.</returns>
        bool Add(GameActorPosition gameActorPosition, bool collidesWithObstacles = false);

        /// <summary>
        /// Replace GameActor with replacement. When Tile and GameObject is given, ArgumentException is thrown.
        /// Layer is specified in GameActorPosition original.
        /// </summary>
        /// <param name="original"></param>
        /// <param name="replacement"></param>
        void ReplaceWith(GameActorPosition original, GameActor replacement);

        /// <summary>
        ///
        /// </summary>
        /// <param name="tilePosition"></param>
        /// <returns>All objects from Object layer standing on given position.</returns>
        List<IGameObject> StayingOnTile(Vector2I tilePosition);

        /// <summary>
        /// List of newly added IAutoupdatables.
        /// </summary>
        List<IAutoupdateableGameActor> NewAutoupdateables { get; }

        /// <summary>
        /// This method adds actor to automatically updated queue.
        /// </summary>
        /// <param name="actor"></param>
        void RegisterToAutoupdate(IAutoupdateableGameActor actor);

        IAreasCarrier AreasCarrier { get; set; }

        /// <summary>
        /// Increment time of simulation.
        /// </summary>
        void IncrementTime(int days = 0, int hours = 0, int minutes = 0, int seconds = 10, int millis = 0);

        /// <summary>
        /// Increment time of simulation.
        /// </summary>
        void IncrementTime(TimeSpan timeSpan);

        void UpdateLayers();

        /// <summary>
        /// [0,1] Winter/Summer
        /// </summary>
        float Summer { get; }

        TimeSpan YearLength { get; set; }

        /// <summary>
        /// [0,1] Night/Day
        /// </summary>
        float Day { get; }

        TimeSpan DayLength { get; set; }

        /// <summary>
        /// Date and Time is currently affecting Temperature.
        /// </summary>
        /// <returns>Time in simulation</returns>
        DateTime RealTime { get; }

        /// <summary>
        /// Implementation of atmosphere.
        /// </summary>
        IAtmosphere Atmosphere { get; set; }

        /// <summary>
        /// Temperature at given point.
        /// </summary>
        /// <param name="position"></param>
        /// <returns></returns>
        float Temperature(Vector2 position);

        /// <summary>
        /// Must be called in init of every IHeatSource;
        /// </summary>
        /// <param name="heatSource"></param>
        void RegisterHeatSource(IHeatSource heatSource);

        /// <summary>
        /// Must be called in IHeatSource when it is getting inactive;
        /// </summary>
        /// <param name="heatSource"></param>
        void UnregisterHeatSource(IHeatSource heatSource);
    }
}