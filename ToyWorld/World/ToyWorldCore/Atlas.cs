using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using VRageMath;
using Utils;
using World.GameActors.GameObjects;
using World.GameActors.Tiles;
using World.Physics;
using World.GameActors;

namespace World.ToyWorldCore
{
    interface IAtlas
    {
        /// <summary>
        /// Adds avatar to Atlas or returns false.
        /// </summary>
        /// <param name="avatar"></param>
        /// <returns></returns>
        bool AddAvatar(World.GameActors.GameObjects.IAvatar avatar);

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

        List<Character> Characters { get; }

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
    }

    public class Atlas : IAtlas
    {
        public List<ITileLayer> TileLayers { get; private set; }

        public List<IObjectLayer> ObjectLayers { get; private set; }

        public Dictionary<int, IAvatar> Avatars { get; private set; }

        public List<Character> Characters { get; private set; }

        public Dictionary<int, StaticTile> StaticTilesContainer { get; private set; }

        public Atlas()
        {
            Avatars = new Dictionary<int, IAvatar>();
            Characters = new List<Character>();
            TileLayers = new List<ITileLayer>();
            ObjectLayers = new List<IObjectLayer>();
            StaticTilesContainer = new Dictionary<int, StaticTile>();
        }

        public ILayer<GameActor> GetLayer(LayerType layerType)
        {
            if (layerType == LayerType.Object
                || layerType == LayerType.ForegroundObject)
            {
                return (ILayer<GameActor>)ObjectLayers.FirstOrDefault(x => x.LayerType == layerType);
            }
            return (ILayer<GameActor>)TileLayers.First(x => x.LayerType == layerType);
        }

        public bool AddAvatar(IAvatar avatar)
        {
            MyContract.Requires<ArgumentNullException>(avatar != null, "Avatar cannot be null");
            try
            {
                Avatars.Add(avatar.Id, avatar);
            }
            catch (ArgumentException)
            {
                return false;
            }
            return true;
        }

        public List<IAvatar> GetAvatars()
        {
            Contract.Ensures(Contract.Result<List<IAvatar>>() != null);
            return Avatars.Values.ToList();
        }


        public bool ContainsCollidingTile(Vector2I coordinates)
        {
            if (((ITileLayer)GetLayer(LayerType.Obstacle)).GetTile(coordinates) != null)
            {
                return true;
            }
            if (((ITileLayer)GetLayer(LayerType.ObstacleInteractable)).GetTile(coordinates) != null)
            {
                return true;
            }
            return false;
        }
    }
}
