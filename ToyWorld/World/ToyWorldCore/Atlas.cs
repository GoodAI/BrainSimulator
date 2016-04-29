using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using VRageMath;
using World.GameActors;
using World.GameActors.GameObjects;
using World.GameActors.Tiles;
using World.Physics;

namespace World.ToyWorldCore
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

        IEnumerable<GameActorPosition> ActorsAt(int x, int y, LayerType type = LayerType.All);

        IEnumerable<GameActorPosition> ActorsInFrontOf<T>(T sender, LayerType type = LayerType.All) where T : class, IDirectable, IGameObject;

        void Remove(GameActorPosition target);

        void ReplaceWith(GameActorPosition original, GameActor replacement);

    }

    public class Atlas : IAtlas
    {
        public List<ITileLayer> TileLayers { get; private set; }

        public List<IObjectLayer> ObjectLayers { get; private set; }

        private IEnumerable<ILayer<GameActor>> Layers
        {
            get
            {
                foreach (ITileLayer layer in TileLayers)
                    yield return layer;
                foreach (IObjectLayer layer in ObjectLayers)
                    yield return layer;
            }
        }

        public Dictionary<int, IAvatar> Avatars { get; private set; }

        public List<ICharacter> Characters { get; private set; }

        public Dictionary<int, StaticTile> StaticTilesContainer { get; private set; }

        public Atlas()
        {
            Avatars = new Dictionary<int, IAvatar>();
            Characters = new List<ICharacter>();
            TileLayers = new List<ITileLayer>();
            ObjectLayers = new List<IObjectLayer>();
            StaticTilesContainer = new Dictionary<int, StaticTile>();
        }

        public ILayer<GameActor> GetLayer(LayerType layerType)
        {
            if (layerType == LayerType.Object
                || layerType == LayerType.ForegroundObject)
            {
                return ObjectLayers.FirstOrDefault(x => x.LayerType == layerType);
            }
            return TileLayers.First(x => x.LayerType == layerType);
        }

        public bool AddAvatar(IAvatar avatar)
        {
            if (avatar == null)
                throw new ArgumentNullException("avatar");
            Contract.EndContractBlock();

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
            if (((ITileLayer)GetLayer(LayerType.Obstacle)).GetActorAt(coordinates.X, coordinates.Y) != null)
            {
                return true;
            }
            if (((ITileLayer)GetLayer(LayerType.ObstacleInteractable)).GetActorAt(coordinates.X, coordinates.Y) != null)
            {
                return true;
            }
            return false;
        }

        public IEnumerable<GameActorPosition> ActorsAt(int x, int y, LayerType type = LayerType.All)
        {
            foreach (ILayer<GameActor> layer in Layers.Where(t => (t.LayerType & type) > 0))
            {
                GameActor actor = layer.GetActorAt(x, y);
                if (actor == null)
                    continue;
                GameActorPosition actorPosition = new GameActorPosition(actor, new Vector2I(x, y));
                yield return actorPosition;
            }
        }

        public IEnumerable<GameActorPosition> ActorsInFrontOf<T>(T sender, LayerType type = LayerType.All) where T : class, IDirectable, IGameObject
        {
            Vector2 direction = Vector2.UnitY;
            direction.Rotate(sender.Direction);
            Vector2 target = sender.Position + direction;
            return ActorsAt((int)Math.Floor(target.X), (int)Math.Floor(target.Y), type);
        }

        public void Remove(GameActorPosition target)
        {
            ReplaceWith(target, null);
        }

        public void ReplaceWith(GameActorPosition original, GameActor replacement)
        {
            throw new NotImplementedException();
        }
    }
}
