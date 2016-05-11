using System;
using System.Collections.Generic;
using System.Diagnostics;
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

        IEnumerable<GameActorPosition> ActorsAt(float x, float y, LayerType type = LayerType.All, float width = 1);

        IEnumerable<GameActorPosition> ActorsInFrontOf<T>(T sender, LayerType type = LayerType.All, float distance = 1,
            float width = 1) where T : class, IDirectable, IGameObject;

        /// <summary>
        /// 
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="sender"></param>
        /// <param name="distance">Distance from center of given object.</param>
        /// <returns>Coordinates of position in front of given sender.</returns>
        Vector2 PositionInFrontOf<T>(T sender, float distance) where T : class, IDirectable, IGameObject;

        /// <summary>
        /// Removes given GameActor from Layer specified in GameActorPosition.
        /// </summary>
        /// <param name="target"></param>
        void Remove(GameActorPosition target);

        /// <summary>
        /// Adds given GameActor to certain position. If position is not free, returns false.
        /// </summary>
        /// <param name="gameActorPosition"></param>
        /// <returns>True if operation were successful.</returns>
        bool Add(GameActorPosition gameActorPosition);

        /// <summary>
        /// Replace GameActor with replacement. When Tile and GameObject is given, ArgumentException is thrown.
        /// Layer is specified in GameActorPosition original.
        /// </summary>
        /// <param name="original"></param>
        /// <param name="replacement"></param>
        void ReplaceWith(GameActorPosition original, GameActor replacement);

        List<IGameObject> StayingOnTile(Vector2I tilePosition);
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

        private IEnumerable<ILayer<GameActor>> GetObstaceLayers()
        {
            foreach (ILayer<GameActor> layer in Layers)
            {
                if ((layer.LayerType & LayerType.Obstacles) > 0)
                {
                    yield return layer;
                }
            }
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
            if (((ITileLayer) GetLayer(LayerType.Obstacle)).GetActorAt(coordinates.X, coordinates.Y) != null)
            {
                return true;
            }
            if (((ITileLayer) GetLayer(LayerType.ObstacleInteractable)).GetActorAt(coordinates.X, coordinates.Y) != null)
            {
                return true;
            }
            return false;
        }

        public IEnumerable<GameActorPosition> ActorsAt(float x, float y, LayerType type = LayerType.All,
            float width = 0.5f)
        {
            Vector2 position = new Vector2(x, y);

            List<ILayer<GameActor>> selectedLayers = Layers.Where(t => type.HasFlag(t.LayerType)).ToList();
            // for all layers except object layer
            IEnumerable<ILayer<GameActor>> selectedTileLayers = selectedLayers.Where(t => LayerType.TileLayers.HasFlag(t.LayerType));
            foreach (ILayer<GameActor> layer in selectedTileLayers)
            {
                GameActorPosition actorPosition = TileAt(layer, position);
                if (actorPosition != null)
                {
                    yield return actorPosition;
                }
            }

            var circle = new CircleShape(position, width);
            IEnumerable<ILayer<GameActor>> selectedObjectLayers = selectedLayers.Where(t => LayerType.ObjectLayers.HasFlag(t.LayerType));
            foreach (ILayer<GameActor> layer in selectedObjectLayers)
            {
                foreach (IGameObject gameObject in ((IObjectLayer) layer).GetGameObjects())
                {
                    GameActorPosition actorPosition = GameObjectAt(gameObject, circle, position, layer);
                    if (actorPosition != null)
                    {
                        yield return actorPosition;
                    }
                }
            }
        }

        private static GameActorPosition GameObjectAt(IGameObject gameObject, IShape shape, Vector2 position, ILayer<GameActor> layer)
        {
            IShape gameObjectShape = gameObject.PhysicalEntity.Shape;
            if (!gameObjectShape.CollidesWith(shape)) return null;
            GameActor actor = (GameActor) gameObject;
            GameActorPosition actorPosition = new GameActorPosition(actor, position, layer.LayerType);
            return actorPosition;
        }

        private static GameActorPosition TileAt(ILayer<GameActor> layer, Vector2 position)
        {
            GameActor actor = layer.GetActorAt((int) Math.Floor(position.X), (int) Math.Floor(position.Y));
            if (actor == null) return null;
            GameActorPosition actorPosition = new GameActorPosition(actor, position, layer.LayerType);
            return actorPosition;
        }

        public IEnumerable<GameActorPosition> ActorsInFrontOf<T>(T sender, LayerType type = LayerType.All,
            float distance = 1, float width = 0.5f) where T : class, IDirectable, IGameObject
        {
            Vector2 target = PositionInFrontOf(sender, distance);
            IEnumerable<GameActorPosition> actorsInFrontOf = ActorsAt(target.X, target.Y, type, width);
            return actorsInFrontOf;
        }

        public Vector2 PositionInFrontOf<T>(T sender, float distance) where T : class, IDirectable, IGameObject
        {
            Vector2 direction = Vector2.UnitY*distance;
            direction.Rotate(sender.Direction);
            Vector2 target = sender.Position + direction;
            return target;
        }

        public void Remove(GameActorPosition target)
        {
            ReplaceWith(target, null);
        }

        public bool Add(GameActorPosition gameActorPosition)
        {
            ILayer<GameActor> layer = GetLayer(gameActorPosition.Layer);


            IObjectLayer gameObjectLayer = GetLayer(LayerType.Object) as IObjectLayer;
            Debug.Assert(gameObjectLayer != null, "gameObjectLayer != null");
            Vector2 position = gameActorPosition.Position;
            GameActor actor = gameActorPosition.Actor;
            IPhysicalEntity physicalEntity;
            if (actor is IGameObject)
            {
                physicalEntity = (actor as IGameObject).PhysicalEntity;
                physicalEntity.Position = position;
            }
            else if (actor is Tile)
            {
                physicalEntity = (actor as Tile).GetPhysicalEntity(new Vector2I(gameActorPosition.Position));
            }
            else
            {
                throw new ArgumentException("actor is unknown Type");
            }
            bool anyCollision = gameObjectLayer.GetPhysicalEntities().Any(x => x.CollidesWith(physicalEntity));
            if (anyCollision)
            {
                return false;
            }
            bool anyObstacleOnPosition =
                GetObstaceLayers().Any(x => x.GetActorAt(new Vector2I(gameActorPosition.Position)) != null);
            if (anyObstacleOnPosition)
            {
                return false;
            }

            return layer.Add(gameActorPosition);
        }

        public void ReplaceWith(GameActorPosition original, GameActor replacement)
        {
            if ((original.Actor is Tile && replacement is GameObject)
                || (original.Actor is GameObject && replacement is Tile))
            {
                throw new ArgumentException("atlas.ReplaceWith tries replace Tile with GameObject or GameObject with Tile");
            }
            foreach (ILayer<GameActor> layer in Layers)
            {
                bool result = layer.ReplaceWith(original, replacement);
                if (result) return;
            }
        }

        public List<IGameObject> StayingOnTile(Vector2I tilePosition)
        {
            return ((IObjectLayer) GetLayer(LayerType.Object)).GetGameObjects(tilePosition);
        }

        public static bool InsideTile(Vector2I tilePosition, Vector2 position)
        {
            Vector2 start = new Vector2(tilePosition);
            Vector2 end = new Vector2(tilePosition) + Vector2.One;
            return position.X >= start.X && position.X <= end.X && position.X >= start.Y && position.Y <= end.Y;
        }

        public static Vector2I OnTile(Vector2 position)
        {
            return new Vector2I(Vector2.Floor(position));
        }
    }
}
