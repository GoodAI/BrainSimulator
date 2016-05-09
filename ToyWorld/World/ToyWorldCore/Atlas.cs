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

        IEnumerable<GameActorPosition> ActorsInFrontOf<T>(T sender, LayerType type = LayerType.All, float distance = 1, float width = 1) where T : class, IDirectable, IGameObject;

        Vector2 PositionInFrontOf<T>(T sender, float distance) where T : class, IDirectable, IGameObject;

        void Remove(GameActorPosition target);

        bool Add(GameActorPosition gameActorPosition);

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

        private List<ILayer<GameActor>> GetObstaceLayers()
        {
            var obstacleLayers = new List<ILayer<GameActor>>();
            foreach (ILayer<GameActor> layer in Layers)
            {
                if ((layer.LayerType & LayerType.Obstacles) > 0)
                {
                    obstacleLayers.Add(layer);
                }
            }
            return obstacleLayers;
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

        public IEnumerable<GameActorPosition> ActorsAt(float x, float y, LayerType type = LayerType.All, float width = 0.5f)
        {
            Vector2 position = new Vector2(x, y);
            // for all layers except object layer
            foreach (ILayer<GameActor> layer in Layers.Where(t => (t.LayerType & type) > 0 && (type & LayerType.Object) != LayerType.Object))
            {
                GameActor actor = layer.GetActorAt((int)Math.Floor(x), (int)Math.Floor(y));

                if (actor == null)
                    continue;
                GameActorPosition actorPosition = new GameActorPosition(actor, position, layer.LayerType);
                yield return actorPosition;
            }

            if ((type & LayerType.Object) <= 0) yield break;
            {
                foreach (IGameObject gameObject in ((IObjectLayer) GetLayer(LayerType.Object)).GetGameObjects())
                {
                    if (!gameObject.PhysicalEntity.Shape.CollidesWith(new CircleShape(position, width))) continue;
                    GameActor actor = (GameActor) gameObject;
                    GameActorPosition actorPosition = new GameActorPosition(actor, position, LayerType.Object);
                    yield return actorPosition;
                }
            }
        }

        public IEnumerable<GameActorPosition> ActorsInFrontOf<T>(T sender, LayerType type = LayerType.All, float distance = 1, float width = 0.5f) where T : class, IDirectable, IGameObject
        {
            var target = PositionInFrontOf(sender, distance);
            var actorsInFrontOf = ActorsAt(target.X, target.Y, type, width);
            return actorsInFrontOf;
        }

        public Vector2 PositionInFrontOf<T>(T sender, float distance) where T : class, IDirectable, IGameObject
        {
            Vector2 direction = Vector2.UnitY * distance;
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
            Circle circle = new Circle(position, 50);
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
                throw new ArgumentException("actor");
            }
            bool anyCollision = gameObjectLayer.GetPhysicalEntities(circle).Any(x => x.CollidesWith(physicalEntity));
            if (anyCollision)
            {
                return false;
            }
            bool anyObstacleOnPosition = GetObstaceLayers().Any(x => x.GetActorAt(new Vector2I(gameActorPosition.Position)) != null);
            if (anyObstacleOnPosition)
            {
                return false;
            }
            
            return layer.Add(gameActorPosition);
        }

        public void ReplaceWith(GameActorPosition original, GameActor replacement)
        {
            foreach (ILayer<GameActor> layer in Layers)
            {
                bool result = layer.ReplaceWith(original, replacement);
                if (result) return;
            }
        }
    }
}
