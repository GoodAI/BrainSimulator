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

namespace World.Atlas.Layers
{
    public class Atlas : IAtlas
    {
        public IAtmosphere Atmosphere { get; set; }
        public List<IAutoupdateableGameActor> NewAutoupdateables { get; private set; }
        public List<ITileLayer> TileLayers { get; private set; }
        public List<IObjectLayer> ObjectLayers { get; private set; }
        public IAreasCarrier AreasCarrier { get; set; }

        private long m_timeTicks;
        public DateTime RealTime
        {
            get { return new DateTime(m_timeTicks); }
        }

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
            m_timeTicks = new DateTime(2000, 1, 1, 0,0,0).Ticks;
            NewAutoupdateables = new List<IAutoupdateableGameActor>();
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
            return TileLayers.FirstOrDefault(x => x.LayerType == layerType);
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

        public IEnumerable<GameActorPosition> ActorsAt(Vector2 position, LayerType type = LayerType.All,
            float width = 0.5f)
        {
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
                foreach (IGameObject gameObject in ((IObjectLayer)layer).GetGameObjects())
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
            GameActor actor = (GameActor)gameObject;
            GameActorPosition actorPosition = new GameActorPosition(actor, position, layer.LayerType);
            return actorPosition;
        }

        private static GameActorPosition TileAt(ILayer<GameActor> layer, Vector2 position)
        {
            GameActor actor = layer.GetActorAt((int)Math.Floor(position.X), (int)Math.Floor(position.Y));
            if (actor == null) return null;
            GameActorPosition actorPosition = new GameActorPosition(actor, position, layer.LayerType);
            return actorPosition;
        }

        public IEnumerable<GameActorPosition> ActorsInFrontOf<T>(T sender, LayerType type = LayerType.All,
            float distance = 1, float width = 0.5f) where T : class, IRotatable, IGameObject
        {
            Vector2 target = PositionInFrontOf(sender, distance);
            IEnumerable<GameActorPosition> actorsInFrontOf = ActorsAt(target, type, width);
            return actorsInFrontOf;
        }

        public Vector2 PositionInFrontOf<T>(T sender, float distance) where T : class, IRotatable, IGameObject
        {
            Vector2 direction = Vector2.UnitY * distance;
            direction.Rotate(sender.Rotation);
            Vector2 target = sender.Position + direction;
            return target;
        }

        private bool IsCoordinateFree(Vector2I position, LayerType type)
        {
            return !ActorsAt(new Vector2(position), type).Any();
        }

        public IEnumerable<Vector2I> FreePositionsAround(Vector2I position, LayerType type = LayerType.All)
        {
            return Neighborhoods.ChebyshevNeighborhood(position).Where(x => IsCoordinateFree(x, type));
        }

        public void Remove(GameActorPosition target)
        {
            ReplaceWith(target, null);
        }

        public bool Add(GameActorPosition gameActorPosition, bool collidesWithObstacles = false)
        {
            if (collidesWithObstacles)
            {
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
            }

            ILayer<GameActor> layer = GetLayer(gameActorPosition.Layer);

            return layer.Add(gameActorPosition);
        }

        public void ReplaceWith(GameActorPosition original, GameActor replacement)
        {
            if ((original.Actor is Tile && replacement is GameObject)
                || (original.Actor is GameObject && replacement is Tile))
            {
                throw new ArgumentException("atlas.ReplaceWith tries replace Tile with GameObject or GameObject with Tile");
            }

            GetLayer(original.Layer).ReplaceWith(original, replacement);
        }

        public List<IGameObject> StayingOnTile(Vector2I tilePosition)
        {
            return ((IObjectLayer)GetLayer(LayerType.Object)).GetGameObjects(tilePosition);
        }

        public static bool InsideTile(Vector2I tilePosition, Vector2 position)
        {
            Vector2 start = new Vector2(tilePosition);
            Vector2 end = new Vector2(tilePosition) + Vector2.One;
            return position.X >= start.X && position.X <= end.X && position.Y >= start.Y && position.Y <= end.Y;
        }

        public static Vector2I OnTile(Vector2 position)
        {
            return new Vector2I(Vector2.Floor(position));
        }

        public void RegisterToAutoupdate(IAutoupdateableGameActor actor)
        {
            NewAutoupdateables.Add(actor);
        }

        public float Temperature(Vector2 position)
        {
            return Atmosphere.Temperature(position);
        }

        public void IncrementTime(int days = 0, int hours = 0, int minutes = 0, int seconds = 10, int millis = 0)
        {
            m_timeTicks += new TimeSpan(days, hours, minutes, seconds, millis).Ticks;
        }

        public float Summer
        {
            get
            {
                long year = YearLength.Ticks;
                long halfYear = year / 2;
                bool secondHalf = m_timeTicks % year >= halfYear;
                float f = m_timeTicks%halfYear/(float) halfYear;
                return secondHalf ? 1 - f : f;
            }
        }

        public TimeSpan YearLength { get; set; }

        public float Light
        {
            get
            {
                long day = DayLength.Ticks;
                long halfDay = day/2;
                bool secondHalf = m_timeTicks%day >= halfDay;
                float f = m_timeTicks%halfDay/(float) halfDay;
                return secondHalf ? 1 - f : f;
            }
        }

        public TimeSpan DayLength { get; set; }

        public void RegisterHeatSource(IHeatSource heatSource)
        {
            Atmosphere.RegisterHeatSource(heatSource);
        }

        public void UnregisterHeatSource(IHeatSource heatSource)
        {
            Atmosphere.UnregisterHeatSource(heatSource);
        }
    }
}
