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
        public List<IAutoupdateable> NewAutoupdateables { get; private set; }
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
            m_timeTicks = new DateTime(2000, 1, 1, 0, 0, 0).Ticks;
            YearLength = TimeSpan.FromMinutes(2);
            DayLength = TimeSpan.FromSeconds(10);
            NewAutoupdateables = new List<IAutoupdateable>();
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
                if (LayerType.Obstacles.HasFlag(layer.LayerType))
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


        public void RegisterToAutoupdate(IAutoupdateable actor)
        {
            NewAutoupdateables.Add(actor);
        }

        public void UpdateLayers()
        {
            foreach (ITileLayer tileLayer in TileLayers)
            {
                tileLayer.UpdateTileStates(this);
            }
        }



        #region Query helpers

        public IEnumerable<GameActorPosition> ActorsAt(Vector2 position, LayerType type = LayerType.All, float width = 0.5f)
        {
            IEnumerable<ILayer<GameActor>> selectedLayers = Layers.Where(t => type.HasFlag(t.LayerType));
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

        public IEnumerable<Vector2I> FreePositionsAround(Vector2I position, LayerType type = LayerType.All)
        {
            return Neighborhoods.ChebyshevNeighborhood(position).Where(x => IsCoordinateFree(x, type));
        }

        public bool MoveToOtherLayer(GameActorPosition actor, LayerType layerType)
        {
            Remove(actor);
            actor.Layer = layerType;
            Add(actor);
            return true;
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

        private bool IsCoordinateFree(Vector2I position, LayerType type)
        {
            return !ActorsAt(new Vector2(position), type).Any();
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

        #endregion

        #region Tile manipulation

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

        public void Remove(GameActorPosition target)
        {
            ReplaceWith(target, null);
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

        #endregion

        #region Temperature

        public float Temperature(Vector2 position)
        {
            return Atmosphere.Temperature(position);
        }

        public void RegisterHeatSource(IHeatSource heatSource)
        {
            Atmosphere.RegisterHeatSource(heatSource);
        }

        public void UnregisterHeatSource(IHeatSource heatSource)
        {
            Atmosphere.UnregisterHeatSource(heatSource);
        }

        #endregion

        #region Atlas time

        public TimeSpan YearLength { get; set; }
        public TimeSpan DayLength { get; set; }

        public float SummerGradient { get; private set; }

        // 0 is Dec/Jan
        // 1 is Jun/Jul
        public float Summer { get; private set; }
        public float Day { get; private set; }
        public bool IsWinterEnabled { get; set; } = true;

        public void IncrementTime(int days = 0, int hours = 0, int minutes = 0, int seconds = 10, int millis = 0)
        {
            IncrementTime(new TimeSpan(days, hours, minutes, seconds, millis));
        }

        public void IncrementTime(TimeSpan timeSpan)
        {
            m_timeTicks += timeSpan.Ticks;

            long year = YearLength.Ticks;
            long halfYear = year / 2;
            bool secondHalf = m_timeTicks % year >= halfYear;

            SummerGradient = secondHalf ? -1 : 1; // summer or fall

            float f = m_timeTicks % halfYear / (float)halfYear;
            Summer = secondHalf ? 1 - f : f;


            const int logEvery = 24;  // Only log every half month
            const int logTimeAmount = 365 / 2;  // Logging lasts for two days
            float logInterval = Summer % (1f / logEvery) * logEvery; // A number between 0,1

            //if ((int)(logInterval * logTimeAmount) == 0) // Logging happens only in the beggining of the interval
            //Log.Instance.Debug(string.Format("Day: {0} \tYear: {1} \tGradient: {2}", Day, Summer, SummerGradient));


            long day = DayLength.Ticks;
            long halfDay = day / 2;
            secondHalf = m_timeTicks % day >= halfDay;
            f = m_timeTicks % halfDay / (float)halfDay;
            Day = secondHalf ? 1 - f : f;
        }

        #endregion
    }
}
