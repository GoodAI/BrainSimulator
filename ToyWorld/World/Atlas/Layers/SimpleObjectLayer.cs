using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using VRageMath;
using World.GameActors;
using World.GameActors.GameObjects;
using World.Physics;

namespace World.Atlas.Layers
{
    public class SimpleObjectLayer : IObjectLayer
    {
        private List<IGameObject> GameObjects { get; set; }

        public bool Render { get; set; }

        public float Thickness { get; private set; }
        public float SpanIntervalFrom { get; private set; }
        public float SpanIntervalTo { get { return SpanIntervalFrom + Thickness; } }

        public LayerType LayerType { get; set; }

        public SimpleObjectLayer(LayerType layerType)
        {
            Thickness = 0.7f;
            SpanIntervalFrom = 1.05f;

            LayerType = layerType;
            GameObjects = new List<IGameObject>();
        }

        public GameObject GetActorAt(int x, int y)
        {
            return GetGameObjects(new RectangleF(x + 0.5f, y + 0.5f, 1, 1)).FirstOrDefault() as GameObject;
        }

        public List<IGameObject> GetGameObjects(Vector2I tilePosition)
        {
            return GetGameObjects(new RectangleF(new Vector2(tilePosition) + Vector2.One / 2, Vector2.One));
        }

        public GameObject GetActorAt(Shape shape)
        {
            foreach (IGameObject gameObject in GameObjects)
            {
                if (gameObject.PhysicalEntity.Shape.CollidesWith(shape))
                {
                    return gameObject as GameObject;
                }
            }
            return null;
        }

        public GameObject GetActorAt(Vector2I position)
        {
            return GetActorAt(position.X, position.Y);
        }

        public List<IGameObject> GetGameObjects(RectangleF rectangle)
        {
            List<IGameObject> list = new List<IGameObject>();

            foreach (IGameObject o in GameObjects)
            {
                var gameObject = (GameObject)o;
                IPhysicalEntity physicalEntity = gameObject.PhysicalEntity;
                RectangleF r;
                RectangleF cover = physicalEntity.CoverRectangle();
                RectangleF.Intersect(ref cover, ref rectangle, out r);
                if (r.Size.Length() > 0f)
                {
                    list.Add(gameObject);
                }
            }

            return list;
        }

        public List<IGameObject> GetGameObjects()
        {
            var newList = new List<IGameObject>();
            newList.AddRange(GameObjects);
            return newList;
        }

        public List<IGameObject> GetGameObjects(Circle circle)
        {
            var list = new List<IGameObject>();

            foreach (IGameObject o in GameObjects)
            {
                var gameObject = (GameObject)o;
                if (circle.Include(gameObject.Position))
                {
                    list.Add(gameObject);
                }
            }

            return list;
        }

        public bool AddGameObject(IGameObject gameObject)
        {
            GameObjects.Add(gameObject);
            return true;
        }

        public List<IPhysicalEntity> GetPhysicalEntities(Circle circle)
        {
            return GetGameObjects(circle).Select(x => x.PhysicalEntity).ToList();
        }

        public List<IPhysicalEntity> GetPhysicalEntities()
        {
            return GetGameObjects().Select(x => x.PhysicalEntity).ToList();
        }

        public List<IPhysicalEntity> GetPhysicalEntities(RectangleF rectangle)
        {
            return GetGameObjects(rectangle).Select(x => x.PhysicalEntity).ToList();
        }

        public bool ReplaceWith<T>(GameActorPosition original, T replacement)
        {
            GameObject item = (GameObject)original.Actor;
            if (item != original.Actor) return false;

            GameObjects.Remove(item);

            if (!(replacement is GameObject)) return true;
            GameObjects.Add(replacement as GameObject);
            return true;
        }

        public bool Add(GameActorPosition gameActorPosition)
        {
            IGameObject gameObject = gameActorPosition.Actor as IGameObject;
            Debug.Assert(gameObject != null, "gameObject != null");

            Vector2 position = gameActorPosition.Position;
            gameObject.Position = position;

            GameObjects.Add(gameObject);
            return true;
        }
    }
}
