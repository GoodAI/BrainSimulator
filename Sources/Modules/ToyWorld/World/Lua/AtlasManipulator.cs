using System;
using System.Collections.Generic;
using VRageMath;
using World.Atlas;
using World.Atlas.Layers;
using World.GameActors;
using World.GameActors.GameObjects;
using World.GameActors.Tiles;

namespace World.Lua
{
    /// <summary>
    /// This class contains functions optimized for calling from Lua code.
    /// </summary>
    public class AtlasManipulator
    {
        private readonly IAtlas m_atlas;

        public AtlasManipulator(IAtlas atlas)
        {
            m_atlas = atlas;
        }

        public void CreateTile(string type, string layer, float x, float y)
        {
            Type t = GameActor.GetGameActorType(type);
            if (t == null) throw new Exception("Object of type " + type + " not found in assembly.");

            int xi = (int) x;
            int yi = (int) y;
            GameActor ga;

            if (t.IsSubclassOf(typeof(DynamicTile)))
            {
                ga = (GameActor)Activator.CreateInstance(t, new Vector2I(xi, yi));
            }
            else if (t.IsSubclassOf(typeof(Tile)))
            {
                ga = (GameActor)Activator.CreateInstance(t);
            }
            else
            {
                throw new Exception("Object of type " + t + " is not subclass of Tile.");
            }

            m_atlas.Add(new GameActorPosition(ga, new Vector2(xi, yi), Layer(layer)));
        }

        public void DestroyTile(string layer, int x, int y)
        {
            m_atlas.Remove(new GameActorPosition(null, new Vector2(x,y), Layer(layer)));
        }

        public LayerType Layer(string layer)
        {
            LayerType t;
            bool tryParse = Enum.TryParse(layer, true, out t);
            if (!tryParse) throw new Exception("Layer " + layer +" was not found.");
            return t;
        }

        public Vector2 WhereIsObject(string name)
        {
            IObjectLayer layer = (IObjectLayer) m_atlas.GetLayer(LayerType.Object);
            return layer.GetGameObject(name).Position;
        }

        public IGameObject GetObject(string name)
        {
            IObjectLayer layer = (IObjectLayer)m_atlas.GetLayer(LayerType.Object);
            return layer.GetGameObject(name);
        }

        public static Vector2 Vector(float x, float y)
        {
            return new Vector2(x, y);
        }

        internal static GameActorPosition GetNearest(int x, int y, string type, IAtlas atlas)
        {
            Type t = GameActor.GetGameActorType(type);

            for (int i = 1; i < 20; i++)
            {
                IEnumerable<Vector2I> vonNeumannNeighborhood = Neighborhoods.VonNeumannNeighborhood(new Vector2I(x, y), i);

                foreach (var xy in vonNeumannNeighborhood)
                    foreach (GameActorPosition gameActorPosition in atlas.ActorsAt((Vector2)xy))
                        if (gameActorPosition.Actor.GetType() == t)
                            return gameActorPosition;
            }
            return null;
        }

        public GameActorPosition GetNearest(int x, int y, string type)
        {
            return GetNearest(x, y, type, m_atlas);
        }

        public string InRoom(IGameObject o)
        {
            return m_atlas.AreasCarrier.Room(o.Position)?.Name;
        }

        public string InArea(IGameObject o)
        {
            return m_atlas.AreasCarrier.AreaName(o.Position);
        }
    }
}
