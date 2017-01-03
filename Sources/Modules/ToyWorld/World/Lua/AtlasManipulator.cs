using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using VRageMath;
using World.Atlas;
using World.Atlas.Layers;
using World.GameActors;
using World.GameActors.Tiles;

namespace World.Lua
{
    class AtlasManipulator
    {
        private IAtlas m_atlas;
        private LuaExecutor m_luaExecutor;

        public AtlasManipulator(LuaExecutor luaExecutor, IAtlas atlas)
        {
            this.m_luaExecutor = luaExecutor;
            this.m_atlas = atlas;
        }

        public void CreateTile(string type, string layer, float x, float y)
        {
            Type t = GameActor.GetType(type);
            if (t == null) throw new Exception("Object of type " + type + " not found in assembly.");

            GameActor ga = (GameActor)Activator.CreateInstance(t);
            if (t.IsSubclassOf(typeof(Tile)))
            {
                int xi = (int) x;
                int yi = (int) y;
                m_atlas.Add(new GameActorPosition(ga, new Vector2(xi, yi), Layer(layer)));
            }
            else
            {
                throw new Exception("Object of type " + t + " is not subclass of Tile.");
            }
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
    }
}
