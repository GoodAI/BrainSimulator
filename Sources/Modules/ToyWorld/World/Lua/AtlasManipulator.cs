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

        public void CreateGameActor(string type, float x, float y, object[] properties)
        {
            Type t = GameActor.GetType(type);
            if (t == null) throw new Exception("Object of type " + t + " not founded in assembly.");

            GameActor ga = (GameActor)Activator.CreateInstance(t);
            if (t.IsSubclassOf(typeof(Tile)))
            {
                int xi = (int) x;
                int yi = (int) y;
            }
            else if (t.IsSubclassOf(typeof(Tile)))
            {
                m_atlas.Add(new GameActorPosition(ga, new Vector2(x, y), LayerType.Object));
            }
            else
            {
                throw new Exception("Object of type " + t + " is not Tile or Object.");
            }
        }
    }
}
