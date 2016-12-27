using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text.RegularExpressions;
using System.Threading;
using VRageMath;
using World.Atlas;
using World.GameActors;
using World.GameActors.GameObjects;
using World.GameActors.Tiles;

namespace World.Lua
{
    public class AvatarHelper
    {
        private readonly IAtlas m_atlas;
        private readonly AutoResetEvent m_scriptSynchronization;
        public IAvatar CurrentAvatar;

        public AvatarHelper(NLua.Lua lua, IAtlas atlas, AutoResetEvent scriptSynchronization)
        {
            m_atlas = atlas;
            m_scriptSynchronization = scriptSynchronization;
            lua.RegisterFunction("help", typeof(AvatarHelper).GetMethod("Help"));
            lua.RegisterFunction("Vector", typeof(AvatarHelper).GetMethod("Vector"));
            lua["atlas"] = atlas;
            CurrentAvatar = atlas.Avatars.Values.First();
            lua["avatar"] = CurrentAvatar;
        }

        public static string Help(object o)
        {
            Type type = o.GetType();
            PropertyInfo[] propertyInfos = type.GetProperties();
            string propertiesJoined = string.Join(",\n\t\t", propertyInfos.Select(x => x.ToString()));
            propertiesJoined = Regex.Replace(propertiesJoined, @"\w+\.", "");
            MethodInfo[] methodInfos = type.GetMethods();
            string methodsJoined = string.Join(",\n\t\t", methodInfos.Select(x => x.ToString()));
            methodsJoined = Regex.Replace(methodsJoined, @"\w+\.", "");
            return "Name: \"" + type.Name + "\"\n\tProperties: { " + propertiesJoined + " } " + "\n\tMethods: { " +
                   methodsJoined + " }.";
        }

        public void Do(Func<object[], bool> stepFunc, params object[] parameters)
        {
            for (int i = 0; i < 100000; i++)
            {
                m_scriptSynchronization.WaitOne();
                object o = stepFunc(parameters);
                bool end = (bool)o;
                if (end)
                {
                    return;
                }
            }
            throw new Exception("Too long script execution.");
        }

        public void Repeat(Action<object[]> stepFunc, int repetitions, params object[] parameters)
        {
            for (int i = 0; i < repetitions; i++)
            {
                m_scriptSynchronization.WaitOne();
                stepFunc(parameters);
            }
        }

        public void Perform(Action<object[]> stepFunc, params object[] parameters)
        {
            m_scriptSynchronization.WaitOne();
            stepFunc(parameters);
        }

        public void Goto(Vector2 position)
        {
            // ah:Goto(Vector(20,30))
            Func<object[], bool> f = GotoI;
            Do(f, position, 0.1f);
        }

        public void Goto(int x, int y)
        {
            // ah:Goto(10,10)
            Func<object[], bool> f = GotoI;
            Do(f, new Vector2(x + 0.5f, y + 0.5f), 0.1f);
        }

        public void Goto(Vector2 position, float distance)
        {
            if (distance <= 0) throw new ArgumentException("Distance must be positive.");
            // ah:Goto("Pinecone", 2)
            Func<object[], bool> f = GotoI;
            Do(f, position, distance);
        }

        public void Goto(string type)
        {
            // ah:Goto("Pinecone")
            GameActorPosition gameActorPosition = GetNearest(CurrentAvatar.Position, type);
            if (gameActorPosition == null) return;
            Vector2 position = gameActorPosition.Position;
            if (gameActorPosition.Actor is Tile)
            {
                position = Tile.Center((Vector2I)position);
            }
            Goto(position);
        }

        public void GotoR(int x, int y)
        {
            // ah:Goto(10,10)
            Vector2 cuPos = CurrentAvatar.Position;
            Func<object[], bool> f = GotoI;
            Vector2 relative = new Vector2(x, y);
            Do(f, cuPos + relative, 0.01f);
        }

        public void GotoR(int x, int y, float distance)
        {
            if (distance <= 0) throw new ArgumentException("Distance must be positive.");
            // ah:Goto(10,10,0.5)
            Vector2 cuPos = CurrentAvatar.Position;
            Func<object[], bool> f = GotoI;
            Vector2 relative = new Vector2(x, y);
            Do(f, cuPos + relative, distance);
        }

        public void Heading(string heading)
        {
            string lower = heading.ToLower();
            string trim = lower.Trim();
            float r = 0;
            switch (trim[0])
            {
                case 'n':
                    r = 0f;
                    break;
                case 'w':
                    r = MathHelper.Pi / 2;
                    break;
                case 's':
                    r = MathHelper.Pi;
                    break;
                case 'e':
                    r = MathHelper.Pi - 2;
                    break;
                default:
                    throw new ArgumentException("Direction was not recognized.");
            }
            RotateTo(r);
        }

        public void TeleportAvatar(float x, float y)
        {
            CurrentAvatar.Position = new Vector2(x, y);
        }

        public void TeleportAvatar(Vector2 vector)
        {
            CurrentAvatar.Position = vector;
        }

        public void TeleportAvatarRelatively(float x, float y)
        {
            CurrentAvatar.Position += new Vector2(x, y);
        }

        public void RotateTo(float finalAngle, float precision)
        {
            if (precision <= 0) throw new ArgumentException("Precision must be positive.");
            // luaTest:RotateRight(100)
            Func<object[], bool> f = RotateToI;
            Do(f, finalAngle, precision);
        }

        public void RotateTo(float finalAngle)
        {
            // luaTest:RotateRight(100)
            Func<object[], bool> f = RotateToI;
            Do(f, finalAngle, MathHelper.Pi / 16);
        }

        private bool GotoI(params object[] parameters)
        {
            Vector2 targetPosition = (Vector2)parameters[0];
            float targetDistance = (float)parameters[1];
            float distance = Vector2.Distance(targetPosition, CurrentAvatar.Position);
            if (targetDistance >= distance) return true;
            float targetAngle = Vector2.AngleTo(CurrentAvatar.Position, targetPosition);
            RotateToI(targetAngle, MathHelper.Pi / 16);
            float wrappedAngle = MathHelper.WrapAngle(targetAngle - CurrentAvatar.Rotation);
            float maxAngle = MathHelper.Pi / 5;
            if (-maxAngle < wrappedAngle && wrappedAngle < maxAngle)
            {
                if (distance > 1)
                {
                    CurrentAvatar.DesiredSpeed = 1;
                }
                else
                {
                    CurrentAvatar.DesiredSpeed = distance * 0.3f;
                }
            }
            return false;
        }

        private bool RotateToI(params object[] a)
        {
            float targetRotation = (float)a[0];
            float precision = (float)a[1];
            if (Math.Abs(CurrentAvatar.Rotation - targetRotation) < MathHelper.Pi / 16) return true;
            if (MathHelper.WrapAngle(CurrentAvatar.Rotation - targetRotation) < 0)
            {
                CurrentAvatar.DesiredLeftRotation = 1;
            }
            else
            {
                CurrentAvatar.DesiredLeftRotation = -1;
            }
            return false;
        }

        public GameActorPosition GetNearest(Vector2 position, string type)
        {
            Assembly[] assemblies = AppDomain.CurrentDomain.GetAssemblies();
            Assembly assembly = assemblies.First(x => x.FullName.StartsWith("World,"));
            Type t = assembly.GetTypes().FirstOrDefault(x => x.Name == type);

            for (int i = 1; i < 20; i++)
            {
                IEnumerable<Vector2I> chebyshevNeighborhood = Neighborhoods.VonNeumannNeighborhood((Vector2I)position, i);

                foreach (var x in chebyshevNeighborhood)
                {
                    foreach (GameActorPosition gameActorPosition in m_atlas.ActorsAt((Vector2)x))
                    {
                        if (gameActorPosition.Actor.GetType() == t)
                        {
                            return gameActorPosition;
                        }
                    }
                }
            }
            return null;
        }

        public GameActorPosition GetNearest(IGameObject gameObject, string type)
        {
            return GetNearest(gameObject.Position, type);
        }

        public static Vector2 Vector(float x, float y)
        {
            return new Vector2(x, y);
        }
    }
}