using System;
using System.Linq;
using VRageMath;
using World.Atlas;
using World.GameActors;
using World.GameActors.GameObjects;
using World.GameActors.Tiles;

namespace World.Lua
{
    public class AvatarCommander
    {
        private readonly IAtlas m_atlas;
        private readonly IAvatar m_currentAvatar;
        private readonly LuaExecutor m_le;

        private readonly float ToWorldSpeedCoef = new Physics.BasicAvatarMover().MaximumSpeed;

        public AvatarCommander(LuaExecutor ex, IAtlas atlas)
        {
            m_atlas = atlas;

            ex.State.RegisterFunction("Vector", typeof(AtlasManipulator).GetMethod("Vector"));

            m_currentAvatar = atlas.Avatars.Values.First();
            ex.State["avatar"] = m_currentAvatar;

            m_le = ex;
        }

        public void StrafeTo(float x, float y, float tolDist = 1e-5f)
        {
            m_le.Do(StrafeToI, new Vector2(x + 0.5f, y + 0.5f), tolDist);
        }

        /// <summary>
        /// Sets up avatars controls so that it performs a strafe-step along a straight line headed to the center of
        /// the given tile, in the next ToyWorld step. Synchronize properly when using this method from lua!
        /// </summary>
        /// <param name="x">x coord of a target tile</param>
        /// <param name="x">y coord of a target tile</param>
        /// <param name="tolDist">tolerance distance from the center of the target tile</param>
        /// <returns>true if avatar already is within tolerance distance to the target tile, false otherwise</returns>
        public bool StrafeToStep(int x, int y, float tolDist = 1e-5f)
        {
            return StrafeToI(new Vector2(x + 0.5f, y + 0.5f), tolDist);
        }

        public void GoTo(Vector2 position)
        {
            // ac:GoTo(Vector(20,30))
            Func<object[], bool> f = GoToI;
            m_le.Do(f, position, 0.1f);
        }

        public void GoTo(int x, int y)
        {
            // ac:GoTo(10,10)
            Func<object[], bool> f = GoToI;
            m_le.Do(f, new Vector2(x + 0.5f, y + 0.5f), 0.1f);
        }

        public void GoTo(Vector2 position, float distance)
        {
            // ac:GoTo("Pinecone", 2)
            if (distance <= 0) throw new ArgumentException("Distance must be positive.");
            Func<object[], bool> f = GoToI;
            m_le.Do(f, position, distance);
        }

        public void GoTo(string type, float distance = (float)0.1)
        {
            // ac:GoTo("Pinecone", 1.2)
            GameActorPosition gameActorPosition = GetNearest(type);
            if (gameActorPosition == null) throw new Exception("No object of type " + type + " found.");
            Vector2 position = gameActorPosition.Position;
            if (gameActorPosition.Actor is Tile)
            {
                position = Tile.Center((Vector2I)position);
            }
            GoTo(position, distance);
        }

        public void GoToR(int x, int y)
        {
            // ac:GoTo(10,10)
            Vector2 cuPos = m_currentAvatar.Position;
            Func<object[], bool> f = GoToI;
            Vector2 relative = new Vector2(x, y);
            m_le.Do(f, cuPos + relative, 0.01f);
        }

        public void GoToR(int x, int y, float distance)
        {
            if (distance <= 0) throw new ArgumentException("Distance must be positive.");
            // ac:GoTo(10,10,0.5)
            Vector2 cuPos = m_currentAvatar.Position;
            Func<object[], bool> f = GoToI;
            Vector2 relative = new Vector2(x, y);
            m_le.Do(f, cuPos + relative, distance);
        }

        /// <summary>
        /// Continuously rotates avatar to heading.
        /// </summary>
        /// <param name="heading">Target heading can be either "N", "S", "W" or "E".</param>
        public void RotateToHeading(string heading)
        {
            string lower = heading.ToLower();
            string trim = lower.Trim();
            float r;
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
                    r = - MathHelper.Pi / 2;
                    break;
                default:
                    throw new ArgumentException("Direction was not recognized.");
                    
            }
            RotateTo(r);
        }

        /// <summary>
        /// Continuously rotates avatar to given heading.
        /// </summary>
        /// <param name="heading">Target heading in degrees. 0 means north, 90 means right.</param>
        public void RotateToHeading(int heading)
        {
            float r = MathHelper.ToRadians(-heading);
            RotateTo(r);
        }

        /// <summary>
        /// Immediately put avatar to different position avatar to given location.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        public void Teleport(float x, float y)
        {
            m_currentAvatar.Position = new Vector2(x, y);
        }

        public void Teleport(Vector2 vector)
        {
            m_currentAvatar.Position = vector;
        }

        public void TeleportR(float x, float y)
        {
            m_currentAvatar.Position += new Vector2(x, y);
        }

        public void RotateTo(float finalAngle, float precision)
        {
            if (precision <= 0) throw new ArgumentException("Precision must be positive.");
            // ac:RotateRight(100)
            Func<object[], bool> f = RotateToI;
            m_le.Do(f, finalAngle, precision);
        }

        public void RotateTo(float finalAngle)
        {
            // ac:RotateRight(100)
            Func<object[], bool> f = RotateToI;
            m_le.Do(f, finalAngle, MathHelper.Pi / 160);
        }

        /// <summary>
        /// Step function (see LuaExecutor.Do(stepFunction, parameters))
        /// Continuously change direction towards given position and
        /// if the direction is precise enough, make step forward.
        /// Return true when close enough to target.
        /// </summary>
        /// <param name="parameters">
        /// Vector2 position of target.
        /// float distance from target - tolerance for stopping.
        /// </param>
        /// <returns></returns>
        private bool GoToI(params object[] parameters)
        {
            ResetAvatarsActions();
            Vector2 targetPosition = (Vector2)parameters[0];
            float targetDistance = (float)parameters[1];
            float distance = Vector2.Distance(targetPosition, m_currentAvatar.Position);
            if (targetDistance >= distance) return true;
            float targetAngle = -Vector2.AngleTo(m_currentAvatar.Position, targetPosition);
            RotateToI(targetAngle, MathHelper.Pi / 16);
            float wrappedAngle = MathHelper.WrapAngle(targetAngle - m_currentAvatar.Rotation);
            const float maxAngle = MathHelper.Pi / 5;
            if (-maxAngle < wrappedAngle && wrappedAngle < maxAngle)
            {
                m_currentAvatar.Direction = m_currentAvatar.Rotation;
                if (distance > 1)
                {
                    m_currentAvatar.DesiredSpeed = 1;
                }
                else
                {
                    m_currentAvatar.DesiredSpeed = distance * 0.3f;
                }
            }
            return false;
        }

        private bool StrafeToI(params object[] parameters)
        {
            ResetAvatarsActions();
            Vector2 targetPosition = (Vector2)parameters[0];
            float targetDistance = (float)parameters[1];

            float distance = Vector2.Distance(targetPosition, m_currentAvatar.Position);
            if (distance <= targetDistance) return true; //done

            m_currentAvatar.Direction = -Vector2.AngleTo(m_currentAvatar.Position, targetPosition);

            if (distance > ToWorldSpeedCoef)
            {
                m_currentAvatar.DesiredSpeed = 1;
            }
            else
            {
                // should reach exact target in one step
                m_currentAvatar.DesiredSpeed = distance / ToWorldSpeedCoef;
            }

            return false;
        }

        /// <summary>
        /// Step function (see LuaExecutor.Do(stepFunction, parameters))
        /// Continuously change direction towards given position.
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        private bool RotateToI(params object[] a)
        {
            ResetAvatarsActions();
            float targetRotation = (float) a[0];
            float precision = (float) a[1];
            float diff = CalculateDifferenceBetweenAngles(m_currentAvatar.Rotation, targetRotation);
            float absDiff = Math.Abs(diff);
            if (absDiff < precision) return true;
            if (MathHelper.WrapAngle(diff) < 0)
            {
                m_currentAvatar.DesiredLeftRotation = Math.Max((float)-Math.Sqrt(absDiff), -1);
            }
            else
            {
                m_currentAvatar.DesiredLeftRotation = Math.Min((float)Math.Sqrt(absDiff), 1);
            }
        
            return false;
        }

        public GameActorPosition GetNearest(string type)
        {
            return AtlasManipulator.GetNearest((int) m_currentAvatar.Position.X, (int) m_currentAvatar.Position.Y,
                type, m_atlas);
        }

        private float CalculateDifferenceBetweenAngles(float firstAngle, float secondAngle)
        {
            float difference = secondAngle - firstAngle;
            while (difference < -MathHelper.Pi) difference += MathHelper.Pi*2;
            while (difference > MathHelper.Pi) difference -= MathHelper.Pi*2;
            return difference;
        }

        public void ResetAvatarsActions()
        {
            m_currentAvatar.ResetControls();
        }

        public string CurrentRoom()
        {
            return m_atlas.AreasCarrier.Room(m_currentAvatar.Position)?.Name;
        }

        public string CurrentArea()
        {
            return m_atlas.AreasCarrier.AreaName(m_currentAvatar.Position);
        }
    }
}