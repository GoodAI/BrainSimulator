using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using GoodAI.Modules.School.Common;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace GoodAI.Modules.School.Worlds
{
    public class RogueAgent : MovableGameObject
    {
        public RogueAgent(Point p, float size = 1.0f)
            : base(GameObjectType.Teacher, GetDefaultTexture(), p.X, p.Y,
            (int)(GetDefaultSize().Width * size), (int)(GetDefaultSize().Height * size)) { }

        public RogueAgent(Point p, string path, float size = 1.0f)
            : base(GameObjectType.Teacher, path, p.X, p.Y,
            (int)(GetDefaultSize().Width * size), (int)(GetDefaultSize().Height * size)) { }

        public static string GetDefaultTexture(){
            return "Agent_TOP_blue_m.png";
        }

        public static Size GetDefaultSize()
        {
            return new Size(28, 28);
        }
    }

    public class RogueTeacher : AbstractTeacherInWorld
    {
        public List<Actions> m_actionsList;
        public enum Actions { MoveUp = 2, MoveDown = 3, MoveLeft = 0, MoveRight = 1 , NoMove = 5}

        public RogueTeacher(Point p, float size = 1.0f)
            : base(GameObjectType.Teacher, GetDefaultTexture(), p.X, p.Y,
            (int)(GetDefaultSize().Width * size), (int)(GetDefaultSize().Height * size))
        {
            m_actionsList = new List<Actions>();
            m_currentMove = -1;
        }

        public RogueTeacher(Point p, List<Actions> actions, float size = 1.0f)
            : base(GameObjectType.Teacher, GetDefaultTexture(), p.X, p.Y,
            (int)(GetDefaultSize().Width * size), (int)(GetDefaultSize().Height * size))
        {
            m_actionsList = actions;
            m_currentMove = -1;
        }

        public override float[] CurrentAction()
        {
            if (m_currentMove < 0)
            {
                m_currentMove++;
                return GetActionVector(Actions.NoMove);
            }
            if (IsDone())
            {
                Stop();
                return GetActionVector(Actions.NoMove);;
            }
            return GetActionVector(m_actionsList[m_currentMove++]);
        }

        public static Actions GetRandomAction(Random rndGen, int degreeOfFreedom)
        {
            Array values = Enum.GetValues(typeof(Actions));
            return (Actions)values.GetValue(rndGen.Next(degreeOfFreedom * 2));
        }

        public override void Stop()
        {
            bitmapPath = GetDefaultStopTexture();
        }

        public override bool IsDone()
        {
            return m_currentMove >= m_actionsList.Count;
        }

        public override void Reset()
        {
            m_currentMove = 0;
        }

        private float[] GetActionVector(Actions a)
        {
            switch (a)
            {
                case Actions.NoMove:
                    return new float[] { 0f, 0f, 0f, 0f };
                case Actions.MoveLeft:
                    return new float[] { 0f, 1f, 0f, 0f };
                case Actions.MoveRight:
                    return new float[] { 1f, 0f, 0f, 0f };
                case Actions.MoveUp:
                    return new float[] { 0f, 0f, 1f, 0f };
                case Actions.MoveDown:
                    return new float[] { 0f, 0f, 0f, 1f };
            }
            throw new ArgumentException("Uknown action");
        }

        public override int ActionsCount()
        {
            return m_actionsList.Count;
        }

        public static string GetDefaultTexture()
        {
            return "Agent_TOP_red_m.png";
        }

        public static string GetDefaultStopTexture()
        {
            return "Agent_TOP_green_m.png";
        }

        public static Size GetDefaultSize()
        {
            return new Size(24,24);
        }
    }

    public class RogueWall : GameObject
    {
        public RogueWall(Point p, float size = 1.0f)
            : base(GameObjectType.Obstacle, GetDefaultTexture(), p.X, p.Y,
            (int)(GetDefaultSize().Width * size), (int)(GetDefaultSize().Height * size)) { }

        public static string GetDefaultTexture()
        {
            return "Armor_Block.png";
        }

        public static Size GetDefaultSize()
        {
            return RoguelikeWorld.DEFAULT_GRID_SIZE;
        }
    }

    public class RogueTarget : GameObject
    {
        public RogueTarget(Point p, float size = 1.0f)
            : base(GameObjectType.NonColliding, GetDefaultTexture(), p.X, p.Y,
            (int)(GetDefaultSize().Width * size), (int)(GetDefaultSize().Height * size)) { }

        public static string GetDefaultTexture()
        {
            return "Target_TOP.png";
        }

        public static Size GetDefaultSize()
        {
            return new Size(32, 32);
        }
    }

    public class RogueMovableTarget : MovableGameObject
    {
        public RogueMovableTarget(Point p, float size = 1.0f)
            : base(GameObjectType.NonColliding, GetDefaultTexture(), p.X, p.Y,
            (int)(GetDefaultSize().Width * size), (int)(GetDefaultSize().Height * size)) { }

        public static string GetDefaultTexture()
        {
            return "Target_TOP.png";
        }

        public static Size GetDefaultSize()
        {
            return new Size(32, 32);
        }
    }

    public class RogueDoor : GameObject, ISwitchable
    {
        private bool _isOn;
        public bool isOn
        {
            get
            {
                return _isOn;
            }
            set
            {
                _isOn = value;
            }
        }

        public RogueDoor(Point p, bool isClosed = true, float size = 1.0f)
            : base(GameObjectType.ClosedDoor, GetDefaultTexture(), p.X, p.Y,
            (int)(GetDefaultSize().Width * size), (int)(GetDefaultSize().Height * size))
        {
            if (!isClosed)
            {
                SwitchOn();
            }
            else
            {
                SwitchOff();
            }
        }

        public void SwitchOn()
        {
            bitmapPath = @"Gate_Open_m.png";
            type = GameObjectType.OpenedDoor;
            
            _isOn = true;
        }

        public void SwitchOff()
        {
            bitmapPath = @"Gate_Open_m.png";
            type = GameObjectType.ClosedDoor;
            _isOn = false;
        }

        public bool SwitchOnCollision()
        {
            return false;
        }

        public void Switch()
        {
            if (_isOn)
            {
                SwitchOff();
            }
            else
            {
                SwitchOn();
            }
        }

        public static string GetDefaultTexture()
        {
            return @"Gate_Close_m.png";
        }

        public static Size GetDefaultSize()
        {
            return RoguelikeWorld.DEFAULT_GRID_SIZE;
        }
    }

    public class RogueLever : GameObject, ISwitchable
    {
        ISwitchable SwitchableObject;

        private bool _isOn;
        public bool isOn
        {
            get
            {
                return _isOn;
            }
            set
            {
                _isOn = value;
            }
        }

        public RogueLever(Point p, ISwitchable switchableObject = null, float size = 1.0f)
            : base(GameObjectType.Obstacle, GetDefaultTexture(), p.X, p.Y,
            (int)(GetDefaultSize().Width * size), (int)(GetDefaultSize().Height * size))
        {
            this.SwitchableObject = switchableObject;
        }

        public void Switch()
        {
            if (isOn)
            {
                SwitchOff();
            }
            else
            {
                SwitchOn();
            }
            SwitchableObject.Switch();
        }

        public void SwitchOn()
        {
            isOn = true;
            bitmapPath = @"Button_ON.png";
            type = GameObjectType.OpenedDoor;
            SwitchableObject.SwitchOn();
        }

        public void SwitchOff()
        {
            isOn = false;
            bitmapPath = @"Button_OFF.png";
            type = GameObjectType.ClosedDoor;
            SwitchableObject.SwitchOff();
        }

        public bool SwitchOnCollision()
        {
            return true;
        }

        public static string GetDefaultTexture()
        {
            return @"Button_ON.png";
        }

        public static Size GetDefaultSize()
        {
            return new Size(32, 32);
        }
    }

    public class RogueKiller : GameObject
    {
        public RogueKiller(Point p, float size = 1.0f)
            : base(GameObjectType.NonColliding, GetDefaultTexture(), p.X, p.Y,
            (int)(GetDefaultSize().Width * size), (int)(GetDefaultSize().Height * size)) { }

        public static string GetDefaultTexture()
        {
            return "Lava_Surface.png";
        }

        public static Size GetDefaultSize()
        {
            return new Size(32, 32);
        }
    }

    public class RogueMovableKiller : MovableGameObject
    {
        public RogueMovableKiller(Point p, float size = 1.0f)
            : base(GameObjectType.NonColliding, GetDefaultTexture(), p.X, p.Y,
            (int)(GetDefaultSize().Width * size), (int)(GetDefaultSize().Height * size)) { }

        public static string GetDefaultTexture()
        {
            return "Enemy.png";
        }

        public static Size GetDefaultSize()
        {
            return new Size(32, 32);
        }
    }
}