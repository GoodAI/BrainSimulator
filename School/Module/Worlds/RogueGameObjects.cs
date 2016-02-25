using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using GoodAI.Modules.School.Common;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;

namespace GoodAI.Modules.School.Worlds
{
    public class RogueAgent : MovableGameObject
    {
        public RogueAgent(PointF p, float scale = 1.0f)
            : base(GetDefaultTexturePath(), p, GetDefaultSize(), GameObjectType.Agent)
        {
            Size = new SizeF(Size.Width * scale, Size.Height * scale);
        }

        public RogueAgent(PointF p, string bitmapPath, float scale = 1.0f)
            : base(bitmapPath, p, GetDefaultSize(), GameObjectType.Agent)
        {
            Size = new SizeF(Size.Width * scale, Size.Height * scale);
        }

        public static string GetDefaultTexturePath()
        {
            return "Agent_TOP_blue_m.png";
        }

        public static SizeF GetDefaultSize()
        {
            return new SizeF(28, 28);
        }
    }

    public class RogueTeacher : AbstractTeacherInWorld
    {
        public readonly List<Actions> ActionsList;
        public enum Actions { MoveUp = 2, MoveDown = 3, MoveLeft = 0, MoveRight = 1, NoMove = 5 }

        public RogueTeacher(PointF p, float scale = 1.0f)
            : base(GetDefaultTexturePath(), p, GetDefaultSize(), GameObjectType.Teacher)
        {
            Size = new SizeF(Size.Width * scale, Size.Height * scale);

            ActionsList = new List<Actions>();
            m_currentMove = -1;
        }

        public RogueTeacher(PointF p, List<Actions> actions, float scale = 1.0f)
            : base(GetDefaultTexturePath(), p, GetDefaultSize(), GameObjectType.Teacher)
        {
            Size = new SizeF(Size.Width * scale, Size.Height * scale);

            Debug.Assert(actions != null);
            ActionsList = actions;
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
                return GetActionVector(Actions.NoMove); ;
            }
            return GetActionVector(ActionsList[m_currentMove++]);
        }

        public static Actions GetRandomAction(Random rndGen, int degreeOfFreedom)
        {
            Array values = Enum.GetValues(typeof(Actions));
            return (Actions)values.GetValue(rndGen.Next(degreeOfFreedom * 2));
        }

        public override void Stop()
        {
            BitmapPath = GetDefaultStopTexture();
        }

        public override bool IsDone()
        {
            return m_currentMove >= ActionsList.Count;
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
            return ActionsList.Count;
        }

        public static string GetDefaultTexturePath()
        {
            return "Agent_TOP_red_m.png";
        }

        public static string GetDefaultStopTexture()
        {
            return "Agent_TOP_green_m.png";
        }

        public static Size GetDefaultSize()
        {
            return new Size(24, 24);
        }
    }

    public class RogueWall : GameObject
    {
        public RogueWall(PointF p, float scale = 1.0f)
            : base(GetDefaultTexturePath(), p, GetDefaultSize(), GameObjectType.Obstacle)
        {
            Size = new SizeF(Size.Width * scale, Size.Height * scale);
        }

        public static string GetDefaultTexturePath()
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
        public RogueTarget(PointF p, float scale = 1.0f)
            : base(GetDefaultTexture(), p, GetDefaultSize(), GameObjectType.NonColliding)
        {
            Size = new SizeF(Size.Width * scale, Size.Height * scale);
        }

        public static string GetDefaultTexture()
        {
            return "Target_TOP.png";
        }

        public static Size GetDefaultSize()
        {
            return RoguelikeWorld.DEFAULT_GRID_SIZE;
        }
    }

    public class RogueMovableTarget : MovableGameObject
    {
        public RogueMovableTarget(PointF p, float scale = 1.0f)
            : base(GetDefaultTexture(), p, GetDefaultSize(), GameObjectType.NonColliding)
        {
            Size = new SizeF(Size.Width * scale, Size.Height * scale);
        }

        public static string GetDefaultTexture()
        {
            return "Target_TOP.png";
        }

        public static Size GetDefaultSize()
        {
            return RoguelikeWorld.DEFAULT_GRID_SIZE;
        }
    }

    public class RogueDoor : GameObject, ISwitchable
    {
        private bool _isOn;
        public bool IsOn
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

        public RogueDoor(PointF p, bool isClosed = true, float scale = 1.0f)
            : base(GetDefaultTexture(), p, GetDefaultSize(), GameObjectType.ClosedDoor)
        {
            Size = new SizeF(Size.Width * scale, Size.Height * scale);

            Switch(!isClosed);
        }

        public void Switch(bool on)
        {
            IsOn = on;
            if (on)
            {
                BitmapPath = @"Gate_Open_m.png";
                Type = GameObjectType.OpenedDoor;
            }
            else
            {
                BitmapPath = @"Gate_Close_m.png";
                Type = GameObjectType.ClosedDoor;

            }
        }

        public bool SwitchOnCollision()
        {
            return false;
        }

        public void Switch()
        {
            Switch(!_isOn);
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
        public bool IsOn
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

        public RogueLever(PointF p, ISwitchable switchableObject = null, bool isOn = false, float scale = 1.0f)
            : base(GetDefaultTexture(), p, GetDefaultSize(), GameObjectType.Obstacle)
        {
            Size = new SizeF(Size.Width * scale, Size.Height * scale);

            SwitchableObject = switchableObject;
            IsOn = isOn;
            if (isOn)
            {
                BitmapPath = @"Button_ON.png";
            }
            else
            {
                BitmapPath = @"Button_OFF.png";
            }
        }

        public void Switch()
        {
            Switch(!_isOn);
        }

        public void Switch(bool on)
        {
            SwitchableObject.Switch();
            IsOn = on;
            if (on)
            {
                BitmapPath = @"Button_ON.png";
            }
            else
            {
                BitmapPath = @"Button_OFF.png";
            }
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
            return RoguelikeWorld.DEFAULT_GRID_SIZE;
        }
    }

    public class RogueKiller : GameObject
    {
        public RogueKiller(PointF p, float scale = 1.0f)
            : base(GetDefaultTexture(), p, GetDefaultSize(), GameObjectType.NonColliding)
        {
            Size = new SizeF(Size.Width * scale, Size.Height * scale);
        }

        public static string GetDefaultTexture()
        {
            return "Lava_Surface.png";
        }

        public static Size GetDefaultSize()
        {
            return RoguelikeWorld.DEFAULT_GRID_SIZE;
        }
    }

    public class RogueMovableKiller : MovableGameObject
    {
        public RogueMovableKiller(PointF p, float scale = 1.0f)
            : base(GetDefaultTexture(), p, GetDefaultSize(), GameObjectType.NonColliding)
        {
            Size = new SizeF(Size.Width * scale, Size.Height * scale);
        }

        public static string GetDefaultTexture()
        {
            return "Enemy.png";
        }

        public static Size GetDefaultSize()
        {
            return RoguelikeWorld.DEFAULT_GRID_SIZE;
        }
    }
}