using System.Collections.Generic;

namespace Utils
{
    public static class ControlMapper
    {
        public enum ControlMode
        {
            Simple,
            KeyboardMouse
        };

        private static readonly Dictionary<string, int> m_controlIndexes = new Dictionary<string, int>();

        private static ControlMode m_mode;
        public static ControlMode Mode
        {
            get
            {
                return m_mode;
            }
            set
            {
                m_mode = value;
                UpdateMapping();
            }
        }

        private static void UpdateMapping()
        {
            switch (Mode)
            {
                case ControlMode.KeyboardMouse:
                    {
                        m_controlIndexes["forward"] = 87; // W
                        m_controlIndexes["backward"] = 83; // S
                        m_controlIndexes["rot_left"] = 65; // A
                        m_controlIndexes["rot_right"] = 68; // D
                        m_controlIndexes["left"] = 69; // Q
                        m_controlIndexes["right"] = 81; // E

                        m_controlIndexes["fof_up"] = 73; // I
                        m_controlIndexes["fof_left"] = 76; // J
                        m_controlIndexes["fof_down"] = 75; // K
                        m_controlIndexes["fof_right"] = 74; // L

                        m_controlIndexes["interact"] = 66; // B
                        m_controlIndexes["use"] = 78; // N
                        m_controlIndexes["pickup"] = 77; // M
                        break;
                    }
                default:
                    {
                        m_controlIndexes["forward"] = 0;
                        m_controlIndexes["backward"] = 1;
                        m_controlIndexes["left"] = 2;
                        m_controlIndexes["right"] = 3;
                        m_controlIndexes["rot_left"] = 4;
                        m_controlIndexes["rot_right"] = 5;
                        m_controlIndexes["fof_left"] = 6;
                        m_controlIndexes["fof_right"] = 7;
                        m_controlIndexes["fof_up"] = 8;
                        m_controlIndexes["fof_down"] = 9;
                        m_controlIndexes["interact"] = 10;
                        m_controlIndexes["use"] = 11;
                        m_controlIndexes["pickup"] = 12;
                        break;
                    }
            }
        }

        public static int Idx(string key)
        {
            return m_controlIndexes[key];
        }
    }
}
