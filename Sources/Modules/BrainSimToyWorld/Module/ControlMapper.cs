using System;
using System.Linq;
using System.Collections.Generic;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;

namespace GoodAI.ToyWorld
{
    public static class ControlMapper
    {
        public enum ControlMode
        {
            Autodetect,
            Simple,
            KeyboardMouse,
            SimpleTaskSpecific
        };

        private static Dictionary<string, int> simpleControls = new Dictionary<string, int>()
        {
            {"forward", 0},
            {"backward", 1},
            {"left", 2},
            {"right", 3},
            {"rot_left", 4},
            {"rot_right", 5},
            {"fof_left", 6},
            {"fof_right", 7},
            {"fof_up", 8},
            {"fof_down", 9},
            {"interact", 10},
            {"use", 11},
            {"pickup", 12}
        };

        private static readonly Dictionary<int, Dictionary<string, int>> m_controls = new Dictionary<int, Dictionary<string, int>>();

        private static int m_controlsID;
        public static int ControlsID
        {
            get
            {
                return m_controlsID;
            }
            set
            {
                if (!m_controls.ContainsKey(value))
                    m_controls[value] = new Dictionary<string, int>();
                m_controlsID = value;
            }
        }

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

        private static int NrOfControls
        {
            get
            {
                int max = m_controls.Values.Select(controlIndexes => controlIndexes.Values.Max()).Max();
                return max + 1; // +1 because values are indices to array -> they start at 0
            }
        }

        public static void CreateControlsFor(int controlID)
        {
            ControlsID = controlID;

            int offset = ControlsID * simpleControls.Count;

            m_controls[ControlsID] = simpleControls.ToDictionary(x => x.Key, x => x.Value + offset);
        }

        private static void UpdateMapping()
        {
            Dictionary<string, int> controlIndexes = new Dictionary<string, int>(simpleControls);
            m_controls.Clear();
            ControlsID = 0;
            m_controls[ControlsID] = controlIndexes;

            if (Mode == ControlMode.KeyboardMouse)
            {
                controlIndexes["forward"] = 87; // W
                controlIndexes["backward"] = 83; // S
                controlIndexes["left"] = 65; // A
                controlIndexes["right"] = 68; // D
                controlIndexes["rot_left"] = 69; // Q
                controlIndexes["rot_right"] = 81; // E

                controlIndexes["fof_up"] = 73; // I
                controlIndexes["fof_left"] = 76; // J
                controlIndexes["fof_down"] = 75; // K
                controlIndexes["fof_right"] = 74; // L

                controlIndexes["interact"] = 66; // B
                controlIndexes["use"] = 78; // N
                controlIndexes["pickup"] = 77; // M
            }
        }

        public static int Idx(string key)
        {
            return m_controls[ControlsID][key];
        }

        public static void CheckControlSize(MyValidator validator, MyAbstractMemoryBlock controls, MyWorkingNode sender)
        {
            validator.AssertError(controls != null, sender, "Controls are not connected");

            if (controls != null)
            {
                int neededControls = NrOfControls;
                int providedControls = controls.Count;
                validator.AssertError(providedControls >= neededControls, sender, String.Format("Wrong number of actions. With current control mode ({0}) you have to provide at least {1} controls. Provide the correct number of controls or change the control mode.", Mode, neededControls));
                validator.AssertWarning(providedControls != neededControls, sender, String.Format("With current control mode ({0}) you should provide {1} controls but you provided {2} controls. Make sure that this is what you want and you have correct control mode chosen.", Mode, neededControls, providedControls));
            }
        }
    }
}
