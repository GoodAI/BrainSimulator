using ManagedCuda.VectorTypes;
using System;
using System.Collections;
using System.Collections.Generic;

namespace GoodAI.Modules.GridWorld
{

    /// <summary>
    /// Defines the behavior of all objects in the world.
    /// </summary>
    public interface IWorldEngine
    {
        // apply the agent's action in the simulator, update all data
        bool ResolveAction(AGENT_ACTIONS action);

        // returns an array of output values representing the state of the environment
        float[] GetGlobalOutputData();
        int GetGlobalOutputDataSize();

        // an array of output values that are variables in the environment
        float[] GetVariablesOutputData();
        int GetVariablesOutputDataSize();

        String[] GetGlobalOutputDataNames();
        String[] GetGlobalOutputVarNames();

        MyWorldEngineParams GetParams();
    }

    public class MyWorldEngineParams
    {
        public bool ForceDoorSwitches;
        public bool ForceDoorSwitchesState;
        public bool ForceLightSwitches;
        public bool ForceLightSwitchesState;
        public int ViewLimit;
        public bool LimitFieldOfView;
    }

    public enum AGENT_ACTIONS
    {
        NOOP = 0,
        LEFT = 1,
        RIGHT = 2,
        UP = 3,
        DOWN = 4,
        BASIC = 5   // basic (universal) action
    }

    public class SimpleGridWorldEngine : IWorldEngine
    {
        public static float NOT_VISIBLE = float.MinValue;

        private IWorldParser w;
        protected MyWorldEngineParams pars;

        public SimpleGridWorldEngine(IWorldParser w)
        {
            this.pars = new MyWorldEngineParams();
            this.w = w;
        }

        /// <summary>
        /// Prints names of output data.
        /// </summary>
        /// <param name="useConstants">Print names of global Data, or only variables?</param>
        /// <returns>String containing all names separated by spaces</returns>
        private String PrintNames(bool useConstants)
        {
            String names = "";
            String[] ns;
            if (useConstants)
            {
                ns = this.GetGlobalOutputDataNames();
            }
            else
            {
                ns = this.GetGlobalOutputVarNames();
            }
            for (int i = 0; i < ns.Length; i++)
            {
                names += " " + ns[i];
            }
            return names;
        }

        public MyWorldEngineParams GetParams()
        {
            return this.pars;
        }

        public bool ResolveAction(AGENT_ACTIONS action)
        {
            MyMovingObject agent = w.GetAgent();
            Tale[,] tales = w.GetTales();

            bool navigation = this.ResolveNavigation(agent, tales, action);
            navigation |= this.ResolveCustomActions(agent, tales, action);

            return navigation;
        }

        public static bool IsNavigationAction(AGENT_ACTIONS action)
        {
            return action == AGENT_ACTIONS.LEFT || action == AGENT_ACTIONS.RIGHT || action == AGENT_ACTIONS.UP || action == AGENT_ACTIONS.DOWN;
        }

        protected bool ResolveNavigation(
            MyMovingObject agent,
            Tale[,] tales,
            AGENT_ACTIONS action)
        {
            int2 pos = agent.GetPosition();

            if (action == AGENT_ACTIONS.UP && pos.y < w.GetHeight() - 1)
            {
                if (CanMakeStepTo(pos.x, pos.y + 1, tales, agent.GetWeight()))
                {
                    agent.MoveUp();
                    return true;
                }
            }
            else if (action == AGENT_ACTIONS.DOWN && agent.GetPosition().y > 0)
            {
                if (CanMakeStepTo(pos.x, pos.y - 1, tales, agent.GetWeight()))
                {
                    agent.MoveDown();
                    return true;
                }
            }
            else if (action == AGENT_ACTIONS.LEFT && agent.GetPosition().x > 0)
            {
                if (CanMakeStepTo(pos.x - 1, pos.y, tales, agent.GetWeight()))
                {
                    agent.MoveLeft();
                    return true;
                }
            }
            else if (action == AGENT_ACTIONS.RIGHT && agent.GetPosition().x < w.GetWidth() - 1)
            {
                if (CanMakeStepTo(pos.x + 1, pos.y, tales, agent.GetWeight()))
                {
                    agent.MoveRight();
                    return true;
                }
            }
            return false;
        }

        protected bool ResolveCustomActions(MyMovingObject agent, Tale[,] tales, AGENT_ACTIONS action)
        {
            bool result = false;
            Tale t;
            if (action == AGENT_ACTIONS.BASIC)
            {
                t = tales[agent.GetPosition().x, agent.GetPosition().y];

                for (int i = 0; i < t.Objects.Count; i++)
                {
                    if (t.Objects[i] is TwoStateObjectControl)
                    {
                        if (t.Objects[i] is DoorControl)
                        {
                            if (!pars.ForceDoorSwitches)
                            {
                                ((DoorControl)t.Objects[i]).applyPressAction();
                                result = true;
                            }
                        }
                        else if (t.Objects[i] is LightsControl)
                        {
                            if (!pars.ForceLightSwitches)
                            {
                                ((LightsControl)t.Objects[i]).applyPressAction();
                                result = true;
                            }
                        }
                        else
                        {
                            ((TwoStateObjectControl)t.Objects[i]).applyPressAction();
                            result = true;
                        }
                    }
                }
            }

            if (pars.ForceDoorSwitches || pars.ForceLightSwitches)
            {
                for (int i = 0; i < tales.GetLength(0); i++)
                {
                    for (int j = 0; j < tales.GetLength(1); j++)
                    {
                        t = tales[i, j];
                        if (t.Objects != null)
                        {
                            for (int k = 0; k < t.Objects.Count; k++)
                            {
                                if (t.Objects[k] is DoorControl && pars.ForceDoorSwitches)
                                {
                                    DoorControl dc = (DoorControl)t.Objects[k];
                                    if (dc.IsOn() != pars.ForceDoorSwitchesState)
                                    {
                                        dc.applyPressAction();
                                    }
                                }
                                else if (t.Objects[k] is LightsControl && pars.ForceLightSwitches)
                                {
                                    LightsControl lc = (LightsControl)t.Objects[k];
                                    if (lc.IsOn() != pars.ForceLightSwitchesState)
                                    {
                                        lc.applyPressAction();
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return result;
        }

        // resolve colisions
        private bool CanMakeStepTo(int toX, int toY, Tale[,] tales, float myWeight)
        {
            if (tales[toX, toY].IsObstacle)
            {
                return false;
            }
            // empty tale with no objects
            if (tales[toX, toY].Objects.Count == 0)
                return true;
            // some of objects is heavier than the agent
            for (int i = 0; i < tales[toX, toY].Objects.Count; i++)
            {
                if (tales[toX, toY].Objects[i].GetWeight() >= myWeight)
                    return false;
            }
            return true;
        }

        public float[] GetVariablesOutputData()
        {
            ArrayList o = new ArrayList();
            o.Add((float)w.GetAgent().GetPosition().x / (w.GetWidth() - 1));
            o.Add((float)w.GetAgent().GetPosition().y / (w.GetHeight() - 1));

            int apx = w.GetAgent().GetPosition().x;
            int apy = w.GetAgent().GetPosition().y;

            MyStaticObject[] so = w.GetStaticObjects();
            for (int i = 0; i < so.Length; i++)
            {
                foreach (KeyValuePair<String, float> item in so[i].properties)
                {
                    if (pars.LimitFieldOfView && !this.IsVisible(apx, apy, so[i]))
                    {
                        o.Add(NOT_VISIBLE);
                    }
                    else
                    {
                        o.Add((float)item.Value);
                    }
                }
            }
            float[] ff = o.ToArray(typeof(float)) as float[];
            return ff;
        }

        private bool IsVisible(int apx, int apy, MyStaticObject so)
        {
            int dx = Math.Abs(so.GetPosition().x - apx);
            int dy = Math.Abs(so.GetPosition().y - apy);

            return dx <= pars.ViewLimit && dy <= pars.ViewLimit;
        }

        public int GetVariablesOutputDataSize()
        {
            int poc = 2;
            MyStaticObject[] so = w.GetStaticObjects();
            for (int i = 0; i < so.Length; i++)
            {
                foreach (KeyValuePair<String, float> item in so[i].properties)
                {
                    poc++;
                }
            }
            return poc;
        }

        public float[] GetGlobalOutputData()
        {
            ArrayList o = new ArrayList();
            o.Add((float)w.GetAgent().GetPosition().x / (w.GetWidth() - 1));
            o.Add((float)w.GetAgent().GetPosition().y / (w.GetHeight() - 1));

            int apx = w.GetAgent().GetPosition().x;
            int apy = w.GetAgent().GetPosition().y;

            MyStaticObject[] so = w.GetStaticObjects();
            for (int i = 0; i < so.Length; i++)
            {
                foreach (KeyValuePair<String, float> item in so[i].properties)
                {
                    if (pars.LimitFieldOfView && !this.IsVisible(apx, apy, so[i]))
                    {
                        o.Add(NOT_VISIBLE);
                    }
                    else
                    {
                        o.Add((float)item.Value);
                    }
                }
                o.Add((float)so[i].GetPosition().x / (w.GetWidth() - 1));
                o.Add((float)so[i].GetPosition().y / (w.GetHeight() - 1));
            }
            float[] ff = o.ToArray(typeof(float)) as float[];
            return ff;
        }


        /// <summary>
        /// Return vector of variable names (corresponds to the GlobalVariable data output)
        /// </summary>
        /// <returns></returns>
        public String[] GetGlobalOutputVarNames()
        {
            List<String> names = new List<String>();
            names.Add("X");
            names.Add("Y");
            MyStaticObject[] so = w.GetStaticObjects();

            int door = 0;
            int lights = 0;
            int doorControl = 0;
            int lightsControl = 0;

            for (int i = 0; i < so.Length; i++)
            {
                foreach (KeyValuePair<String, float> item in so[i].properties)
                {
                    if (so[i] is MyDoor)
                    {
                        names.Add("D" + door);
                        door++;
                    }
                    if (so[i] is Lights)
                    {
                        names.Add("L" + lights);
                        lights++;
                    }
                    if (so[i] is DoorControl)
                    {
                        names.Add("DC" + doorControl);
                        doorControl++;
                    }
                    if (so[i] is LightsControl)
                    {
                        names.Add("LC");
                        lightsControl++;
                    }
                }
            }
            return names.ToArray() as String[];
        }

        /// <summary>
        /// Return vector of all data names (corresponds to the GlobalData data output), includes also all constants
        /// </summary>
        /// <returns></returns>
        public String[] GetGlobalOutputDataNames()
        {
            List<String> names = new List<String>();
            names.Add("X");
            names.Add("Y");
            MyStaticObject[] so = w.GetStaticObjects();

            int door = 0;
            int lights = 0;
            int lightsControl = 0;
            int doorControl = 0;

            for (int i = 0; i < so.Length; i++)
            {
                // assumed that each thing has only one property
                foreach (KeyValuePair<String, float> item in so[i].properties)
                {
                    //MyLog.DEBUG.WriteLine("object type is: " + so[i].GetType().Name);
                    if (so[i] is MyDoor)
                    {
                        names.Add("D" + door);
                        names.Add("D" + door + "X");
                        names.Add("D" + door + "Y");
                        door++;
                    }
                    else if (so[i] is Lights)
                    {
                        names.Add("L" + lights);
                        names.Add("L" + lights + "X");
                        names.Add("L" + lights + "Y");
                        lights++;
                    }

                }
                if (so[i] is LightsControl)
                {
                    names.Add("LC" + lightsControl);
                    names.Add("LC" + lightsControl + "X");
                    names.Add("LC" + lightsControl + "Y");
                    lightsControl++;
                }
                else if (so[i] is DoorControl)
                {
                    names.Add("DC" + doorControl);
                    names.Add("DC" + doorControl + "X");
                    names.Add("DC" + doorControl + "Y");
                    doorControl++;
                }
            }
            return names.ToArray() as String[];

        }
        public int GetGlobalOutputDataSize()
        {
            int poc = 2;
            MyStaticObject[] so = w.GetStaticObjects();
            for (int i = 0; i < so.Length; i++)
            {
                foreach (KeyValuePair<String, float> item in so[i].properties)
                {
                    poc++;
                }
                poc += 2; //also add the xy position
            }
            return poc;
        }
    }
}

