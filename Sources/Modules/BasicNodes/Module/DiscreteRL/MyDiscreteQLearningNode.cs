using GoodAI.Core.Memory;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using YAXLib;


namespace GoodAI.Modules.Harm
{
    /// <author>GoodAI</author>
    /// <meta>jv</meta>
    /// <status>Working</status>
    /// <summary>
    /// Implements Q-Learning (Q-Lambda) algorithm based on matrix representation of data.
    /// </summary>
    /// <description>
    /// Node does the following:
    ///    <ul>
    ///    <li> <b>Reveives data:</b> Accepts world state description and currently selected action (code 1ofN).</li>
    ///    <li> <b>Learns:</b> Updates own Q(s,a) matix.</li>
    ///    <li> <b>Publishes what it has learned: </b>Publishes utilities of actions in the current state, these are scaled by the value of motivation.</li>
    ///   </ul>
    ///
    /// Note that input data have to be rescaled to integer values.  
    /// No. of matrix dimensions and dimension sizes adapts itself to the size of data seen so far 
    /// (that is no. of variables and no. of values of each variable).
    /// 
    /// <h3>Inputs</h3>
    /// <ul>
    ///     <li> <b>GlobalData:</b> Vector describing state of the environment (can contain variables and constants). Values are scaled by the InputRescaleSize parameter.</li>
    ///     <li> <b>SelectedAction:</b> Vector of size [numberOfActions]. The highest value indicates action that has been executed by the agent.</li>
    ///     <li> <b>Reward:</b> If the value is non-zero, the RL receives the reward.</li>
    ///     <li> <b>Motivation:</b> Scales action utilities: the higher the motivation, higher values on the output.</li>
    /// </ul>
    /// <h3>Output</h3>
    /// <ul>
    ///     <li> <b>Utilities:</b> Vector of action utilities in a given state (the higher the value, the better to use the action).</li>
    /// </ul>
    /// <h3>Memory Blocks</h3>
    /// <ul>
    ///     <li> <b>RewardStats:</b> Two values that indicate {Total Reward/Step, Total Reward}, where Total Reward is sum of all rewards received during the simulation.</li>
    /// </ul>
    /// <h3>Parameters</h3>
    /// <ul>
    ///     <li> <b>InputRescaleSize:</b> Memory is indexed by integers, user should ideally rescale variable values to fit into integers.</li>
    ///     <li> <b>Number of Primitive Actions:</b> Number of primitive actions produced by the agent (e.g. 6 for the current gridworld, 9 for the tictactoe game).</li>
    ///     <li> <b>SumRewards:</b> Sum across the values in the vector of rewards?</li>
    /// </ul>
    /// 
    /// </description>
    public class MyDiscreteQLearningNode : MyAbstractDiscreteQLearningNode
    {
        [MyInputBlock(2)]
        public MyMemoryBlock<float> RewardInput
        {
            get { return GetInput(2); }
        }

        [MyInputBlock(3)]
        public MyMemoryBlock<float> MotivationInput
        {
            get { return GetInput(3); }
        }

        [MyBrowsable, Category("IO"), DisplayName("Sum Rewards"),
        Description("Sum across the values in the vector of rewards?")]
        [YAXSerializableField(DefaultValue = false)]
        public bool SumRewards { get; set; }

        // First element is rewards/step, the second is total no of rewards received
        public MyMemoryBlock<float> RewardStats { get; private set; }

        public MyQSAMemory Memory { get; private set; }
        public MyDiscreteQLearning LearningAlgorithm { get; private set; }
        public MyModuleParams LearningParams { get; private set; }

        public override void UpdateMemoryBlocks()
        {
            UtilityOutput.Count = NoActions;
            RewardStats.Count = 2;

            LearningParams = new MyModuleParams();
            Memory = new MyQSAMemory(GlobalDataInput.Count, NoActions);
            LearningAlgorithm = new MyDiscreteQLearning(LearningParams, Memory);

            if (GlobalDataInput != null)
            {
                if (NoActions == 6)
                {
                    MyLog.DEBUG.WriteLine("6 actions set by the user, will use action names for gridworld");
                    Rds = new MyRootDecisionSpace(GlobalDataInput.Count, new String[] { " -", " <", " >", " ^", " v", " P" }, LearningParams);
                }
                else if (NoActions == 3)
                {
                    MyLog.DEBUG.WriteLine("3 actions set by the user, will use action names for pong");
                    Rds = new MyRootDecisionSpace(GlobalDataInput.Count, new String[] { " <", " -", " >" }, LearningParams);
                }
                else
                {
                    MyLog.DEBUG.WriteLine("Unknown no. of actions, will use automatic naming of actions");
                    String[] names = new String[NoActions];
                    for (int i = 0; i < NoActions; i++)
                    {
                        names[i] = "A" + i;
                    }
                    Rds = new MyRootDecisionSpace(GlobalDataInput.Count, names, LearningParams);
                }
                CurrentStateOutput.Count = GlobalDataInput.Count;
            }
        }


        /// <summary>
        /// Method creates 2D array of max action utilities and max action labels over across selected dimensions.
        /// The values in the memory are automatically scaled into the interval 0,1. Realtime values are multiplied by motivations.
        /// </summary>
        /// <param name="values">array passed by reference for storing utilities of best action</param>
        /// <param name="labelIndexes">array of the same size for best action indexes</param>
        /// <param name="XVarIndex">global index of state variable in the VariableManager</param>
        /// <param name="YVarIndex">the same: y axis</param>
        /// <param name="showRealtimeUtilities">show current utilities (scaled by the current motivation)</param>
        /// <param name="policyNumber">not used here, this RL learns only one policy</param>
        public override void ReadTwoDimensions(ref float[,] values, ref int[,] labelIndexes,
            int XVarIndex, int YVarIndex, bool showRealtimeUtilities = false, int policyNumber = 0)
        {
            Vis.ReadTwoDimensions(ref values, ref labelIndexes, XVarIndex, YVarIndex, showRealtimeUtilities);
        }

        public int DecodeExecutedAction()
        {
            SelectedActionInput.SafeCopyToHost();
            int selected = 0;
            for (int i = 0; i < SelectedActionInput.Count; i++)
            {
                if (SelectedActionInput.Host[selected] < SelectedActionInput.Host[i])
                {
                    selected = i;
                }
            }
            return selected;
        }

        public float GetCurrentReward()
        {
            RewardInput.SafeCopyToHost();

            if (this.SumRewards)
            {
                float reward = 0;
                for (int i = 0; i < RewardInput.Count; i++)
                {
                    reward += RewardInput.Host[i];
                }
                return reward;
            }
            else
            {
                return RewardInput.Host[0];
            }
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);
            if (!SumRewards && RewardInput != null && RewardInput.Count != 1)
            {
                validator.AddWarning(this, "Only one reward should be on the input");
            }
            if (MotivationInput != null && MotivationInput.Count != 1)
            {
                validator.AddWarning(this, "Dimension of MotivationInput should be 1");
            }
        }

        public MyVariableUpdateTask StateSpaceUpdate { get; private set; }
        public MyLearnTask Learn { get; private set; }
        public MyReadUtilsTask ReadUtils { get; private set; }
        public MyActionUtilsVisualization Vis { get; protected set; }

        /// <summary>
        ///  <ul>
        ///    <li>  Checks which variables should be contained in the decision space.</li>
        ///    <li>  For each dimension of GlobalDataInput checks if there were multiple values, 
        /// if yes, it is identified as a variable and added into the DS. </li>
        ///    <li>  Also, inputs are rescaled by the value of RescaleVariables.</li>
        ///   </ul>
        /// </summary>
        [Description("Monitor changes of inputs"), MyTaskInfo(OneShot = false)]
        public class MyVariableUpdateTask : MyTask<MyDiscreteQLearningNode>
        {

            private bool warnedNegative = false;
            public override void Init(int nGPU)
            {
            }

            public override void Execute()
            {
                float[] inputs = this.RescaleAllInputs();

                Owner.Rds.VarManager.MonitorAllInputValues(inputs, null);

                int[] s_tt = Owner.Rds.VarManager.GetCurrentState();
                if (s_tt.Count() != Owner.CurrentStateOutput.Count)
                {
                    MyLog.WARNING.WriteLine("Unexpected size of current state");
                }
                else
                {
                    for (int i = 0; i < s_tt.Count(); i++)
                    {
                        Owner.CurrentStateOutput.Host[i] = s_tt[i];
                    }
                    Owner.CurrentStateOutput.SafeCopyToDevice();
                }
            }

            private float[] RescaleAllInputs()
            {
                Owner.GlobalDataInput.SafeCopyToHost();
                float[] inputs = new float[Owner.GlobalDataInput.Count];

                for (int i = 0; i < Owner.GlobalDataInput.Count; i++)
                {
                    Owner.GlobalDataInput.Host[i] *= Owner.RescaleVariables;
                    inputs[i] = Owner.GlobalDataInput.Host[i];
                    if (inputs[i] < 0)
                    {
                        if (!warnedNegative)
                        {
                            MyLog.DEBUG.WriteLine("WARNING: negative value on input detected, all negative values will be set to 0");
                            warnedNegative = true;
                        }
                        inputs[i] = 0;
                    }
                }
                return inputs;
            }
        }

        /// <summary>
        /// Updates the values of the Q(s,a) matrix by means of Q-Learning with eligibility trace.
        /// 
        /// <h3>Parameters</h3>
        /// <ul>
        ///     <li><b>Alpha:</b> Learning factor - the higher the faster the learning is.</li>
        ///     <li><b>Gamma:</b> How far into the future algorithm looks for learning.</li>
        ///     <li><b>RewardScale:</b> Increases the value of reward stored (helps increasing accuracy).</li>
        ///     <li><b>Lambda:</b> How strongly are past values updated (bigger values increase the learning speed, but can cause learning oscilations).</li>
        ///     <li><b>EligibilityTraceEnabled:</b> Use the thrace?</li>
        ///     <li><b>EligibilityTraceLen:</b> How many past steps to update at once (determined efectively by the lambda parameter).</li>
        /// </ul>
        /// 
        /// </summary>
        [Description("Q-Learning"), MyTaskInfo(OneShot = false)]
        public class MyLearnTask : MyTask<MyDiscreteQLearningNode>
        {
            [MyBrowsable, Category("Learning"), Description("Learning factor - the higher the faster the learning is")]
            [YAXSerializableField(DefaultValue = 0.5f)]
            public float Alpha { get; set; }

            [MyBrowsable, Category("Learning"), Description("How far into the future algorithm looks for learning")]
            [YAXSerializableField(DefaultValue = 0.7f)]
            public float Gamma { get; set; }

            [MyBrowsable, Category("Learning"), Description("Increases the value of reward (helps increasing accuracy)")]
            [YAXSerializableField(DefaultValue = 10000000)]
            public int RewardScale { get; set; }

            [MyBrowsable, Category("Eligibility Trace"), Description("How strongly are past values updated (bigger values increase the learning speed, but can cause learning oscilations)")]
            [YAXSerializableField(DefaultValue = 0.60f)]
            public float Lambda { get; set; }

            [MyBrowsable, Category("Eligibility Trace"), Description("Use the thrace?")]
            [YAXSerializableField(DefaultValue = true)]
            public bool EligibilityTraceEnabled { get; set; }

            [MyBrowsable, Category("Eligibility Trace"), Description("How many past steps to update at once (determined efectively by the lambda parameter)")]
            [YAXSerializableField(DefaultValue = 40)]
            public int EligibilityTraceLen { get; set; }


            public override void Init(int nGPU)
            {
            }

            public override void Execute()
            {
                UpdateParams();
                Owner.RewardInput.SafeCopyToHost();

                int a_t = Owner.DecodeExecutedAction();                     // previously executed action
                int[] s_tt = Owner.Rds.VarManager.GetCurrentState();        // current state s_t'                
                float r_t = Owner.GetCurrentReward() * RewardScale;         // current reward

                Owner.LearningAlgorithm.Learn(r_t, s_tt, a_t);  // learn, update the eligibility inside
            }

            private void UpdateParams()
            {
                Owner.LearningParams.Alpha = Alpha;
                Owner.LearningParams.Gamma = Gamma;
                Owner.LearningParams.Lambda = Lambda;
                Owner.LearningParams.TraceLength = EligibilityTraceLen;
                Owner.LearningParams.UseTrace = EligibilityTraceEnabled;
                Owner.LearningParams.RewardScale = RewardScale;
            }
        }

        /// <summary>
        /// Reads utility values in the current state, rescales them by the MotivationInput and publishes to the UtilityOutput.
        /// </summary>
        [Description("Pubish utility values scaled by the value of motivation"), MyTaskInfo(OneShot = false)]
        public class MyReadUtilsTask : MyTask<MyDiscreteQLearningNode>
        {
            public override void Init(int nGPU)
            {
            }

            public override void Execute()
            {

                int[] states = Owner.Rds.VarManager.GetCurrentState();

                float[] utils = Owner.LearningAlgorithm.ReadUtils(states);

                if (utils.Length != Owner.UtilityOutput.Count)
                {
                    MyLog.ERROR.WriteLine("Current utility values have different dims. that no. of actions");
                }
                else
                {
                    Owner.UtilityOutput.SafeCopyToHost();
                    for (int i = 0; i < utils.Length; i++)
                    {
                        Owner.UtilityOutput.Host[i] = utils[i] * Owner.MotivationInput.Host[0];
                    }
                    Owner.UtilityOutput.SafeCopyToDevice();
                }
            }
        }

        /// <summary>
        /// Updates the <b>RewardStats</b> values = {Total Reward/Step, Total Reward}.
        /// </summary>
        [Description("Visualize learned data"), MyTaskInfo(OneShot = false)]
        public class MyActionUtilsVisualization : MyTask<MyDiscreteQLearningNode>
        {
            float[,] values;
            int[,] labelIndexes;

            public override void Init(int nGPU)
            {
                values = new float[2, 2];       // dummy arrays for testing, TODO delete
                labelIndexes = new int[,] { };

                Owner.RewardStats.SafeCopyToHost();
                Owner.RewardStats.Host[0] = 0;
                Owner.RewardStats.Host[1] = 0;
                Owner.RewardStats.SafeCopyToDevice();
            }

            public override void Execute()
            {
                Owner.RewardStats.SafeCopyToHost();
                Owner.RewardStats.Host[1] += Owner.GetCurrentReward();
                if (this.SimulationStep > 0)
                {
                    Owner.RewardStats.Host[0] = Owner.RewardStats.Host[1] / (float)this.SimulationStep;
                }
                Owner.RewardStats.SafeCopyToDevice();
                Owner.MotivationInput.SafeCopyToHost();
            }

            /// <summary>
            /// Method creates 2D array of max action utilities and max action labels over across selected dimensions.
            /// The values in the memory are automatically scaled into the interval 0,1. Realtime values are multiplied by motivations.
            /// </summary>
            /// <param name="values">array passed by reference for storing utilities of best action</param>
            /// <param name="labelIndexes">array of the same size for best action indexes</param>
            /// <param name="XVarIndex">global index of state variable in the VariableManager</param>
            /// <param name="YVarIndex">the same: y axis</param>
            /// <param name="showRealtimeUtilities">show current utilities (scaled by the current motivation)</param>
            public void ReadTwoDimensions(ref float[,] values, ref int[,] labelIndexes,
                int XVarIndex, int YVarIndex, bool showRealtimeUtilities)
            {
                if (XVarIndex >= Owner.Rds.VarManager.MAX_VARIABLES)
                {
                    XVarIndex = Owner.Rds.VarManager.MAX_VARIABLES - 1;
                }
                if (YVarIndex >= Owner.Rds.VarManager.MAX_VARIABLES)
                {
                    YVarIndex = Owner.Rds.VarManager.MAX_VARIABLES - 1;
                }
                if (YVarIndex < 0)
                {
                    YVarIndex = 0;
                }
                if (XVarIndex < 0)
                {
                    XVarIndex = 0;
                }
                MyQSAMemory mem = Owner.Memory;
                int[] sizes = mem.GetStateSizes();              // size of the matrix
                int[] indexes = Owner.Rds.VarManager.GetCurrentState();

                int[] actionGlobalIndexes = mem.GetGlobalActionIndexes();

                MyVariable varX = Owner.Rds.VarManager.GetVarNo(XVarIndex);
                MyVariable varY = Owner.Rds.VarManager.GetVarNo(YVarIndex);

                float[] varXvals = varX.Values.ToArray();
                float[] varYvals = varY.Values.ToArray();

                Array.Sort(varXvals);
                Array.Sort(varYvals);

                int sx = 0;
                int sy = 0;

                sx = varX.Values.Count;
                sy = varY.Values.Count;

                if (values == null || labelIndexes == null ||
                    values.GetLength(0) != sx || values.GetLength(1) != sy ||
                    labelIndexes.GetLength(0) != sx || labelIndexes.GetLength(1) != sy)
                {
                    values = new float[sx, sy];
                    labelIndexes = new int[sx, sy];
                }

                for (int i = 0; i < sx; i++)
                {
                    indexes[XVarIndex] = (int)varXvals[i];

                    for (int j = 0; j < sy; j++)
                    {
                        indexes[YVarIndex] = (int)varYvals[j];

                        float[] utilities = mem.ReadData(indexes);
                        float memoryMaxValue = Owner.LearningAlgorithm.GetMaxVal();

                        if (memoryMaxValue != 0)
                        {
                            for (int k = 0; k < utilities.Length; k++)
                            {
                                utilities[k] = utilities[k] / memoryMaxValue;
                            }
                        }

                        float maxValue = 0.0f;
                        int maxIndex = 0;

                        if (utilities.Length != actionGlobalIndexes.Length)
                        {
                            MyLog.DEBUG.WriteLine("ERROR: unexpected length of utilities array, will place default values");
                            utilities = new float[actionGlobalIndexes.Length];
                        }
                        else if (actionGlobalIndexes.Length == 0)
                        {
                            MyLog.DEBUG.WriteLine("WARNING: this DS contains no actions. Will use the action 0");
                            utilities = new float[1];
                            actionGlobalIndexes = new int[] { 0 };
                        }
                        else
                        {
                            maxValue = utilities.Max();
                            maxIndex = utilities.ToList().IndexOf(maxValue);
                        }
                        if (showRealtimeUtilities)
                        {
                            Owner.MotivationInput.SafeCopyToHost();
                            float motivation = Owner.MotivationInput.Host[0];

                            values[i, j] = maxValue * motivation;
                        }
                        else
                        {
                            values[i, j] = maxValue;

                        }
                        labelIndexes[i, j] = actionGlobalIndexes[maxIndex];
                    }
                }
            }
        }
    }
}
