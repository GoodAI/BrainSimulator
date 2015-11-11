using GoodAI.Core.Memory;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.GridWorld;
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
    /// Implements HARM based on the discrete Q-Learning algorithm.
    /// </summary>
    /// <description>
    /// HARM - Hierarchy, Abstraction, Rinforcements, Motivations. 
    /// A system, that is able to autonomously identify own capabilities and tries to learn them (in a MDP environments).
    /// The system:
    /// 
    ///  <ul> 
    ///    <li> Senses environment (vector of data: variables and constants) </li>
    ///    <li> Acts in the environment (produces actions)</li>
    ///    <li> Observes consequences of own actions. If some variable changes, the agent asumes that it is a consequence of its action and tries to learn this new ability:</li>
    ///     <ul>
    ///     <li> It creates new Stochastic Return Predictor (SRP) = decision space + discrete RL algorithm + source of motivation to execute this behavior</li>
    ///     <li> This new SRP has a goal to learn how to change this variable. It is done by making connection link from variable change to reward of this SRP</li>
    ///     <li> The decision space of this SRP should contain only subset of environment variables</li>
    ///     <li> From now on, the SRP tries to learn how to change the variable by observing actions taken by the agent and receiving own reward</li>
    ///     <li> All SRPs vote by publishing utilities of child actions in a given state.</li>
    ///     <li> Utilities produced by the SRP are scaled by the amount of motivation. Motivation increases in time and is set to 0 when the reward is received.</li>
    ///    </ul>
    ///  </ul> 
    /// 
    /// The node publishes utilities of primitive actions in a given state. 
    /// The utility of action is computed as a sum of utilities (scaled by motivations) from all parent SRPs. 
    /// Therefore all SRPs learn in parallel and resulting strategy followed by the agent is a result of all intentions of the agent.
    /// <br>
    /// Current version uses one level of hierarchy of RL decision spaces based on interaction 
    /// with the environment.
    /// </br>
    /// <h3>Before use:</h3>
    ///  <ul> 
    ///     <li> Works well in environments which fulfill the MDP (Markov Decision Process) environments.</li>
    ///     <li> Works only for positive integer values of variables. Need to rescale input values by the RescaleVariables parameter. </li>
    ///     <li> Currently uses one level of hierarchy of SRPs.</li>
    ///     <li> Variable and action subspacing should be used carefully with respect to the environment, or disabled for slower, but safer learning.</li>
    ///  </ul> 
    /// </description>
    public class MyDiscreteHarmNode : MyAbstractDiscreteQLearningNode
    {
        [MyInputBlock(2)]
        public MyMemoryBlock<float> MotivationsInput
        {
            get { return GetInput(2); }
        }

        [MyInputBlock(3)]
        public MyMemoryBlock<float> MotivationsOverrideInput
        {
            get { return GetInput(3); }
        }

        [MyBrowsable, Category("IO"), DisplayName("Connected to global data"),
        Description("Used only for parsing variable names. If the HARM is connected to " +
        "Variables output of the gridworld, set to false.")]
        [YAXSerializableField(DefaultValue = false)]
        public bool ConnectedToGlobal { get; set; }

        [MyBrowsable, Category("SRP Creation"), DisplayName("Maximum of actions"),
        Description("Total max. number of actions (primitive + abstract ones)")]
        [YAXSerializableField(DefaultValue = 12)]
        public int MaxActions { get; set; }

        public MyMemoryBlock<float> SubspaceUtils { get; private set; }

        public MyModuleParams LearningParams { get; private set; }
        public MyHierarchyMaintainer Hierarchy { get; private set; }

        public override void UpdateMemoryBlocks()
        {
            UtilityOutput.Count = NoActions;
            LearningParams = new MyModuleParams();

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
                Hierarchy = new MyHierarchyMaintainer(Rds, LearningParams);
                SubspaceUtils.Count = MaxActions * MaxActions;
                SubspaceUtils.ColumnHint = MaxActions;
            }
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);

            if (MotivationsOverrideInput != null && MotivationsOverrideInput.Count != 1)
            {
                validator.AddWarning(this, "Dimension of MotivationsOverrideInput should be 1");
            }
            if (MotivationsInput != null && MotivationsInput.Count < NoActions)
            {
                validator.AddWarning(this, "Motivations input should have AT LEAST " +
                    "the same no. of dimensions as no. of primitive actions.");
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
        /// <param name="policyNumber">optinal parameter. In case that the agent has more strategies, you can choose which one to read from.</param>
        public override void ReadTwoDimensions(ref float[,] values, ref int[,] labelIndexes,
            int XVarIndex, int YVarIndex, bool showRealtimeUtilities = false, int policyNumber = 0)
        {
            MyStochasticReturnPredictor predictor = Vis.GetPredictorNo(policyNumber);

            Vis.ReadTwoDimensions(ref values, ref labelIndexes, predictor, XVarIndex, YVarIndex, showRealtimeUtilities);
        }



        public MyPostSimulationStepTask PostSimStep { get; private set; }
        public MyVariableUpdateTask StateSpaceUpdate { get; private set; }
        public MyActionUpdateTask UpdateActions { get; private set; }
        public MyHierarchyCreationTask HierarchyCreation { get; private set; }
        public MyHierarchicalLearningTask LearnFromRewards { get; private set; }
        public MyMotivationOverrideTask MotivationOverride { get; private set; }
        public MyHierarchicalActionSelection ReadUtils { get; private set; }
        public MyActionUtilsVisualization Vis { get; protected set; }

        /// <summary>
        /// Monitors which variables should be contained in the decision space.
        /// 
        /// For each dimension of GlobalDataInput checks if there were multiple values, 
        /// if yes, it is identified as a variable and added into the DS.
        /// 
        /// Also, inputs are rescaled by the value of RescaleVariables.
        /// </summary>
        [Description("Monitor Variables"), MyTaskInfo(OneShot = false)]
        public class MyVariableUpdateTask : MyTask<MyDiscreteHarmNode>
        {
            private bool warnedNegative = false;

            public override void Init(int nGPU)
            {
                if (Owner.Owner.World is MyGridWorld)
                {
                    MyGridWorld w = (MyGridWorld)Owner.Owner.World;
                    String[] names = null;
                    if (Owner.ConnectedToGlobal)
                    {
                        names = w.Engine.GetGlobalOutputDataNames();
                    }
                    else
                    {
                        names = w.Engine.GetGlobalOutputVarNames();
                    }
                    for (int i = 0; i < names.Length; i++)
                    {
                        Owner.Rds.VarManager.GetVarNo(i).SetLabel(names[i]);
                    }
                }
                else
                {
                    MyLog.DEBUG.WriteLine("MyGridWorld world expected, will not parse variable names.");
                }
            }

            public override void Execute()
            {
                Owner.GlobalDataInput.SafeCopyToHost();
                float[] inputs = this.RescaleAllInputs();

                if (this.SimulationStep > 10)
                {
                    Owner.Rds.VarManager.MonitorAllInputValues(inputs, Owner.Hierarchy);
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
        /// Number of primitive actions is predefined, abastract acitons are created here.
        /// </summary>
        [Description("Monitor Actions"), MyTaskInfo(OneShot = false)]
        public class MyActionUpdateTask : MyTask<MyDiscreteHarmNode>
        {
            public override void Init(int nGPU)
            {
                // example how to manually create one new decision space with XY and all primitive actions
                // Owner.hierarchy.manualSubspacing(new int[] { 0, 1 }, new int[] { 0, 1, 2, 3, 4, 5 }, 2, "Lights-TEST", 1);
            }

            public override void Execute()
            {
                if (Owner.SelectedActionInput.Count != Owner.Hierarchy.GetNoPrimitiveActions())
                {
                    MyLog.ERROR.WriteLine("Unexpected no. of primitive actions on the SelectedAction input, ignoring!");
                    return;
                }

                int selected = this.decodeExecutedAction();
                Owner.Rds.ActionManager.SetJustExecuted(selected);
            }

            private int decodeExecutedAction()
            {
                Owner.SelectedActionInput.SafeCopyToHost();
                int selected = 0;
                for (int i = 0; i < Owner.SelectedActionInput.Count; i++)
                {
                    if (Owner.SelectedActionInput.Host[selected]
                        < Owner.SelectedActionInput.Host[i])
                    {
                        selected = i;
                    }
                }
                return selected;
            }
        }

        /// <summary>
        /// Implements Q-learning in multiple separate SRPs.
        ///
        /// <h3>Parameters</h3>
        /// <ul>
        ///     <li><b>Alpha: </b>Learning factor - the higher the faster the learning is.</li>
        ///     <li><b>Gamma: </b>How far into the future algorithm looks for learning.</li>
        ///     <li><b>RewardScale: </b>Increases the value of reward stored (helps increasing accuracy).</li>
        ///     <li><b>Lambda: </b>How strongly are past values updated (bigger values increase the learning speed, but can cause learning oscilations).</li>
        ///     <li><b>EligibilityTraceEnabled: </b>Use the trace?</li>
        ///     <li><b>EligibilityTraceLen: </b>How many past steps to update at once (determined efectively by the lambda parameter).</li>
        /// </ul>
        /// </summary>
        [Description("Learning"), MyTaskInfo(OneShot = false)]
        public class MyHierarchicalLearningTask : MyTask<MyDiscreteHarmNode>
        {
            [MyBrowsable, Category("Local Learning"), Description("Learning rate")]
            [YAXSerializableField(DefaultValue = 0.5f)]
            public float Alpha { get; set; }

            [MyBrowsable, Category("Local Learning"), Description("How far to look into ghe future <0,1>")]
            [YAXSerializableField(DefaultValue = 0.7f)]
            public float Gamma { get; set; }

            [MyBrowsable, Category("Local Learning"),
            Description("How strong the Eligibility trace will be (can cause unstable learning if too big)")]
            [YAXSerializableField(DefaultValue = 0.60f)]
            public float Lambda { get; set; }

            [MyBrowsable, Category("Local Learning"),
            Description("Enable/disable use of Eligibility traces in learning")]
            [YAXSerializableField(DefaultValue = true)]
            public bool EligibilityTraceEnabled { get; set; }

            [MyBrowsable, Category("Local Learning"),
            Description("Length of Eligibility trace (in steps)")]
            [YAXSerializableField(DefaultValue = 40)]
            public int EligibilityTraceLen { get; set; }

            [MyBrowsable, Category("Local Learning"),
            Description("Higher values of reward propagate better through the memory")]
            [YAXSerializableField(DefaultValue = 10000000)]
            public int RewardScale { get; set; }

            public override void Init(int nGPU)
            {
            }

            public override void Execute()
            {
                UpdateParams();
                Owner.Hierarchy.Learn();
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
        /// Implements subspacing (on a global level), variable/action removing/addding.
        /// 
        /// <h3>Parameters</h3>
        /// <ul>
        ///     <li><b>ActionTraceLength: </b>Length of trace meory for actinos.</li>
        ///     <li><b>VariableTraceLength: </b>Length of trace meory for variables.</li>
        ///     <li><b>ActionSubspacingThreshold: </b>The lower the threshold, the less actions will be in the Decision Space.</li>
        ///     <li><b>VariableSubspacingThreshold: </b>The lower the threshold, the less variables will be in the Decision Space.</li>
        ///     <li><b>HistoryForgettingRate: </b>How to weight the importance of actions/variables in the past.</li>
        ///     <li><b>SubspaceActions: </b>Subspace actions, or use all actions in new Decision Space.</li>
        ///     <li><b>SubspaceVariables: </b>Subspace variables? If disabled, all sensory data are considered in all decision spaces.</li>
        ///     <li><b>OnlineSubspaceVariables: </b>Subspace variables online? If some variable does not change often enough (relatively to receving reward) can be removed from the DS..</li>
        ///     <li><b>OnlineHistoryForgettingRate: </b>If variable changes 1 is added to its value. All values are decayed each step by the OnlineHistoryForgettingRate.</li>
        ///     <li><b>OnlineVariableRemovingThreshold: </b>After receiving the reward, the all variables in the DS are checked how often they changed in the history. If the value is under this threshold, variable will be removed from the DS.</li>
        /// </ul>
        /// </summary>
        [Description("SRP Creation"), MyTaskInfo(OneShot = false)]
        public class MyHierarchyCreationTask : MyTask<MyDiscreteHarmNode>
        {
            [MyBrowsable, Category("Subspacing"),
            Description("Length of the memory trace of actions")]
            [YAXSerializableField(DefaultValue = 100)]
            public int ActionTraceLength { get; set; }

            [MyBrowsable, Category("Subspacing"),
            Description("Length of trace meory for variables")]
            [YAXSerializableField(DefaultValue = 100)]
            public int VariableTraceLength { get; set; }

            [MyBrowsable, Category("Subspacing"),
            Description("The lower the threshold, the less actions will be in the Decision Space")]
            [YAXSerializableField(DefaultValue = 0.01f)]
            public float ActionSubspacingThreshold { get; set; }

            [MyBrowsable, Category("Subspacing"),
            Description("The lower the threshold, the less variables will be in the Decision Space")]
            [YAXSerializableField(DefaultValue = 0.01f)]
            public float VariableSubspacingThreshold { get; set; }

            [MyBrowsable, Category("Subspacing"),
            Description("How to weight the importance of actions/variables in the past")]
            [YAXSerializableField(DefaultValue = 0.01f)]
            public float HistoryForgettingRate { get; set; }

            [MyBrowsable, Category("Subspacing"),
            Description("Subspace actions, or use all actions in new Decision Space")]
            [YAXSerializableField(DefaultValue = false)]
            public bool SubspaceActions { get; set; }

            [MyBrowsable, Category("Subspacing"),
            Description("Subspace variables? If disabled, all sensory data are considered in all decision spaces.")]
            [YAXSerializableField(DefaultValue = false)]
            public bool SubspaceVariables { get; set; }

            [MyBrowsable, Category("Online Variable Removing"),
            Description("Subspace variables online? If some variable does not change often enough (relatively to receving reward) can be removed from the DS.")]
            [YAXSerializableField(DefaultValue = false)]
            public bool OnlineSubspaceVariables { get; set; }

            [MyBrowsable, Category("Online Variable Removing"),
            Description("If variable changes 1 is added to its value. All values are decayed each step by the OnlineHistoryForgettingRate.")]
            [YAXSerializableField(DefaultValue = 0.001f)]
            public float OnlineHistoryForgettingRate { get; set; }

            [MyBrowsable, Category("Online Variable Removing"),
            Description("After receiving the reward, the all variables in the DS are checked how often they changed in the history." +
            "If the value is under this threshold, variable will be removed from the DS")]
            [YAXSerializableField(DefaultValue = 0.001f)]
            public float OnlineVariableRemovingThreshold { get; set; }


            public override void Init(int nGPU)
            {
            }

            public override void Execute()
            {
                UpdateParams();
                Owner.Hierarchy.NewVariableAdding();  // add newly discovered variables into existing DSs
                Owner.Hierarchy.Subspacing();         // create new DSs
            }

            private void UpdateParams()
            {
                Owner.LearningParams.VarLength = VariableTraceLength;
                Owner.LearningParams.ActionLength = ActionTraceLength;
                Owner.LearningParams.ActionSubspacingThreshold = ActionSubspacingThreshold;
                Owner.LearningParams.VariableSubspacingThreshold = VariableSubspacingThreshold;
                Owner.LearningParams.HistoryForgettingRate = HistoryForgettingRate;

                Owner.LearningParams.SubspaceActions = SubspaceActions;
                Owner.LearningParams.SubspaceVariables = SubspaceVariables;

                Owner.LearningParams.BuildMultilevelHierarchy = false;
                Owner.LearningParams.EnableAbstractNavigation = true;

                Owner.LearningParams.OnlineSubspaceVariables = OnlineSubspaceVariables;
                Owner.LearningParams.OnlineHistoryForgettingRate = OnlineHistoryForgettingRate;
                Owner.LearningParams.OnlineVariableRemovingThreshold = OnlineVariableRemovingThreshold;
            }
        }

        /// <summary>
        /// Infers utility values of primitive actions from the RL hierarchy. Publishes utilities.
        /// 
        /// <h3>Parameters</h3>
        /// <ul>
        ///     <li><b>MotivationChange: </b>How much to increase the motivation each step if no reward is received.</li>
        ///     <li><b>UseHierarchicalASM: </b>Eeach SRP publishes utilities of all actions or should use own Action Selection Method (ASM) (select only one action)?</li>
        ///     <li><b>MinEpsilon: </b>If the UseHierarchicalASM is enabled, this is minimum probability of randomization in the ASM.</li>
        ///     <li><b>PropagateUtilitiesInHierarchy: </b>If disabled, utility does not propagete from SRPs to child actions.</li>
        /// </ul>
        /// </summary>
        [Description("Action Selection"), MyTaskInfo(OneShot = false)]
        public class MyHierarchicalActionSelection : MyTask<MyDiscreteHarmNode>
        {
            [MyBrowsable, Category("Action Selection"),
            Description("This value is added each step to the motivation in the motivation source")]
            [YAXSerializableField(DefaultValue = 0.01f)]
            public float MotivationChange { get; set; }

            [MyBrowsable, Category("Action Selection"),
            Description("Choose only one action in each SRP and propagate only value " +
                "of one selected action? (if disabled, all action utilities propagate to childs")]
            [YAXSerializableField(DefaultValue = false)]
            public bool UseHierarchicalASM { get; set; }

            [MyBrowsable, Category("Action Selection"),
            Description("Minimum randomization if UseHierarchicalASM is enabled.")]
            [YAXSerializableField(DefaultValue = 0.1f)]
            public float MinEpsilon { get; set; }

            [MyBrowsable, Category("Action Selection"),
            Description("Scale utilities by motivations when propagating down to childs of SRPs?")]
            [YAXSerializableField(DefaultValue = true)]
            public bool PropagateUtilitiesInHierarchy { get; set; }

            public override void Init(int nGPU)
            {
            }

            /// <summary>
            /// Infer utility values of primitive actions from the hierarchy and publish.
            /// </summary>
            public override void Execute()
            {
                this.UpdateParams();

                float[] outputs = Owner.Hierarchy.InferUtilities();

                if (outputs.Length != Owner.UtilityOutput.Count)
                {
                    MyLog.ERROR.WriteLine("List of inferred utilites is different than no. of primitive actions!");
                }
                else
                {
                    Owner.UtilityOutput.SafeCopyToHost();
                    for (int i = 0; i < outputs.Length; i++)
                    {
                        Owner.UtilityOutput.Host[i] = outputs[i];
                    }
                    Owner.UtilityOutput.SafeCopyToDevice();
                }
            }

            private void UpdateParams()
            {
                Owner.LearningParams.MinEpsilon = MinEpsilon;
                Owner.LearningParams.UseHierarchicalASM = UseHierarchicalASM;
                Owner.LearningParams.MotivationChange = MotivationChange;
                Owner.LearningParams.HierarchicalMotivationScale = 0.1f; // the higher in the hierarchy, the higher is the motivation 
                Owner.LearningParams.PropagateUtilitiesInHierarchy = PropagateUtilitiesInHierarchy;
            }
        }

        /// <summary>
        /// Mandatory task, cleans-up data after each step.
        /// </summary>
        [Description("Post-simulation step - mandatory"), MyTaskInfo(OneShot = false)]
        public class MyPostSimulationStepTask : MyTask<MyDiscreteHarmNode>
        {
            public override void Init(int nGPU)
            {
            }

            public override void Execute()
            {
                Owner.Hierarchy.PostSimulationStep();
            }
        }


        /// <summary>
        /// Monitors manual motivation inputs. 
        /// Overrides all inner motivations for actions if the threshold on the input MotivationsOverride is above 0.5.
        /// </summary>
        [Description("Manual Motivation Override"), MyTaskInfo(OneShot = false)]
        public class MyMotivationOverrideTask : MyTask<MyDiscreteHarmNode>
        {
            public override void Init(int nGPU)
            {
            }

            public override void Execute()
            {
                Owner.MotivationsOverrideInput.SafeCopyToHost();
                if (Owner.MotivationsOverrideInput.Host[0] < 0.5)
                {
                    return;
                }
                Owner.MotivationsInput.SafeCopyToHost();
                int noMotivations = Owner.MotivationsInput.Count;
                if (noMotivations > Owner.Rds.ActionManager.GetNoActions())
                {
                    noMotivations = Owner.Rds.ActionManager.GetNoActions();
                }
                for (int i = 0; i < noMotivations; i++)
                {
                    Owner.Rds.ActionManager.Actions[i].OverrideMotivationSourceWith(Owner.MotivationsInput.Host[i]);
                }
            }
        }

        /// <summary>
        /// Only a placeholder for Observer methods and it does need to be enabled.
        /// </summary>
        [Description("Data Visualization"), MyTaskInfo(OneShot = false)]
        public class MyActionUtilsVisualization : MyTask<MyDiscreteHarmNode>
        {
            private Dictionary<int, bool> warned;
            private Dictionary<String, MyVariable> variableWarned;

            public override void Init(int nGPU)
            {
                warned = new Dictionary<int, bool>();
                variableWarned = new Dictionary<string, MyVariable>();
            }

            public override void Execute()
            {
            }

            public MyStochasticReturnPredictor GetPredictorNo(int ind)
            {
                if (ind < 0 || ind >= Owner.Rds.ActionManager.Actions.Count)
                {
                    if (!warned.ContainsKey(ind))
                    {
                        warned.Add(ind, true);
                        MyLog.DEBUG.WriteLine("Observer: Abstract action index (" + ind + ") of predictor out of bounds");
                    }
                    return null;
                }
                int start = Owner.Rds.ActionManager.GetNoPrimitiveActinos();
                if (start + ind >= Owner.Rds.ActionManager.Actions.Count)
                {
                    if (!warned.ContainsKey(ind))
                    {
                        MyLog.DEBUG.WriteLine("Observer: No SRP of this index (" + ind + ") created so far");
                        warned.Add(ind, true);
                    }
                    return null;
                }
                if (Owner.Rds.ActionManager.Actions[start + ind] is MyStochasticReturnPredictor)
                {
                    if (warned.ContainsKey(ind))
                    {
                        warned.Remove(ind);
                    }
                    return (MyStochasticReturnPredictor)Owner.Rds.ActionManager.Actions[start + ind];
                }
                else
                {
                    MyLog.ERROR.WriteLine("Observer: Non-SRP action found where it should be!");
                    return null;
                }
            }

            /// <summary>
            /// If the variable can be controlled by own SRP, return it.
            /// </summary>
            /// <param name="ind">Global index of variable</param>
            /// <returns>SRP that controls the state of this variable</returns>
            public MyStochasticReturnPredictor GetSRPForVar(int ind)
            {
                if (ind < 0 || ind >= Owner.Rds.VarManager.GetMaxVariables())
                {
                    MyLog.DEBUG.WriteLine("Observer: Index of variable out of bounds");
                    return null;
                }
                return Owner.Rds.VarManager.GetVarNo(ind).MyAction;
            }


            /// <summary>
            /// For a given predictor, the method creates 2D array of max action utilities and max action labels over selected dimensions.
            /// The values in the memory are automatically scaled into the interval 0,1. Realtime values are multililed by motivations (therfore are bigger).
            /// </summary>
            /// <param name="values">array passed by reference for storing utilities of best action</param>
            /// <param name="labelIndexes">array of the same size for best action indexes</param>
            /// <param name="predictor">an asbtract action</param>
            /// <param name="XVarIndex">global index of state variable in the VariableManager</param>
            /// <param name="YVarIndex">the same: y axis</param>
            /// <param name="showRealtimeUtilities">show current utilities (scaled by motivations from the source and the hierarchy?)</param>
            public void ReadTwoDimensions(ref float[,] values, ref int[,] labelIndexes,
                MyStochasticReturnPredictor predictor, int XVarIndex, int YVarIndex, bool showRealtimeUtilities)
            {
                MyRootDecisionSpace rds = predictor.Rds;

                if (XVarIndex >= rds.VarManager.MAX_VARIABLES)
                {
                    XVarIndex = rds.VarManager.MAX_VARIABLES - 1;
                }
                if (YVarIndex >= rds.VarManager.MAX_VARIABLES)
                {
                    YVarIndex = rds.VarManager.MAX_VARIABLES - 1;
                }
                if (YVarIndex < 0)
                {
                    YVarIndex = 0;
                }
                if (XVarIndex < 0)
                {
                    XVarIndex = 0;
                }
                MyQSAMemory mem = predictor.Mem;
                int[] sizes = mem.GetStateSizes();              // size of the matrix
                int[] indexes = predictor.Ds.GetCurrentState(); // initial indexes

                int[] actionGlobalIndexes = mem.GetGlobalActionIndexes();   // global indexes of actions in the memory

                int promotedIndex = predictor.GetPromotedVariableIndex();

                MyVariable varX = rds.VarManager.GetVarNo(XVarIndex);
                MyVariable varY = rds.VarManager.GetVarNo(YVarIndex);

                float[] varXvals = varX.Values.ToArray();
                float[] varYvals = varY.Values.ToArray();

                Array.Sort(varXvals);
                Array.Sort(varYvals);

                int sx = 0;
                int sy = 0;
                if (XVarIndex == promotedIndex)
                {
                    sx = 1;
                    indexes[XVarIndex] = 0;

                    varXvals = new float[] { 0 };

                    sy = this.ReadSize(predictor.Ds, varY, YVarIndex, predictor.GetLabel());
                }
                else if (YVarIndex == promotedIndex)
                {
                    sy = 1;
                    indexes[YVarIndex] = 0;

                    varYvals = new float[] { 0 };

                    sx = this.ReadSize(predictor.Ds, varX, XVarIndex, predictor.GetLabel());
                }
                else
                {
                    sx = this.ReadSize(predictor.Ds, varX, XVarIndex, predictor.GetLabel());
                    sy = this.ReadSize(predictor.Ds, varY, YVarIndex, predictor.GetLabel());
                }

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
                        if (predictor.GetMaxMemoryValue() != 0)
                        {
                            for (int k = 0; k < utilities.Length; k++)
                            {
                                utilities[k] = utilities[k] / predictor.GetMaxMemoryValue();
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
                            values[i, j] = maxValue * predictor.GetMyTotalMotivation();
                        }
                        else
                        {
                            values[i, j] = maxValue;

                        }
                        labelIndexes[i, j] = actionGlobalIndexes[maxIndex];
                    }
                }
            }


            /// <summary>
            /// Size is given by the no of variables , but only in case that the variable is contained in the ds.  
            /// </summary>
            /// <returns></returns>
            private int ReadSize(IDecisionSpace ds, MyVariable v, int varInd, String predictorName)
            {
                if (ds.IsVariableIncluded(varInd))
                {
                    if (variableWarned.ContainsKey(predictorName + v.GetLabel()))
                    {
                        variableWarned.Remove(predictorName + v.GetLabel());
                    }
                    return v.Values.Count;
                }
                if (!variableWarned.ContainsKey(predictorName + v.GetLabel()))
                {
                    variableWarned.Add(predictorName + v.GetLabel(), v);
                    MyLog.DEBUG.WriteLine("Observer Warning: variable " + v.GetLabel() + " not contained in the DS: " + predictorName);
                }
                return 1;
            }
        }
    }
}

