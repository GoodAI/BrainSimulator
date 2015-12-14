using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace GoodAI.Modules.Common
{
    /// <author>GoodAI</author>
    /// <meta>mv</meta>
    /// <status>Working</status>
    /// <summary>Provides simple pattern-based data routing</summary>
    /// <description>Node, which routes input data to its output by a given pattern.</description>
    public class MultiplexerNode : MyWorkingNode, IMyVariableBranchViewNodeBase
    {
        [MyBrowsable, YAXSerializableField(DefaultValue = 0), YAXElementFor("IO")]
        public int OutputColHint { get; set; }

        [MyOutputBlock]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [ReadOnly(false)]
        [YAXSerializableField, YAXElementFor("IO")]
        public override int InputBranches
        {
            get { return base.InputBranches; }
            set
            {
                base.InputBranches = Math.Max(value, 2);
            }
        }

        private MyMemoryBlock<float>[] m_inputBlocks;

        public RoutingTask Routing { get; private set; }

        public MultiplexerNode()
        {
            InputBranches = 2;
        }

        public override void UpdateMemoryBlocks()
        {
            if (m_inputBlocks == null || m_inputBlocks.Length != InputBranches)
            {
                m_inputBlocks = new MyMemoryBlock<float>[InputBranches];
            }

            Output.ColumnHint = 1;

            for (int i = 0; i < InputBranches; i++)
            {
                MyMemoryBlock<float> ai = GetInput(i);
                m_inputBlocks[i] = ai;

                if (ai == null)
                    continue;

                // output will have the columnHint of the first vector that has columnHint > 1
                if (Output.ColumnHint == 1 && ai.ColumnHint > 1)
                {
                    Output.ColumnHint = ai.ColumnHint;
                }
            }

            // the auto-computed column hint may be overridden by the user:
            if (OutputColHint > 0)
            {
                Output.ColumnHint = OutputColHint;
            }

            Output.Count = m_inputBlocks[0] != null ? m_inputBlocks[0].Count : 0;
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);

            //All connected inputs should have same size
            for (int i = 0; i < InputBranches - 1; ++i)
            {
                validator.AssertError(m_inputBlocks[i].Count == m_inputBlocks[i+1].Count, this, "All inputs have to be the same size.");
            }
        }

        /// <summary>
        /// <br>Each number defines, for how long the given input will be copied to output. </br>
        /// 
        /// <br>E.g. - if you enter: 500,200,500 then for the first 500 steps, Input 1 will be copied into output, for next 200 steps Input 2 will be copied into output and then Input 3 will be copied into output for another 500 steps.</br>
        /// 
        /// <br>After that, 0 will be set as output or pattern will be applied again (according to Rotate param). </br>
        /// 
        /// <br>You can enter negative numbers too - they define, for how long output will be filled with 0. So -1000,500,200,-300,-100,500 is valid pattern too. </br>
        /// 
        /// <br>You can enter zero as well - that means that you want to skip whichever input should be that number for. Therefore 100,0,50 will output only first and third input</br>
        /// 
        /// <br>Pattern can be longer than number of inputs. 
        /// E.g. pattern -200,1000,-200,-200,0,1000,300,400,-200,600 with 3 inputs will route inputs: none, 1, none, 3, 1, 2, 3.</br>
        /// 
        /// <br>While pattern which is shorter, will triger only the starting inputs. 
        /// E.g. pattern -200,1000,100 with 3 inputs will give you only the first two into output.</br>
        /// </summary>
        [Description("Route inputs")]
        public class RoutingTask : MyTask<MultiplexerNode>
        {
            [MyBrowsable, Category("Parameters")]
            [YAXSerializableField(DefaultValue = false), Description("If True, Node will use the pattern forever. If False, pattern is applied only once and then output is filled with 0s")]
            public bool Rotate { get; set; }

            private string m_pattern;
            [MyBrowsable, Category("Parameters")]
            [YAXSerializableField(DefaultValue = "1000,100,1000"), Description("Comma-separated list of ints. Defines, how the node will route input signals.")]
            public string Pattern
            {
                get
                {
                    return m_pattern;
                }
                set
                {
                    try
                    {
                        m_signals = Array.ConvertAll(value.Split(','), int.Parse);
                        m_absSignals = Array.ConvertAll(m_signals, Math.Abs);
                        m_mod = m_absSignals.Sum();

                        m_pattern = value;

                        UpdateTransitions();
                    }
                    catch (FormatException e)
                    {
                        MyLog.ERROR.WriteLine(Owner.Id + ">" + e.Message);
                    }
                }
            }

            [MyBrowsable, Category("Parameters")]
            [YAXSerializableField(DefaultValue = false), Description("Will copy lastly copied input instead of zeros if set to True.")]
            public bool CopyLast { get; set; }

            private int[] m_signals;
            private int[] m_absSignals;
            private int m_mod;

            private int[] m_outputs;    //which input to copy to output in given phase
            private int[] m_transitions;    //when to change the phase
            private int m_idx;

            private void UpdateTransitions()
            {
                if (Owner == null)  // Owner is null during deserialization
                    return;
                int states = m_absSignals.Where(x => x > 0).Count();
                m_outputs = new int[states];
                m_transitions = new int[states];

                int idx = 0;
                int outputIdx = 0;
                for (int i = 0; i < m_signals.Length; ++i)
                {
                    int signal = m_signals[i];

                    if (signal == 0)    // zero - skip this input
                    {
                        outputIdx = (outputIdx + 1) % Owner.InputBranches;
                        continue;
                    }

                    if (signal < 0) // signal negative -> set -1 as input branch index -> interpret it as "fill with zeros" later
                        m_outputs[idx] = -1;
                    else
                    {
                        m_outputs[idx] = outputIdx % Owner.InputBranches;
                        outputIdx++;
                    }

                    m_transitions[idx] = Math.Abs(signal);  //set the length of exposition

                    if (idx > 0)
                        m_transitions[idx] += m_transitions[idx - 1];   //add all previous expositions if not on first element
                    idx++;
                }
                m_transitions[m_transitions.Length - 1] = 0;    //last transition happens when phase is 0 again
            }

            public override void Init(int nGPU)
            {
                UpdateTransitions();
                m_idx = 0;
            }

            public override void Execute()
            {
                if (SimulationStep == 0)
                    Owner.Output.Fill(0);

                if (!Rotate && SimulationStep >= m_mod)
                {
                    if (CopyLast)
                        Owner.GetInput(m_outputs[m_idx]).CopyToMemoryBlock(Owner.Output, 0, 0, Owner.GetInput(m_outputs[m_idx]).Count);
                    else
                        Owner.Output.Fill(0);
                    return;
                }

                int phase = (int)(SimulationStep % m_mod);
                if (phase == m_transitions[m_idx])
                    m_idx = (m_idx + 1) % m_transitions.Length;

                if (m_outputs[m_idx] == -1)
                    Owner.Output.Fill(0);
                else
                    Owner.GetInput(m_outputs[m_idx]).CopyToMemoryBlock(Owner.Output, 0, 0, Owner.GetInput(m_outputs[m_idx]).Count);
            }
        }
    }
}
