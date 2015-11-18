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
    /// <summary>Provides simple pattern-based signal routing</summary>
    /// <description>Node, which routes input signals to its output by a given pattern.</description>
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
        /// <br>Eeach number defines, for how long the given input will be copied to output. </br>
        /// 
        /// <br>E.g. - if you enter: 500,200,500 then for the first 500 steps, Input 1 will be copied into output, for next 200 steps Input 2 will be copied into output and then Input 3 will be copied into output for another 500 steps.</br>
        /// 
        /// <br>After that, 0 will be set as output or pattern will be applied again (according to Rotate param). </br>
        /// 
        /// <br>You can enter negative numbers too - they define, for how long output will be filled with 0. So -1000,500,200,-300,-100,500 is valid pattern too. Pattern can be longer than number of inputs. E.g. pattern -200,1000,-200,-200,0,1000,300,400,-200,600 with 3 inputs will route inputs: none, 1, none, 3, 1, 2, 3. While pattern which is shorter, will triger only the starting inputs. E.g. pattern -200,1000,100 with 3 inputs will give you only the first two into output.</br>
        /// </summary>
        [Description("Route inputs")]
        public class RoutingTask : MyTask<MultiplexerNode>
        {
            [MyBrowsable, Category("Parameters")]
            [YAXSerializableField(DefaultValue = false), Description("If True, Node will use the pattern forever. If False, pattern is applied only once and then output is filled with 0s")]
            public bool Rotate { get; set; }

            [MyBrowsable, Category("Parameters")]
            [YAXSerializableField(DefaultValue = "1000,100,1000"), Description("Comma-separated list of ints. Defines, how the node will route input signals.")]
            public string Pattern { get; set; }

            public override void Init(int nGPU)
            {
            }

            public override void Execute()
            {
                int[] signals = Array.ConvertAll(Pattern.Split(','), int.Parse);
                int[] abs = Array.ConvertAll(signals, Math.Abs);

                int mod = abs.Sum();
                if (!Rotate && SimulationStep >= mod)
                {
                    Owner.Output.Fill(0);
                    return;
                }
                int phase = (int)(SimulationStep % mod);

                int cnt = 0;
                for (int i = 0; i < signals.Length; i++)
                {
                    if (phase >= cnt && phase < cnt + abs[i])
                    {
                        if (signals[i] < 0)
                        {
                            Owner.Output.Fill(0);
                        }
                        else
                        {
                            //get negative numbers preceeding this number
                            int[] skipping = signals.Take(i).Where(x => x < 0).ToArray();
                            //skip idle phases from counting input index
                            int idx = (i - skipping.Length) % Owner.InputBranches;

                            Owner.GetInput(idx).CopyToMemoryBlock(Owner.Output, 0, 0, Owner.GetInput(idx).Count);
                        }
                    }
                    cnt += abs[i];
                }
            }
        }
    }
}
