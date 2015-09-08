using GoodAI.Core.Memory;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using YAXLib;

namespace GoodAI.Core.Nodes
{

    /// <author>GoodAI</author>
    /// <meta>df,mb</meta>
    /// <status>Working</status>
    /// <summary>Splits input to several outputs</summary>
    /// <description>Use Branches property to specify splitting.
    /// Sizes are specified as comma separated list. You can use '*', in places which should be
    /// calculated automatically. </description>
    public class MyFork : MyWorkingNode
    {
        [MyInputBlock]
        public MyMemoryBlock<float> Input
        {
            get { return GetInput(0); }
        }
        
        public int InputSize
        {
            get { return Input != null ? Input.Count : 0; }
        }  

        private string m_branches;
        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = "1,1"), YAXElementFor("IO")]
        public string Branches
        {
            get { return m_branches; }
            set
            {
                m_branches = value;
                InitOutputs();
            }
        }        

        public void InitOutputs()
        {          
            if (Branches != null && Branches != String.Empty)
            {
                string[] branchConf = Branches.Split(',');

                if (branchConf.Length != OutputBranches) {

                    for (int i = 0; i < OutputBranches; i++)
                    {
                        MyMemoryBlock<float> mb = GetOutput(i);
                        MyMemoryManager.Instance.RemoveBlock(this, mb);
                    }

                    OutputBranches = branchConf.Length;

                    for (int i = 0; i < branchConf.Length; i++)
                    {
                        MyMemoryBlock<float> mb = MyMemoryManager.Instance.CreateMemoryBlock<float>(this);
                        mb.Name = "Output_" + (i + 1);
                        mb.Count = -1;
                        m_outputs[i] = mb;
                    }
                }                               

               UpdateMemoryBlocks();                         
            }
        }

        private void UpdateOutputBlocks()
        {
            if (Branches != null && Branches != String.Empty)
            {
                string[] branchConf = Branches.Split(',');
                int[] branchSizes = new int[OutputBranches];

                List<int> stars = new List<int>();
                int sum = 0;
                for (int i = 0; i < branchConf.Length; i++)
                {
                    if (String.Equals(branchConf[i].Trim(), "*"))
                    {
                        stars.Add(i);
                    }
                    else
                    {
                        branchSizes[i] = int.Parse(branchConf[i], CultureInfo.InvariantCulture);
                        sum += branchSizes[i];
                    }
                }

                if (stars.Count > 0)
                {
                    //TODO fix for InputSize == 0, i.e. when a .brain file is loaded
                    int starSize = (InputSize - sum) / stars.Count;
                    int rest = InputSize - sum - starSize * stars.Count;

                    for (int j = 0; j < stars.Count; j++)
                    {
                        branchSizes[stars[j]] = starSize;
                    }
                    branchSizes[stars[stars.Count - 1]] += rest;
                }

                for (int i = 0; i < branchConf.Length; i++)
                {
                    GetOutput(i).Count = branchSizes[i];
                }
            }
        }

        public override void UpdateMemoryBlocks()
        {
            UpdateOutputBlocks();

            //column hint update
            for (int i = 0; i < OutputBranches; i++)
            {
                MyMemoryBlock<float> mb = GetOutput(i);

                if (Input != null && mb.Count > Input.ColumnHint)
                {
                    mb.ColumnHint = Input.ColumnHint;
                }
            }
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);
            int totalOutputs = 0;

            for (int i = 0; i < OutputBranches; i++)
            {
                totalOutputs += GetOutput(i).Count;
                validator.AssertError(GetOutput(i).Count > 0, this, "Invalid size of '" + GetOutput(i).Name + "'. Check 'Branches' setting.");
            }

            validator.AssertError(totalOutputs == InputSize, this, "Sum of ouput sizes must be equal to the input size");
        }

        public MyForkTask DoFork { get; private set; }

        /// <summary>
        /// The input is split and copied to outputs according the given output sizes.
        /// </summary>
        [Description("Perform fork operation")]
        public class MyForkTask : MyTask<MyFork>
        {
            public override void Init(int nGPU)
            {
                
            }

            public override void Execute()
            {
                int offset = 0;

                for (int i = 0; i < Owner.OutputBranches; i++)
                {
                    MyMemoryBlock<float> ai = Owner.GetOutput(i);

                    if (ai != null)
                    {
                        Owner.Input.CopyToMemoryBlock(ai, offset, 0, ai.Count);
                        offset += ai.Count;
                    }
                }                               
            }
        }
    }
}
