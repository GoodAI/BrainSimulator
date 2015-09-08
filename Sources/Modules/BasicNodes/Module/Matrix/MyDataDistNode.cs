using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.Matrix
{

    /// <author> Honza Knopp</author>
    /// <status> Under dev., only a few of basic functions so far...</status>
    ///    
    /// <summary>
    /// </summary>
    /// <description>
    /// <h2> TODO: </h2>
    ///    User input (i.e. A+B*3-5)
    /// </description>
    public class MyDataDistNode : MyWorkingNode
    {


        [MyInputBlock(0)]
        public MyMemoryBlock<float> A
        {
            get { return GetInput(0); }
        }

        [MyInputBlock(1)]
        public MyMemoryBlock<float> B
        {
            get { return GetInput(1); }
        }


        [MyOutputBlock]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }






        [MyBrowsable, Category("Behavior"), YAXSerializableField(DefaultValue = Matrix.MatOperation.Multiplication), YAXElementFor("Behavior")]
        public Matrix.MatOperation Operation { get; set; }


        




        public override void UpdateMemoryBlocks()
        {
            Output.Count = 1;
        }

        
        
        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);
        }

        public override string Description
        {
            get
            {
                return Operation.ToString();
            }
        }

   

        public MyExecuteTask Execute { get; private set; }
        [Description("Execute")]
        public class MyExecuteTask : MyTask<MyDataDistNode>
        {
  

            //private MyMatrixOps mat_opCublas;
            private MyMatrixAutoOps mat_operation;



            public override void Init(int nGPU)
            {

                //mat_opCublas = new MyMatrixCublasOps(Owner);// | Matrix.MatOperation.MaxIndex, Owner.A);
                mat_operation = new MyMatrixAutoOps(Owner, Owner.Operation, Owner.A); // it may need A for setting up kernel size!
            }



            public override void Execute()
            {
                //mat_opCublas.Run(Owner.Operation, Owner.A, Owner.B, Owner.Output);
                if (Owner.B != null)
                {
                    mat_operation.Run(Owner.Operation, Owner.A, Owner.B, Owner.Output);
                }
                else
                {
                    mat_operation.Run(Owner.Operation, Owner.A,1,Owner.Output);
                }
            }



            public static double EvaluateString(string expression)
            {
                System.Data.DataTable table = new System.Data.DataTable();
                table.Columns.Add("expression", typeof(string), expression);
                System.Data.DataRow row = table.NewRow();
                table.Rows.Add(row);
                return double.Parse((string)row["expression"]);
            }
        }
    }



}