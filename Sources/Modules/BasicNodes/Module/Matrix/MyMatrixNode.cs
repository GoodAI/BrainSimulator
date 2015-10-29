using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.Matrix
{

    /// <author>GoodAI</author>
    /// <meta>jk</meta>
    /// <status> Working </status>
    /// <summary>
    ///   This performs several operations like addition, multiplication etc.
    /// </summary>
    /// <description>
    /// 
    /// <h3> Features: </h3>
    /// <ul>
    ///  <li> Allows multiplication addition with different input sizes, so instead of only A*B (where A,B are matrices), it supports A*v (v is vector), v*A or even const*A. </li>
    ///  <li> For several opts (getRow, const*A), two input types are supported: 1) memory block from another node; 2) number in "ExecuteParams/DataInput".</li>
    ///  <li> This is just a node layer above the MatrixAutoOps class, so you can use it simple in your code too.</li>
    /// </ul>
    /// 
    /// <h3> Usage of operations: </h3>
    /// <ul>
    ///  <li><b>Addition,Multiplication,Substraction,MultiplElemntWise </b> (two memBlock inputs, or one into A and set the DataInput0 paramter): Two inputs (each of them can be matrix, or vector, or constat). Be careful about the coorect sizes/dimensions of the inputs, it does column/row-wise operation. If only input to the A, then it perforsm multuiplication with the value at DataInput.</li>
    ///  <li><b>DotProd </b> (two memBlock inputs): performs trans(vec) * vec. be carful about the size and dimensions.</li>
    ///  <li><b>MinIndex, MaxIndex </b> (one mem block input): returns min/max index in the vector of its Absolute values.</li>
    ///  <li><b>GetCol,GetRow </b> (two memBlock inputs, or one into A and set the DataInput0 paramter): returns n-th column of the input. N can be DataInput0 or value in the memory block in the B-input.</li>
    ///  <li><b>Minus </b> (one memBlock input): returns -A</li>
    ///  <li><b>Normalize </b> (one memBlock input): return normalized matrix A, Norm2 used in this case.</li>
    ///  <li><b>Norm2 </b> (one memBlock input): returns norm2 of the matrix A</li>
    ///  <li><b>Exp, Abs, Log, Round, Floor, Ceil </b> (one memBlock input): returns Exp/Log/Abs/Floor/Round/Ceil of each element of the matrix A as a new matrix.</li>
    /// <ul>
    /// <h3> Parmaters</h3>
    /// </ul>
    ///   <li> <b>Behaviour/Operation:</b> Matrix operation. </li>
    ///   <li> <b>Execute/DataInput0:</b> If one value should be inserted directly, the par. has usage in Multipl, Additon, GetRow... </li>
    /// </ul>
    /// </description>

    public class MyMatrixNode : MyWorkingNode
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






        [MyBrowsable, Category("Behavior"), YAXSerializableField(DefaultValue = Matrix.MatOperation.Multiplication), YAXElementFor("Behavior"), Description("Matrix operation")]
        public Matrix.MatOperation Operation { get; set; }


        public override void UpdateMemoryBlocks()
        {
            Output = MyMatrixOps.SetupResultSize(Operation, A, B, Output);
        }



        public override void Validate(MyValidator validator)
        {
            validator.AssertError(MyMatrixOps.Validate(Operation, A, B, Output), this, "Wrong matrix dimensions for the specific operation");
        }

        public override string Description
        {
            get
            {
                return Operation.ToString();
            }
        }



        public MyExecuteTask Execute { get; private set; }
        /// <summary>
        /// parameter ,,Execute/params/DataInput0'' can be used for some operations when second input is not given. For example Addition, Multiplication, GetRow, GetCol...
        /// </summary>
        [Description("Execute")]
        public class MyExecuteTask : MyTask<MyMatrixNode>
        {
            private MyMatrixAutoOps mat_operation;

            [MyBrowsable, Category("Params"), YAXSerializableField(DefaultValue = float.NaN), Description("If one value should be inserted directly, the par. has usage in Multipl, Additon, GetRow...")]
            public float DataInput0 { get; set; }


            public override void Init(int nGPU)
            {
                mat_operation = new MyMatrixAutoOps(Owner, Owner.Operation, Owner.A); // it may need A for setting up kernel size!
            }


            public override void Execute()
            {
                if (Owner.B != null)
                {
                    mat_operation.Run(Owner.Operation, Owner.A, Owner.B, Owner.Output);
                }
                else
                {
                    if (float.IsNaN(DataInput0))
                    {
                        mat_operation.Run(Owner.Operation, Owner.A, Owner.Output);
                    }
                    else
                    {
                        mat_operation.Run(Owner.Operation, Owner.A, DataInput0, Owner.Output);
                    }
                }
            }


        }
    }



}