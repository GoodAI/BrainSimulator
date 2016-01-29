using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using YAXLib;

namespace GoodAI.Modules.Transforms
{

    public abstract class MyTransform : MyWorkingNode
    {
        [MyOutputBlock]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }
              
        public int OutputSize
        {
            get { return Output.Count; }
            set { Output.Count = value; }
        }

        [MyInputBlock(0)]
        public MyMemoryBlock<float> Input
        {
            get { return GetInput(0); }
        }
        
        public int InputSize
        {
            get { return Input != null ? Input.Count : 0; }
        }

        public override void UpdateMemoryBlocks()
        {
            OutputSize = InputSize;

            if (Input != null)
            {
                Output.ColumnHint = Input.ColumnHint;
            }
        }
    }
    /// <author>GoodAI</author>
    /// <meta>mb</meta>
    /// <status>Working</status>
    /// <summary>Returns absolute for each element in the input memory block.</summary>
    /// <description>
    /// 
    /// </description>
    [YAXSerializeAs("AbsoluteValue")]  
    public class MyAbsoluteValue : MyTransform
    {
        public MyMemoryBlock<float> Temp { get; private set; }

        public override void UpdateMemoryBlocks()
        {
            base.UpdateMemoryBlocks();
            Temp.Count = 1;
        }
        /// <summary>
        /// The node also provides couple of normalizations such as vector normalization and the scalar normalization.
        /// </summary>
        [Description("Absolute Value")]
        public class MyAbsoluteValueTask : MyTask<MyAbsoluteValue>
        {
            private MyCudaKernel m_kernel;
            private MyProductKernel<float> m_dotKernel;
            private MyReductionKernel<float> m_sumKernel;
            private MyCudaKernel m_mulKernel;

            [MyBrowsable, Category("Params")]
            [YAXSerializableField]
            public bool VectorNormalization { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField]
            public bool ScalarNormalization { get; set; }

            public override void Init(int nGPU)
            {
                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\TransformKernels", "AbsoluteValueKernel");
                m_kernel.SetupExecution(Owner.OutputSize);

                m_mulKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\TransformKernels", "PolynomialFunctionKernel");
                m_mulKernel.SetupExecution(Owner.OutputSize);

                m_dotKernel = MyKernelFactory.Instance.KernelProduct<float>(Owner, nGPU, ProductMode.f_DotProduct_f);
                m_sumKernel = MyKernelFactory.Instance.KernelReduction<float>(Owner, nGPU, ReductionMode.f_Sum_f);
            }

            public override void Execute()
            {
                if (VectorNormalization)
                {
                    //ZXC m_dotKernel.Run(Owner.Temp, 0, Owner.Input, Owner.Input, Owner.InputSize, /* distributed: */ 0);
                    m_dotKernel.Run(Owner.Temp, Owner.Input, Owner.Input);
                    Owner.Temp.SafeCopyToHost();
                    float length = (float)Math.Sqrt(Owner.Temp.Host[0]);

                    if (length != 0)
                    {
                        m_mulKernel.Run(0f, 0f, 1.0f / length, 0f, Owner.Input, Owner.Output, Owner.InputSize);
                    }
                    else
                    {
                        Owner.Output.Fill(0);
                    }
                }
                else if (ScalarNormalization)
                {
                    //ZXC m_sumKernel.Run(Owner.Temp, Owner.Input, Owner.InputSize, 0, 0, 1, /* distributed: */ 0);
                    m_sumKernel.Run(Owner.Temp, Owner.Input);
                    Owner.Temp.SafeCopyToHost();

                    float length = Owner.Temp.Host[0];

                    if (length != 0)
                    {
                        m_mulKernel.Run(0f, 0f, 1.0f / length, 0f, Owner.Input, Owner.Output, Owner.InputSize);
                    }
                    else
                    {
                        Owner.Output.Fill(0);
                    }
                }
                else
                {
                    m_kernel.Run(Owner.Input, Owner.Output, Owner.OutputSize);
                }
            }
        }

        public MyAbsoluteValueTask DoTransform { get; private set; }

        public override string Description
        {
            get
            {
                return DoTransform.VectorNormalization ? "f(x) = x / |x|" : "f(x) = |x|";
            }
        }
    }

    /// <author>GoodAI</author>
    /// <meta>xx</meta>
    /// <status>Working</status>
    /// 
    /// <summary>Reduction of the input.</summary>
    /// 
    /// <description>
    ///  The node applies several reduction rechniques to scale down the input memory block.
    ///  
    ///  <h3> Operation </h3>
    ///    The desired reduction technique is always in the form "input"_"operation type"_"output". So "f_MinIdx_fi" means that the input is a float memory block on which
    ///    the Min function is applied and the output is: minimal value as a float, and its index as integer.  "i_MinIdxMaxIdx_4i" takes integer as an input and the output is four integers: minimal value of the
    ///    input, index of the min-value, maximum value of the input and the max-value index
    ///  
    /// </description>
    [YAXSerializeAs("Reduction")]
    public class MyReduction : MyTransform
    {
        [MyBrowsable, Category("Params"), TypeConverter(typeof(MyReductionModeTypeConverter))]
        [YAXSerializableField(DefaultValue = ReductionMode.f_Sum_f)]
        public ReductionMode Mode { get; set; }

        public class MyReductionModeTypeConverter : EnumConverter
        {
            public MyReductionModeTypeConverter() : base(typeof(ReductionMode)) { }

            public override bool GetStandardValuesSupported(ITypeDescriptorContext context)
            {
                return true;
            }

            public override StandardValuesCollection GetStandardValues(ITypeDescriptorContext context)
            {
                List<ReductionMode> standardValues = new List<ReductionMode>();
                foreach (ReductionMode m in typeof(ReductionMode).GetEnumValues())
                    standardValues.Add(m);
                return new StandardValuesCollection(standardValues);
            }
        }

        /// <summary>
        /// Performs the reduction operation.
        /// </summary>
        [Description("Reduction")]
        public class MyReductionTask : MyTask<MyReduction>
        {
            private MyReductionKernel<float> m_kernel;

            public override void Init(int nGPU)
            {
                m_kernel = MyKernelFactory.Instance.KernelReduction<float>(Owner, nGPU, Owner.Mode);
            }

            public override void Execute()
            {
                // no in offset, no out offset, stride 1
                //ZXC m_kernel.Run(Owner.Output, Owner.Input, Owner.InputSize, 0, 0, 1, /* distributed: */ 0);
                m_kernel.Run(Owner.Output, Owner.Input);
            }
        }

        public MyReductionTask DoTransform { get; private set; }

        public override string Description
        {
            get
            {
                string desc = Mode.ToString().Split('_')[1];
                switch (desc)
                {
                    case "Sum":
                        return "f(x)=\u2211x";
                    case "MinIdx":
                        return "[min(x),idx]";
                    case "MaxIdx":
                        return "[max(x),idx]";
                    case "MinMax":
                        return "[min(x),max(x)]";
                    case "MinIdxMaxIdx":
                        return "[min(x),idx,max(x),idx]";
                    default:
                        return "f(x)=...";
                }
            }
        }

        public override void UpdateMemoryBlocks()
        {
            string sig = Mode.ToString().Split('_')[2];
            if (sig.Length == 1)
                OutputSize = 1;
            else if (sig.Length == 4 || sig[0] == '4')
                OutputSize = 4;
            else
                OutputSize = 2;
        }
    }


    /// <author>GoodAI</author>
    /// <meta>xx</meta>
    /// <status>Working</status>
    /// 
    /// <summary>Reduction of the input.</summary>
    /// 
    /// <description>
    ///  The node applies several reduction rechniques to scale down the input memory block.
    ///  
    ///  <h3> Operation </h3>
    ///    The desired reduction technique is always in the form "input"_"operation type"_"output". So "f_MinIdx_fi" means that the input is a float memory block on which
    ///    the Min function is applied and the output is: minimal value as a float, and its index as integer.  "i_MinIdxMaxIdx_4i" takes integer as an input and the output is four integers: minimal value of the
    ///    input, index of the min-value, maximum value of the input and the max-value index
    ///  
    /// </description>
    //[YAXSerializeAs("Product")]
    //public class MyProduct : MyTransform
    //{
    //    [MyBrowsable, Category("Params"), TypeConverter(typeof(MyProductModeTypeConverter))]
    //    [YAXSerializableField(DefaultValue = ProductMode.f_DotProduct_f)]
    //    public ProductMode Mode { get; set; }

    //    public class MyProductModeTypeConverter : EnumConverter
    //    {
    //        public MyProductModeTypeConverter() : base(typeof(ProductMode)) { }

    //        public override bool GetStandardValuesSupported(ITypeDescriptorContext context)
    //        {
    //            return true;
    //        }

    //        public override StandardValuesCollection GetStandardValues(ITypeDescriptorContext context)
    //        {
    //            List<ProductMode> standardValues = new List<ProductMode>();
    //            foreach (ProductMode m in typeof(ProductMode).GetEnumValues())
    //                if (m.ToString().Split('_')[1] != "Cosine" && m.ToString().Split('_')[1] != "DotProduct")
    //                    standardValues.Add(m);
    //            return new StandardValuesCollection(standardValues);
    //        }
    //    }

    //    /// <summary>
    //    /// Performs the reduction operation.
    //    /// </summary>
    //    [Description("Product")]
    //    public class MyProductTask : MyTask<MyProduct>
    //    {
    //        private MyProductKernel<float> m_kernel;

    //        public override void Init(int nGPU)
    //        {
    //            m_kernel = MyKernelFactory.Instance.KernelProduct<float>(Owner, nGPU, Owner.Mode);
    //        }

    //        public override void Execute()
    //        {
    //            // no in offset, no out offset, stride 1
    //            //ZXC m_kernel.Run(Owner.Output, Owner.Input, Owner.InputSize, 0, 0, 1, /* distributed: */ 0);
    //            m_kernel.Run(Owner.Output, Owner.Input1, Owner.Input2);
    //        }
    //    }

    //    public MyProductTask DoTransform { get; private set; }

    //    public override string Description
    //    {
    //        get
    //        {
    //            string desc = Mode.ToString().Split('_')[1];
    //            switch (desc)
    //            {
    //                case "Sum":
    //                    return "f(x)=\u2211x";
    //                case "MinIdx":
    //                    return "[min(x),idx]";
    //                case "MaxIdx":
    //                    return "[max(x),idx]";
    //                case "MinMax":
    //                    return "[min(x),max(x)]";
    //                case "MinIdxMaxIdx":
    //                    return "[min(x),idx,max(x),idx]";
    //                default:
    //                    return "f(x)=...";
    //            }
    //        }
    //    }

    //    public override void UpdateMemoryBlocks()
    //    {
    //        string sig = Mode.ToString().Split('_')[2];
    //        if (sig.Length == 1)
    //            OutputSize = 1;
    //        else if (sig.Length == 4 || sig[0] == '4')
    //            OutputSize = 4;
    //        else
    //            OutputSize = 2;
    //    }
    //}


    /// <author>GoodAI</author>
    /// <meta>mb</meta>
    /// <status>Working</status>
    /// <summary>For each input indicates (by 1.0f) wheather its value is in an interval.</summary>
    /// <description>
    /// Set <b>Levels</b> to number of interval you want to indicate. 
    /// The size of output equals size of input times number of selected levels.
    /// </description>
    [YAXSerializeAs("Thresholding")]
    public class MyThreshold : MyTransform
    {

        [MyBrowsable, Category("Params")]
        [YAXAttributeFor("Params"), YAXSerializableField(DefaultValue = 2)]
        public int Levels { get; set; }

        /// <summary>
        /// Set <b>Minimum</b> and <b>Maximum</b> for interval which is to be indicated.<br/>
        /// 1.0f is then assigned to the i-th position if the value falls into i-th interval
        /// of length (Maximum - Minimum) / Levels; 0.0f is assigned otherwise.
        /// If <b>StrictThreshold</b> is true, then values that fall outside of interval
        /// &lt;Minimum, Maximum&gt; will not be in any of the Levels intervals. Otherwise
        /// input values less than Minimum are internally changed to Minimum and vice versa for Maximum.
        /// </summary>
        [Description("Threshold")]
        public class MyThresholdTask : MyTask<MyThreshold>
        {
            private MyCudaKernel m_kernel;

            [MyBrowsable, Category("Params"), DisplayName("M\tinimum")]
            [YAXAttributeFor("Params"), YAXSerializableField(DefaultValue = float.NegativeInfinity)]
            public float Minimum { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXAttributeFor("Params"), YAXSerializableField(DefaultValue = float.PositiveInfinity)]
            public float Maximum { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXAttributeFor("Params"), YAXSerializableField(DefaultValue = false)]
            public bool StrictThreshold {get; set; }

            public override void Init(int nGPU)
            {
                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\TransformKernels", "ThresholdKernel");
            }

            public override void Execute()
            {
                m_kernel.SetupExecution(Owner.InputSize);
                m_kernel.Run(Minimum, Maximum, StrictThreshold ? 1 : 0, Owner.Input, Owner.Output, Owner.InputSize, Owner.Levels);
            }
        }

        public MyThresholdTask DoTransform { get; private set; }

        public override string Description
        {
            get
            {
                if (DoTransform.StrictThreshold)
                {
                    return "__\u2015\u203E\u203E strict";
                }
                else
                {
                    return "__\u2015\u203E\u203E";
                }
            }
        }

        public override void UpdateMemoryBlocks()
        {
            OutputSize = InputSize * Levels;
        }
    }
    /// <author>GoodAI</author>
    /// <meta>mb</meta>
    /// <status>Working</status>
    /// <summary>Goniometric function</summary>
    /// <description>
    /// The node applies user specified goniometric function on each element of the input memory block.
    /// </description>
    [YAXSerializeAs("GoniometricFunction")]
    public class MyGoniometricFunction : MyTransform
    {
        public enum MyGonioType
        {
            [Description("sin(x)")]
            Sine = 0,
            [Description("cos(x)")]
            Cosine = 1,
            [Description("tan(x)")]
            Tan = 2,
            [Description("tanh(x)")]
            Tanh = 3,
            [Description("sinh(x)")]
            Sinh = 4,
            [Description("cosh(x)")]
            Cosh = 5,
            [Description("asin(x)")]
            Asin = 6,
            [Description("acos(x)")]
            Acos = 7,
            [Description("atan2(x,y)")]
            Atan2 = 10  // 8 and 9 is for atan and acot
        }

        [MyBrowsable, Category("Params")]
        [YAXAttributeFor("Params"), YAXSerializableField(DefaultValue = MyGonioType.Sine)]
        public MyGonioType Type { get; set; }

        public MyGoniometricTask DoTransform { get; private set; }

        public override string Description
        {
            get
            {
                return "f(x) = " + Type.GetAttributeProperty((DescriptionAttribute x) => x.Description);
            }
        }

        public override void UpdateMemoryBlocks()
        {
            if (Type == MyGonioType.Atan2)
                OutputSize = InputSize / 2;
            else
                OutputSize = InputSize;
        }

        /// <summary>
        /// The node contains sine, cosine, tangent and their hyperbolic and inverse equivalents. Atan2 takes pairs of floats on input.
        /// </summary>
        [Description("Goniometric")]
        public class MyGoniometricTask : MyTask<MyGoniometricFunction>
        {
            private MyCudaKernel m_kernel;

            public override void Init(int nGPU)
            {
                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\TransformKernels", "GoniometricFunctionKernel");
            }

            public override void Execute()
            {
                m_kernel.SetupExecution(Owner.OutputSize);
                m_kernel.Run(Owner.Input, Owner.Output, Owner.InputSize, (int)Owner.Type);
            }
        }
    }

    /// <author>GoodAI</author>
    /// <meta>df</meta>
    /// <status>Working</status>
    /// <summary>Applies polynomial function on each member of input array.</summary>
    /// <description>f(x) = a<sub>3</sub>x<sup>3</sup> + a<sub>2</sub>x<sup>2</sup> + a<sub>1</sub>x + a<sub>0</sub></description>
    [YAXSerializeAs("PolynomialFunction")]
    public class MyPolynomialFunction : MyTransform
    {
        /// <summary>Applies polynomial function with given coeffitients (up to third degree) on all input values.</summary>
        [Description("Polynomial Function")]
        public class MyPolynomialFunctionTask : MyTask<MyTransform>
        {
            private MyCudaKernel m_kernel;

            [MyBrowsable, Category("Params")]
            [YAXAttributeFor("Params"), YAXSerializableField(DefaultValue = 0)]
            public float A3 { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXAttributeFor("Params"), YAXSerializableField(DefaultValue = 0)]
            public float A2 { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXAttributeFor("Params"), YAXSerializableField(DefaultValue = 1)]
            public float A1 { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXAttributeFor("Params"), YAXSerializableField(DefaultValue = 0)]
            public float A0 { get; set; }

            public override void Init(int nGPU)
            {
                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\TransformKernels", "PolynomialFunctionKernel");
            }

            public override void Execute()
            {
                m_kernel.SetupExecution(Owner.OutputSize);
                m_kernel.Run(A3, A2, A1, A0, Owner.Input, Owner.Output, Owner.OutputSize);
            }      
      
            public string Description 
            {
                get 
                {
                    string result = "";

                    if (A3 != 0)
                    {
                        if (A3 < 0) result += "-";
                        if (Math.Abs(A3) != 1) result += Math.Abs(A3);
                        result += "x^3";
                    } 
                    if (A2 != 0)
                    {
                        if (result != "" && A2 > 0) result += "+";
                        if (A2 < 0) result += "-";
                        if (Math.Abs(A2) != 1) result += Math.Abs(A2);
                        result += "x^2";
                    }
                    if (A1 != 0)
                    {
                        if (result != "" && A1 > 0) result += "+";
                        if (A1 < 0) result += "-";
                        if (Math.Abs(A1) != 1) result += Math.Abs(A1);
                        result += "x";
                    }
                    if (A0 != 0)
                    {
                        if (result != "" && A0 > 0) result += "+";                        
                        result += A0;
                    }
                    else if (result == "")
                    {
                        result = "0";
                    }

                    return "f(x)=" + result;
                }
            }

        }

        public MyPolynomialFunctionTask DoTransform { get; private set; }

        public override string Description
        {
            get
            {
                if (DoTransform != null) 
                {
                    return DoTransform.Description;
                }
                else 
                {
                    return base.Description;
                }
            }
        }
    }

    /// <author>GoodAI</author>
    /// <meta>mb,jk,df</meta>
    /// <status>Working</status>
    /// <summary>Filter.</summary>
    /// <description>
    ///   Node to apply a filter to each element of the input memory block.
    /// </description>
    public class MyLowHighFilter : MyTransform
    {
        /// <summary>Node to restrict the range of each element of the input memory block.
        ///   There are two methods to apply:
        ///   <ul>
        ///    <li> Standard: simply cuts all values higher or lower. </li> 
        ///    <li> Modulo: apllies the <b>modulus operator</b> that computes the remainder from the integer division. So the result is: ''value % Maximum + Minimum''</li> 
        ///   </ul>
        /// </summary>
        [Description("Range Restriction")]
        public class MyLowHighFilterTask : MyTask<MyTransform>
        {
            private MyCudaKernel m_kernel;

            [MyBrowsable, Category("Params"), DisplayName("M\tinimum")]
            [YAXAttributeFor("Params"), YAXSerializableField(DefaultValue = float.NegativeInfinity)]
            public float Minimum { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXAttributeFor("Params"), YAXSerializableField(DefaultValue = float.PositiveInfinity)]
            public float Maximum { get; set; }

            public enum MyRangeRestrOperation
            {
                Standard,
                Modulo
            }

            [MyBrowsable, Category("Params")]
            [YAXAttributeFor("Params"), YAXSerializableField(DefaultValue = MyRangeRestrOperation.Standard)]
            public MyRangeRestrOperation RangeRestrOperation { get; set; }


            public override void Init(int nGPU)
            {
                if (RangeRestrOperation == MyRangeRestrOperation.Modulo)
                {
                    //m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\TransformKernels", "ModuloKernel");
                }
                else
                { // standard!
                    m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\TransformKernels", "CropKernel");
                }
            }

            public override void Execute()
            {
                if (RangeRestrOperation == MyRangeRestrOperation.Modulo)
                {
                    Owner.Input.SafeCopyToHost();
                    Owner.Output.SafeCopyToHost();
                    /*int min =  int.MaxValue, max = int.MinValue;
                    //--- find minimum
                    foreach (float val in Owner.Input.Host)
                    {
                        if (val < min) min = (int)val;
                        if (val > max) max = (int)val;
                    }*/
                    //--- move values to zero, compute modulo, move them back...
                    for (int i = 0; i < Owner.OutputSize; i++)
                    {
                        Owner.Output.Host[i] = (int)Owner.Input.Host[i] % (int)Maximum + Minimum;
                    }
                    //m_kernel.Run(Owner.Input, Divisor, Owner.Output, Owner.OutputSize);                    
                    Owner.Output.SafeCopyToDevice();
                }
                else
                {
                    m_kernel.SetupExecution(Owner.OutputSize);
                    m_kernel.Run(Minimum, Maximum, Owner.Input, Owner.Output, Owner.OutputSize);
                }
            }

            public string Description
            {
                get
                {
                    string result = "";
                    if (RangeRestrOperation == MyRangeRestrOperation.Modulo)
                    {
                        result += "%: ";
                    }
                    if (float.IsInfinity(Minimum))
                    {
                        result += Minimum < 0 ? "-\u221E" : "\u221E";
                    }
                    else
                    {
                        result += Minimum;
                    }

                    result += " < f(x) < ";

                    if (float.IsInfinity(Maximum))
                    {
                        result += Maximum < 0 ? "-\u221E" : "\u221E";
                    }
                    else
                    {
                        result += Maximum;
                    }

                    return result;
                }
            }
        }

        /// <summary> Returns index of the max value in the mem. block </summary>
        [Description("Find max value ")]
        public class MyFindMaxTask : MyTask<MyTransform>
        {
            public override void Init(Int32 nGPU)
            {

            }

            public override void Execute()
            {
                Owner.Input.SafeCopyToHost();
                float maxValue = Owner.Input.Host.Max();

                for (int i = 0; i < Owner.InputSize; i++)
                {
                    if (Owner.Input.Host[i] != maxValue)
                    {
                        Owner.Output.Host[i] = 0.00f;
                    }
                    else
                    {
                        Owner.Output.Host[i] = 1.00f;
                    }
                }
                Owner.Output.SafeCopyToDevice();
            }
        }

        /// <summary> Rounds elements in the mem. block. </summary>
        [Description("Round")]
        public class MyRoundTask : MyTask<MyTransform>
        {
            private MyCudaKernel m_kernel;

            public override void Init(Int32 nGPU)
            {
                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\TransformKernels", "RoundKernel");
                m_kernel.SetupExecution(Owner.OutputSize);
            }
            public override void Execute()
            {
                m_kernel.Run(Owner.Input, Owner.Output, Owner.Output.Count);
            }
        }

        /// <summary> Rounds elemetns in the input mem. block. downwards. </summary>
        [Description("Floor")]
        public class MyFloorTask : MyTask<MyTransform>
        {
            private MyCudaKernel m_kernel;

            public override void Init(Int32 nGPU)
            {
                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\TransformKernels", "FloorKernel");
                m_kernel.SetupExecution(Owner.OutputSize);
            }
            public override void Execute()
            {
                m_kernel.Run(Owner.Input, Owner.Output, Owner.Output.Count); 
            }
        }

        [MyTaskGroup("Mode")]
        public MyLowHighFilterTask DoTransform { get; private set; }
        [MyTaskGroup("Mode")]
        public MyFindMaxTask FindMaxValue { get; private set; }
        [MyTaskGroup("Mode")]
        public MyRoundTask DoRound  { get; private set; }
        [MyTaskGroup("Mode")]
        public MyFloorTask DoFloor { get; private set; }
        //public MyModuloTask DoModulo { get; private set; }

        public override string Description
        {
            get
            {
                if (DoTransform != null && DoTransform.Enabled)
                {
                    return DoTransform.Description;
                }
                else if (FindMaxValue != null && FindMaxValue.Enabled)
                {
                    return "f(x)=max(x)";
                }
                else if (DoRound != null && DoRound.Enabled)
                {
                    return "f(x)=round(x)";
                }
                else if (DoFloor != null && DoFloor.Enabled)
                {
                    return "f(x)=floor(x)";
                }
                else
                {
                    return base.Description;
                }
            }
        }
    }  
}
