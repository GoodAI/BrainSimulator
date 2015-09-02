using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.VSA
{
    /// <author>GoodAI</author>
    /// <tag>#mm</tag>
    /// <status>Not Optimized</status>
    /// <summary>
    ///   Generates static random permutations with either a single cycle or random cycles.
    ///   Optionally, generates random combinations with either unique numbers or allowed duplicates.
    /// </summary>
    /// <description>
    ///   A single cycle is created by applying the Sattolo's shuffle (as seen on <seealso cref="http://en.wikipedia.org/wiki/Fisher–Yates_shuffle#Sattolo.27s_algorithm"/>). 
    ///   Random cycles are created by using the Fisher-Yates shuffle (as seen on <seealso cref="http://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_.22inside-out.22_algorithm"/>).
    ///   Use the <see cref="Power"/> property to pre-compute a desired power of the base permutation.
    /// </description>
    public class MyCombinationBook : MyCombinationBase
    {
        public enum CombinationMode
        {
            PermutationOneCycle,
            PermutationRandomCycles,
            MakePower,
            CombinationRandom,
            CombinationUnique,
        }


        #region Properties

        [MyBrowsable, Category("Function")]
        [YAXSerializableField(DefaultValue = 1)]
        public int Power { get; set; }

        [MyBrowsable, Category("Function")]
        [YAXSerializableField(DefaultValue = CombinationMode.PermutationOneCycle)]
        public CombinationMode Mode { get; set; }

        #endregion

        #region MyNode overrides

        public override void Validate(MyValidator validator)
        {
            if (Mode == CombinationMode.MakePower)
                base.Validate(validator);
        }

        public override void UpdateMemoryBlocks()
        {
            Output.Count = SymbolSize;
            Output.ColumnHint = ColumnHint;
        }

        public override string Description
        {
            get
            {
                string desc = "";

                switch (Mode)
                {
                    case CombinationMode.PermutationOneCycle:
                        desc = "\u03c0\u2093: " + InitMemTask.CodeVector;
                        break;

                    case CombinationMode.PermutationRandomCycles:
                        desc = "\u03a0\u2093: " + InitMemTask.CodeVector;
                        break;

                    case CombinationMode.MakePower:
                        desc = "\u03c0 \u21e8 \u03c0";
                        break;

                    case CombinationMode.CombinationUnique:
                    case CombinationMode.CombinationRandom:
                        desc = "[" + Min + "," + Max + ")";
                        break;
                }

                switch (Mode)
                {
                    case CombinationMode.PermutationOneCycle:
                    case CombinationMode.PermutationRandomCycles:
                    case CombinationMode.MakePower:
                        return desc + GetPowerString(Power);
                    case CombinationMode.CombinationRandom:
                    case CombinationMode.CombinationUnique:
                        return desc + GetPowerString(SymbolSize);
                    default:
                        return desc;
                }
            }
        }

        #endregion

        #region MyRandomPool overrides

        protected override string GlobalVariableName
        {
            get
            {
                string ret = "PERM_VECTOR" + SymbolSize;

                if (Mode == CombinationMode.CombinationUnique || Mode == CombinationMode.CombinationRandom)
                    ret += "_" + Min + "_" + Max;

                return ret;
            }
        }

        protected override int PatternCount
        {
            get { return typeof(MyCodeVector).GetEnumValues().Length; }
        }

        public override int Seed
        {
            get { return 2345; }
        }


        protected override float[] GenerateRandomVectors()
        {
            float[] codeVectors = new float[SymbolSize * PatternCount];
            Random rnd = new Random(Seed);


            int idx = -1;
            while (++idx < SymbolSize)
                codeVectors[idx] = idx;                           // Identity

            --idx;
            while (++idx < 2 * SymbolSize)
                codeVectors[idx] = (idx + 1) % SymbolSize;        // Shift by one to the right

            while (++idx < 3 * SymbolSize)
                codeVectors[idx] = SymbolSize - idx % SymbolSize; // Involution

            switch (Mode)
            {
                case CombinationMode.PermutationOneCycle:
                    for (int i = 3; i < PatternCount; i++)
                        ShuffleSattolo(new ArraySegment<float>(codeVectors, i * SymbolSize, SymbolSize), rnd);
                    break;

                case CombinationMode.PermutationRandomCycles:
                    for (int i = 3; i < PatternCount; i++)
                        ShuffleFisherYates(new ArraySegment<float>(codeVectors, i * SymbolSize, SymbolSize), rnd);
                    break;

                case CombinationMode.CombinationUnique:
                    var hs = new HashSet<float>();

                    for (int i = 3; i < PatternCount; i++)
                        GenerateCombinationUnique(new ArraySegment<float>(codeVectors, i * SymbolSize, SymbolSize), hs, Min, Max, rnd);
                    break;

                case CombinationMode.CombinationRandom:
                    for (int i = 3; i < PatternCount; i++)
                        GenerateCombination(new ArraySegment<float>(codeVectors, i * SymbolSize, SymbolSize), Min, Max, rnd);
                    break;
            }


            return codeVectors;
        }

        #endregion

        #region Static methods

        /// <summary>
        /// Returns a superscript string representing the parameter.
        /// </summary>
        public static string GetPowerString(int power)
        {
            string pow = string.Empty;

            if (power == 0)
                pow = "\u2070";
            else if (power < 0)
                pow = "\u207b";

            return pow + GetPowerStringInternal(Math.Abs(power));
        }

        static string GetPowerStringInternal(int power)
        {
            if (power == 0)
                return string.Empty;

            string pow = GetPowerStringInternal(power / 10);

            int p = power % 10;

            switch (p)
            {
                case 0:
                    pow += '\u2070';
                    break;
                case 1:
                    pow += '\u00b9';
                    break;
                case 2:
                    pow += '\u00b2';
                    break;
                case 3:
                    pow += '\u00b3';
                    break;
                case 4:
                    pow += '\u2074';
                    break;
                case 5:
                    pow += '\u2075';
                    break;
                case 6:
                    pow += '\u2076';
                    break;
                case 7:
                    pow += '\u2077';
                    break;
                case 8:
                    pow += '\u2078';
                    break;
                case 9:
                    pow += '\u2079';
                    break;
            }

            return pow;
        }

        /// <summary>
        /// Applies the <paramref name="multKernel"/> operation on <paramref name="identity"/> and <paramref name="codeVector"/> <paramref name="power"/>-times.
        /// </summary>
        public static void MakePower(
            CUdeviceptr identity, CUdeviceptr codeVector, CudaDeviceVariable<float> output,
            MyCudaKernel multKernel, int power, int symbolSize)
        {
            if (power == 1)
            {
                output.CopyToDevice(codeVector, 0, 0, sizeof(float) * symbolSize);
                return;
            }


            output.CopyToDevice(identity, 0, 0, sizeof(float) * symbolSize);


            // Multiply identity power-times by codeVector
            var method = power > 0
                ? (int)MyJoin.MyJoinOperation.Permutation
                : (int)MyJoin.MyJoinOperation.Inv_Permutation;

            for (int i = 0; i < Math.Abs(power); i++)
                multKernel.Run(output.DevicePointer, codeVector, output.DevicePointer, method, symbolSize);
        }

        #endregion


        public MyInitMemTask InitMemTask { get; private set; }
        public MyMakePowerTask PerformTask { get; private set; }


        /// <summary>
        ///   Initializes all the memory needed for creating combinations.
        /// </summary>
        [Description("Inititialize memory Task"), MyTaskInfo(OneShot = true)]
        public class MyInitMemTask : MyTask<MyCombinationBook>
        {
            private MyCudaKernel m_kernel;


            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = MyCombinationVector.Identity)]
            public MyCombinationVector CodeVector { get; set; }


            public override void Init(int nGPU)
            {
                if (Owner.Mode != CombinationMode.MakePower)
                {
                    m_kernel = MyKernelFactory.Instance.Kernel(Owner.GPU, @"Common\CombineVectorsKernel", "CombineTwoVectorsKernel");
                    m_kernel.SetupExecution(Owner.SymbolSize);
                }

                MyMemoryManager.Instance.ClearGlobalVariable(Owner.GlobalVariableName, nGPU);
            }

            public override void Execute()
            {
                if (Owner.Mode == CombinationMode.MakePower)
                    return;


                // Get the pointers to the device mem
                var identity = MyMemoryManager.Instance.GetGlobalVariable(Owner.GlobalVariableName, Owner.GPU, Owner.GenerateRandomVectors).DevicePointer;

                var codeVector = identity + (int)CodeVector * Owner.SymbolSize * sizeof(float);

                MakePower(identity, codeVector, Owner.Output.GetDevice(Owner), m_kernel, Owner.Power, Owner.SymbolSize);
            }
        }


        /// <summary>
        ///   Copies combinations to the output
        /// </summary>
        [Description("Perform Task")]
        public class MyMakePowerTask : MyTask<MyCombinationBook>
        {
            private MyCudaKernel m_kernel;


            public override void Init(int nGPU)
            {
                if (Owner.Mode == CombinationMode.MakePower)
                {
                    m_kernel = MyKernelFactory.Instance.Kernel(Owner.GPU, @"Common\CombineVectorsKernel", "CombineTwoVectorsKernel");
                    m_kernel.SetupExecution(Owner.Input.Count);
                }

                MyMemoryManager.Instance.ClearGlobalVariable(Owner.GlobalVariableName, nGPU);
            }

            public override void Execute()
            {
                if (Owner.Mode != CombinationMode.MakePower)
                    return;


                // Get the pointers to the device mem
                var identity = MyMemoryManager.Instance.GetGlobalVariable(Owner.GlobalVariableName, Owner.GPU, Owner.GenerateRandomVectors).DevicePointer;

                MakePower(identity, Owner.Input.GetDevicePtr(Owner.GPU), Owner.Output.GetDevice(Owner), m_kernel, Owner.Power, Owner.SymbolSize);
            }
        }
    }
}
