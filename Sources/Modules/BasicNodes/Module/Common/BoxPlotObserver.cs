using GoodAI.Core;
using GoodAI.Core.Observers;
using GoodAI.Core.Utils;
using ManagedCuda;
using ManagedCuda.VectorTypes;
using System.ComponentModel;
using System.Linq;
using YAXLib;

namespace GoodAI.Modules.Common
{
    public class BoxPlotObserver : MyNodeObserver<BoxPlotNode>
    {
        #region input

        public enum BoxPlotObserverMinMaxMode { UserDefined, Dynamic }
        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = BoxPlotObserverMinMaxMode.UserDefined)]
        public BoxPlotObserverMinMaxMode Mode { get; set; }

        private int rowHintV;
        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 1)]
        public int RowHint
        {
            get
            {
                return rowHintV;
            }
            set
            {
                int oc = Target == null ? 5 : Target.Output.Count;
                if (value < 0 || 1024 <= value) rowHintV = 1;
                else if ((oc / 5) % value != 0) rowHintV = 1;
                else rowHintV = value;
            }
        }

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 0.0f)]
        public float MinValue { get; set; }
        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 1.0f)]
        public float MaxValue { get; set; }

        private int boxPlotWidth = 20;
        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 20)]
        public int BoxWidth
        {
            get { return boxPlotWidth; }
            set 
            {
                if (value < 3) boxPlotWidth = 3;
                else if (value >= 1024) boxPlotWidth = 1023;
                else boxPlotWidth = value;
            } 
        }
        private int boxPlotHeight = 200;
        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 200)]
        public int BoxHeight
        {
            get { return boxPlotHeight; }
            set 
            {
                if (value < 5) boxPlotHeight = 5;
                else if (value >= 1024) boxPlotHeight = 1023;
                else boxPlotHeight = value;
            } 
        }

        private int horizontalGapV = 5;
        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 5)]
        public int HorizontalGap {
            get
            {
                return horizontalGapV;
            }
            set
            {
                if (value >= 1024) horizontalGapV = 1023;
                else if (value < 0) horizontalGapV = 0;
                else horizontalGapV = value;
            }
        }

        private int verticalGapV;
        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 5)]
        public int VerticalGap
        {
            get { return verticalGapV; }
            set
            {
                if (value >= 1024) verticalGapV = 1023;
                else if (value < 1) verticalGapV = 1;
                else verticalGapV = value;
            }
        }
        #endregion

        private CudaDeviceVariable<int> m_box;
        private CudaStream[] m_streams;
        private bool firstExec = true;

        MyCudaKernel m_kernel_drawBoxPlot;
        MyCudaKernel m_kernel_test;

        public BoxPlotObserver()
        {
            // dafault values don't work in observer?
            Mode = BoxPlotObserverMinMaxMode.Dynamic;
            rowHintV = 1;
            MaxValue = 1.0f;
            MinValue = 0.0f;
            horizontalGapV = 5;
            verticalGapV = 5;
            m_kernel_drawBoxPlot = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Common\DrawBoxPlotKernel", "DrawBoxPlotKernel");
            m_kernel_test = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Common\DrawBoxPlotKernel", "DrawWhiteKernel");
        }

        protected override void Execute()
        {
            if (firstExec)
            {
                firstExec = false;
                m_kernel_test.SetupExecution(TextureHeight * TextureWidth);
                m_kernel_test.Run(TextureWidth, TextureHeight, VBODevicePointer);
            }

            if (Mode == BoxPlotObserverMinMaxMode.Dynamic)
            {
                MinValue = Target.Output.Host.Min();
                MaxValue = Target.Output.Host.Max();
            }

            float range = MaxValue - MinValue;
            int[] box = new int[Target.Output.Count];
            for (int i = 0; i < Target.Output.Count; i++)
            {
                box[i] = (int)((Target.Output.Host[i] - MinValue) / range * (float)(boxPlotHeight - 1));
            }

            m_box = new CudaDeviceVariable<int>(Target.Output.Count);
            m_box.CopyToDevice(box);

            for (int i = 0; i < Target.OutputRowsN; i++)
            {
                int xpos = i / rowHintV;
                int ypos = i % rowHintV;
                drawBoxPlotAtPostion(m_streams[i], i * 5, horizontalGapV + xpos * (horizontalGapV + boxPlotWidth), verticalGapV + ypos * (verticalGapV + boxPlotHeight));
            }
            Target.Output.SafeCopyToDevice();
        }

        private void drawBoxPlotAtPostion(CudaStream s, int BoxIndex, int xBoxOffset, int yBoxOffset)
        {
            m_kernel_drawBoxPlot.SetupExecution(new dim3(boxPlotHeight, 1, 1), new dim3(boxPlotWidth, 1, 1));
            m_kernel_drawBoxPlot.RunAsync(s, m_box.DevicePointer, BoxIndex, xBoxOffset, yBoxOffset, TextureWidth, TextureHeight, boxPlotWidth, boxPlotHeight, VBODevicePointer);
            //m_kernel_drawBoxPlot.Run(Box.DevicePointer, BoxIndex, xBoxOffset, yBoxOffset, TextureWidth, TextureHeight, BoxPlotWidth, BoxPlotHeight, VBODevicePointer);
        }

        protected override void Reset()
        {
            firstExec = true;

            TextureWidth = (Target.OutputRowsN / rowHintV) * (boxPlotWidth + horizontalGapV) + horizontalGapV;
            TextureHeight = rowHintV * (boxPlotHeight + verticalGapV) + verticalGapV;

            Target.Output.SafeCopyToHost();

            m_streams = new CudaStream[Target.OutputRowsN];
            for (int i = 0; i < m_streams.Length; i++)
            {
                m_streams[i] = new CudaStream();
            }
        }
    }
}
