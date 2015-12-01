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
    public class MyBoxPlotObserver : MyNodeObserver<MyBoxPlotNode>
    {
        public enum BoxPlotObserverMinMaxMode { UserDefined, Dynamic }

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = BoxPlotObserverMinMaxMode.UserDefined)]
        public BoxPlotObserverMinMaxMode Mode { get; set; }

        private int RowHintV;
        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 1)]
        public int RowHint
        {
            get
            {
                return RowHintV;
            }
            set
            {
                int oc = Target == null ? 5 : Target.Output.Count;
                if (value < 0 || 1024 <= value) RowHintV = 1;
                else if ((oc / 5) % value != 0) RowHintV = 1;
                else RowHintV = value;
            }
        }

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 0.0f)]
        public float MinValue { get; set; }
        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 1.0f)]
        public float MaxValue { get; set; }


        MyCudaKernel m_kernel_drawBoxPlot;
        MyCudaKernel m_kernel_test;
        public MyBoxPlotObserver()
        {
            // dafault values don't work in observer?
            Mode = BoxPlotObserverMinMaxMode.Dynamic;
            RowHintV = 1;
            MaxValue = 1.0f;
            MinValue = 0.0f;
            HorizontalGapV = 5;
            VerticalGapV = 5;
            m_kernel_drawBoxPlot = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Common\DrawBoxPlotKernel", "DrawBoxPlotKernel");
            m_kernel_test = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Common\DrawBoxPlotKernel", "DrawWhiteKernel");
        }

        private int BoxPlotWidth = 20;
        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 20)]
        public int BoxWidth
        {
            get { return BoxPlotWidth; }
            set 
            {
                if (value < 3) BoxPlotWidth = 3;
                else if (value >= 1024) BoxPlotWidth = 1023;
                else BoxPlotWidth = value;
            } 
        }
        private int BoxPlotHeight = 200;
        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 200)]
        public int BoxHeight
        {
            get { return BoxPlotHeight; }
            set 
            {
                if (value < 5) BoxPlotHeight = 5;
                else if (value >= 1024) BoxPlotHeight = 1023;
                else BoxPlotHeight = value;
            } 
        }

        private int HorizontalGapV = 5;
        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 5)]
        public int HorizontalGap {
            get
            {
                return HorizontalGapV;
            }
            set
            {
                if (value >= 1024) HorizontalGapV = 1023;
                else if (value < 0) HorizontalGapV = 0;
                else HorizontalGapV = value;
            }
        }

        private int VerticalGapV;
        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 5)]
        public int VerticalGap
        {
            get { return VerticalGapV; }
            set
            {
                if (value >= 1024) VerticalGapV = 1023;
                else if (value < 1) VerticalGapV = 1;
                else VerticalGapV = value;
            }
        }

        private CudaDeviceVariable<int> Box;
        private CudaStream[] m_streams;
        private bool firstExec = true;

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
                box[i] = (int)((Target.Output.Host[i] - MinValue) / range * (float)(BoxPlotHeight - 1));
            }

            Box = new CudaDeviceVariable<int>(Target.Output.Count);
            Box.CopyToDevice(box);

            for (int i = 0; i < Target.OutputRowsN; i++)
            {
                int xpos = i / RowHintV;
                int ypos = i % RowHintV;
                drawBoxPlotAtPostion(m_streams[i], i * 5, HorizontalGapV + xpos * (HorizontalGapV + BoxPlotWidth), VerticalGapV + ypos * (VerticalGapV + BoxPlotHeight));
            }
            Target.Output.SafeCopyToDevice();
        }

        private void drawBoxPlotAtPostion(CudaStream s, int BoxIndex, int xBoxOffset, int yBoxOffset)
        {
            m_kernel_drawBoxPlot.SetupExecution(new dim3(BoxPlotHeight, 1, 1), new dim3(BoxPlotWidth, 1, 1));
            m_kernel_drawBoxPlot.RunAsync(s, Box.DevicePointer, BoxIndex, xBoxOffset, yBoxOffset, TextureWidth, TextureHeight, BoxPlotWidth, BoxPlotHeight, VBODevicePointer);
            //m_kernel_drawBoxPlot.Run(Box.DevicePointer, BoxIndex, xBoxOffset, yBoxOffset, TextureWidth, TextureHeight, BoxPlotWidth, BoxPlotHeight, VBODevicePointer);
        }

        protected override void Reset()
        {
            firstExec = true;

            TextureWidth = (Target.OutputRowsN / RowHintV) * (BoxPlotWidth + HorizontalGapV) + HorizontalGapV;
            TextureHeight = RowHintV * (BoxPlotHeight + VerticalGapV) + VerticalGapV;

            Target.Output.SafeCopyToHost();

            m_streams = new CudaStream[Target.OutputRowsN];
            for (int i = 0; i < m_streams.Length; i++)
            {
                m_streams[i] = new CudaStream();
            }
        }
    }
}
