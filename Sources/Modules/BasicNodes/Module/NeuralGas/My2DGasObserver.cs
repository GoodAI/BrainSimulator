using GoodAI.Core;
using GoodAI.Core.Observers;
using GoodAI.Core.Utils;
using System.ComponentModel;
using System.Drawing;
using YAXLib;

namespace GoodAI.Modules.NeuralGas
{
    public class My2DGasObserver : MyNodeObserver<MyGrowingNeuralGasNode>
    {
        [MyBrowsable, Category("Canvas")]
        [YAXSerializableField(DefaultValue = 100), YAXElementFor("Structure")]
        public int X_PIXELS 
        {
            get
            {
                return TextureWidth;
            }
            set
            {
                TextureWidth = value;
                TriggerReset();
            }
        }

        [MyBrowsable, Category("Canvas")]
        [YAXSerializableField(DefaultValue = 100), YAXElementFor("Structure")]
        public int Y_PIXELS
        {
            get
            {
                return TextureHeight;
            }
            set
            {
                TextureHeight = value;
                TriggerReset();
            }
        }

        [MyBrowsable, Category("Data")]
        [YAXSerializableField(DefaultValue = -5.00f), YAXElementFor("Structure")]
        public float X_MIN { get; set; }

        [MyBrowsable, Category("Data")]
        [YAXSerializableField(DefaultValue = 5.00f), YAXElementFor("Structure")]
        public float X_MAX { get; set; }

        [MyBrowsable, Category("Data")]
        [YAXSerializableField(DefaultValue = -5.00f), YAXElementFor("Structure")]
        public float Y_MIN { get; set; }

        [MyBrowsable, Category("Data")]
        [YAXSerializableField(DefaultValue = 5.00f), YAXElementFor("Structure")]
        public float Y_MAX { get; set; }

        [YAXSerializableField]
        private uint MARKER_COLOR = 0xFF00CCFF;
        [YAXSerializableField]
        private uint WINNER_1_COLOR = 0xFFFF0000;
        [YAXSerializableField]
        private uint WINNER_2_COLOR = 0xFF00FF00;
        [YAXSerializableField]
        private uint CONNECTION = 0xFF808080;
        [YAXSerializableField]
        private uint BACKGROUND = 0xFFFFFFFF;

        [MyBrowsable, Category("Control")]
        [YAXSerializableField(DefaultValue = DrawConnections.False), YAXElementFor("Structure")]
        public DrawConnections DRAW_CONNECTIONS { get; set; }

        public enum DrawConnections
        { 
            True,
            False
        }

        [MyBrowsable, Category("\tVisualization")]
        [Description("Color")]
        public Color MarkerColor
        {
            get { return Color.FromArgb((int)MARKER_COLOR); }
            set { MARKER_COLOR = (uint)value.ToArgb(); }
        }

        [MyBrowsable, Category("\tVisualization")]
        [Description("Color")]
        public Color Winner1Color
        {
            get { return Color.FromArgb((int)WINNER_1_COLOR); }
            set { WINNER_1_COLOR = (uint)value.ToArgb(); }
        }

        [MyBrowsable, Category("\tVisualization")]
        [Description("Color")]
        public Color Winner2Color
        {
            get { return Color.FromArgb((int)WINNER_2_COLOR); }
            set { WINNER_2_COLOR = (uint)value.ToArgb(); }
        }

        [MyBrowsable, Category("\tVisualization")]
        [Description("Background color")]
        public Color BackgroundColor
        {
            get { return Color.FromArgb((int)BACKGROUND); }
            set { BACKGROUND = (uint)value.ToArgb(); }
        }
        [MyBrowsable, Category("\tVisualization")]
        [Description("Connection color")]
        public Color ConnectionColor
        {
            get { return Color.FromArgb((int)CONNECTION); }
            set { CONNECTION = (uint)value.ToArgb(); }
        }

        private MyCudaKernel m_clearCanvasKernel;
        private MyCudaKernel m_draw2DGasKernel;
        private MyCudaKernel m_draw2DConnectionKernel;

        public My2DGasObserver()
        {
            X_PIXELS = 100;
            Y_PIXELS = 100;
            X_MIN = -5.00f;
            Y_MIN = -5.00f;
            X_MAX = 5.00f;
            Y_MAX = 5.00f;

            MARKER_COLOR = 0xFF00CCFF;
            WINNER_1_COLOR = 0xFFFF0000;
            WINNER_2_COLOR = 0xFF00FF00;
            BACKGROUND = 0xFFFFFFFF;
            CONNECTION = 0xFF808080;

            m_clearCanvasKernel = MyKernelFactory.Instance.Kernel(@"GrowingNeuralGas\ClearCanvasKernel");
            m_draw2DGasKernel = MyKernelFactory.Instance.Kernel(@"GrowingNeuralGas\Draw2DGasKernel");
            m_draw2DConnectionKernel = MyKernelFactory.Instance.Kernel(@"GrowingNeuralGas\Draw2DConnectionKernel");
        }        

        protected override void Reset()
        {
            TextureHeight = Y_PIXELS;
            TextureWidth = X_PIXELS;
        }

        protected override void Execute()
        {
            m_clearCanvasKernel.SetConstantVariable("D_BACKGROUND",BACKGROUND);
            m_clearCanvasKernel.SetConstantVariable("D_X_PIXELS", X_PIXELS);
            m_clearCanvasKernel.SetConstantVariable("D_Y_PIXELS", Y_PIXELS);

            m_clearCanvasKernel.SetupExecution(X_PIXELS * Y_PIXELS);

            m_clearCanvasKernel.Run(VBODevicePointer);


            if (DRAW_CONNECTIONS == DrawConnections.True)
            {
                m_draw2DConnectionKernel.SetConstantVariable("D_MARKER_COLOR", CONNECTION);
                m_draw2DConnectionKernel.SetConstantVariable("D_X_MIN", X_MIN);
                m_draw2DConnectionKernel.SetConstantVariable("D_X_MAX", X_MAX);
                m_draw2DConnectionKernel.SetConstantVariable("D_Y_MIN", Y_MIN);
                m_draw2DConnectionKernel.SetConstantVariable("D_Y_MAX", Y_MAX);
                m_draw2DConnectionKernel.SetConstantVariable("D_X_PIXELS", X_PIXELS);
                m_draw2DConnectionKernel.SetConstantVariable("D_Y_PIXELS", Y_PIXELS);

                m_draw2DConnectionKernel.SetupExecution(Target.MAX_CELLS * Target.MAX_CELLS);

                m_draw2DConnectionKernel.Run(
                    Target.ConnectionMatrix,
                    Target.ReferenceVector,
                    Target.INPUT_SIZE,
                    Target.MAX_CELLS,
                    VBODevicePointer
                    );
            }


            m_draw2DGasKernel.SetConstantVariable("D_MARKER_COLOR", MARKER_COLOR);
            m_draw2DGasKernel.SetConstantVariable("D_WINNER_1_COLOR", WINNER_1_COLOR);
            m_draw2DGasKernel.SetConstantVariable("D_WINNER_2_COLOR", WINNER_2_COLOR);
            m_draw2DGasKernel.SetConstantVariable("D_X_MIN", X_MIN);
            m_draw2DGasKernel.SetConstantVariable("D_X_MAX", X_MAX);
            m_draw2DGasKernel.SetConstantVariable("D_Y_MIN", Y_MIN);
            m_draw2DGasKernel.SetConstantVariable("D_Y_MAX", Y_MAX);
            m_draw2DGasKernel.SetConstantVariable("D_X_PIXELS", X_PIXELS);
            m_draw2DGasKernel.SetConstantVariable("D_Y_PIXELS", Y_PIXELS);

            m_draw2DGasKernel.SetupExecution(Target.MAX_CELLS);

            m_draw2DGasKernel.Run(
                Target.s1,
                Target.s2,
                Target.ReferenceVector,
                Target.INPUT_SIZE,
                Target.ActivityFlag,
                Target.MAX_CELLS,
                VBODevicePointer
                );
        }
    }
}
