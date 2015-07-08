using GoodAI.Core.Nodes;
using GoodAI.Core.Observers;
using GoodAI.Core.Utils;
using GoodAI.Core.Memory;
using GoodAI.Core.Observers.Helper;
using ManagedCuda;
using ManagedCuda.VectorTypes;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using System.Threading.Tasks;
using YAXLib;
using GoodAI.Core.Task;
using GoodAI.Core;

namespace GoodAI.Core.Observers
{
    //using GoodAI.Core.Observers;

    public class MyTextObserver : MyAbstractMemoryBlockObserver
    {
        [YAXSerializableField]
        protected int m_MaxLines;

        [MyBrowsable, Category("Display"), Description("Maximal number of lines")]
        public int MaxLines
        {
            get
            {
                return m_MaxLines;
            }
            set
            {
                if (value < 0)
                    return;

                m_MaxLines = value;
                TriggerReset();
            }
        }

        protected String m_History;


        private int m_isIntBlock;                
        private bool m_backgroudFillNeeded = false;

        private int m_matrixCols = 0;
        private int m_matrixRows = 0;
        private int m_nbValues = 0;        

        private int m_nbCharacterPerBox;
        private int m_matrixBoxWidthInPixel;
        private int m_matrixBoxHeightInPixel = MyDrawStringHelper.CharacterHeight;

        private CudaDeviceVariable<float> m_characters; // Reference to the characters bitmaps
        private MyCudaKernel m_drawMatrixKernel;
        private MyCudaKernel m_setKernel;

        public MyTextObserver() //constructor with node parameter
        {
            MaxLines = 10;

            m_drawMatrixKernel = MyKernelFactory.Instance.Kernel(@"Observers\DrawMatrixKernel", true);
            m_setKernel = MyKernelFactory.Instance.Kernel(@"Common\SetKernel", true);
            m_drawMatrixKernel.SetConstantVariable("D_CHARACTER_WIDTH", MyDrawStringHelper.CharacterWidth);
            m_drawMatrixKernel.SetConstantVariable("D_CHARACTER_HEIGHT", MyDrawStringHelper.CharacterHeight);
            m_drawMatrixKernel.SetConstantVariable("D_CHARACTER_SIZE", MyDrawStringHelper.CharacterWidth * MyDrawStringHelper.CharacterHeight);
            m_characters = MyMemoryManager.Instance.GetGlobalVariable<float>("CHARACTERS_TEXTURE", MyKernelFactory.Instance.DevCount - 1, MyDrawStringHelper.LoadDigits);

            TargetChanged += MyMatrixObserver_TargetChanged;
        }

        void MyMatrixObserver_TargetChanged(object sender, PropertyChangedEventArgs e)
        {
            if (Target == null)
                return;

            Type type = Target.GetType().GenericTypeArguments[0];
            m_isIntBlock = type == typeof(Single) ? 0 : 1;            
        }                

        protected override void Execute()
        {            
            if (m_matrixCols * m_matrixRows == 0)
                return;

            if (m_backgroudFillNeeded)
            {
                m_backgroudFillNeeded = false;
                m_drawMatrixKernel.SetConstantVariable("D_CANVAS", VBODevicePointer);
                m_setKernel.Run(VBODevicePointer, 0, 0xFFFFFFFF, TextureWidth * TextureHeight);                
            }

            m_drawMatrixKernel.Run(m_isIntBlock);
        }

        protected override void Reset()
        {
            base.Reset();
            /*
            m_nbCharacterPerBox = (1 + 1 + (NbDecimals > 0 ? (1 + NbDecimals) : 0) + 1 + 3);
            m_matrixBoxWidthInPixel = (m_nbCharacterPerBox + 1) * MyDrawStringHelper.CharacterWidth;

            if (m_matrixCols > 0)
            {
                m_matrixRows = (int)Math.Ceiling((float)(m_nbValues) / m_matrixCols);
                m_xStart = 0;
                m_yStart = 0;
                m_xLength = 0;
                m_yLength = 0;
            }
            else
            {
                m_nbValues = Target.Count;
                if (Target.ColumnHint > 0)
                {
                    m_matrixCols = Math.Min(Target.Count, Target.ColumnHint);
                    m_matrixRows = (int)Math.Ceiling((float)(m_nbValues) / m_matrixCols);
                }
                else
                {
                    m_matrixCols = m_nbValues;
                    m_matrixRows = 1;
                }
            }

            if (m_xLength == 0)
                m_xLength = m_matrixCols;
            if (m_yLength == 0)
                m_yLength = m_matrixRows;

            TextureWidth = m_xLength * m_matrixBoxWidthInPixel;
            TextureHeight = m_yLength * m_matrixBoxHeightInPixel;

            m_drawMatrixKernel.SetConstantVariable("D_TEXTURE_WIDTH", TextureWidth);
            m_drawMatrixKernel.SetConstantVariable("D_TEXTURE_HEIGHT", TextureHeight);
            m_drawMatrixKernel.SetConstantVariable("D_MATRIX_COLS", m_matrixCols);
            m_drawMatrixKernel.SetConstantVariable("D_MATRIX_ROWS", m_matrixRows);

            int nbDisplayedValues = m_xLength * m_yLength;

            if (m_yStart + m_yLength >= m_matrixRows)
            {
                // There may be some empty boxes that we dont need to render
                int space = m_nbValues % Math.Max(1, m_matrixCols);
                int inMatrixNbEmptyBoxes = space > 0 ? m_matrixCols - space : 0;
                int inCropNbEmptyBoxes = (m_xStart + m_xLength) - (m_matrixCols - inMatrixNbEmptyBoxes);
                nbDisplayedValues -= inCropNbEmptyBoxes;
            }

            m_drawMatrixKernel.SetConstantVariable("D_NB_VALUES", nbDisplayedValues);
            m_drawMatrixKernel.SetConstantVariable("D_START_COL", m_xStart);
            m_drawMatrixKernel.SetConstantVariable("D_LENGTH_COL", m_xLength);
            m_drawMatrixKernel.SetConstantVariable("D_START_ROW", m_yStart);
            m_drawMatrixKernel.SetConstantVariable("D_LENGTH_ROW", m_yLength);

            m_drawMatrixKernel.SetConstantVariable("D_CHARACTER_MAP", m_characters.DevicePointer);
            m_drawMatrixKernel.SetConstantVariable("D_CHARACTER_MAP_NB_CHARS", MyDrawStringHelper.CharacterMapNbChars);
            m_drawMatrixKernel.SetupExecution(TextureWidth * TextureHeight);

            m_drawMatrixKernel.SetConstantVariable("D_VALUES_INTEGER", Target.GetDevicePtr(this));
            m_drawMatrixKernel.SetConstantVariable("D_VALUES_FLOAT", Target.GetDevicePtr(this));
            m_drawMatrixKernel.SetConstantVariable("D_NB_CHARACTER_PER_BOX", m_nbCharacterPerBox);
            m_drawMatrixKernel.SetConstantVariable("D_NB_DECIMALS", NbDecimals);
            m_setKernel.SetupExecution(TextureWidth * TextureHeight);*/

            m_backgroudFillNeeded = true;
        }
    }
}
