using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Text.RegularExpressions;
using GoodAI.Core.Utils;
using GoodAI.Core.Observers.Helper;
using ManagedCuda;
using ManagedCuda.VectorTypes;
using ManagedCuda.BasicTypes;
using GoodAI.Core.Memory;
using GoodAI.Core.Task;
using System.Drawing.Imaging;
using System.Reflection;


namespace GoodAI.Core.Observers.Helper
{
    public class MyDrawStringHelper
    {
        public const int CharacterWidth = 9;
        public const int CharacterHeight = 14;
        public const int CharacterMapNbChars = 95;
        private const int CharacterSize = CharacterWidth * CharacterHeight;        

        public static float[] LoadDigits()
        {                        
            Image charactersTexture = MyResources.GetImage("plot_char.png"); 
                // = Image.FromFile("res/plot_char.png");

            int width = charactersTexture.Width;
            int height = charactersTexture.Height;
            int size = width * height;

            BitmapData bitmapData = new Bitmap(charactersTexture).LockBits(
                new Rectangle(0, 0, charactersTexture.Width, charactersTexture.Height),
                ImageLockMode.ReadOnly, charactersTexture.PixelFormat);

            byte[] bytes = new byte[size * 4];
            float[] alphaValues = new float[size];
            Marshal.Copy(bitmapData.Scan0, bytes, 0, size * 4);                       

            for (int i = 0; i < size; i++)
            {                
                alphaValues[i] = (float)bytes[i * 4 + 3] / 255;
            }

            return alphaValues;
        }

        private static int[] StringToDigitIndexes(string str)
        {
            int[] res = new int[str.Length];

            for (int i = 0; i < str.Length; i++)
            {
                int charValue = str[i];
                if (charValue >= ' ' && charValue <= '~')
                    res[i] = charValue - ' ';
                else
                {
                    if (charValue == 160)
                        res[i] = 0;
                    else
                        throw new Exception("Invalid Character: '" + str[i] + "'");
                }
            }
            return res;
        }

        public static void DrawString(string str, int x, int y, uint bgColor, uint fgColor, CUdeviceptr image, int imageWidth, int imageHeight, int maxStringSize = 20)
        {
            // Crop if the string is too long
            if (str.Length > maxStringSize)
                str = str.Substring(0, maxStringSize);

            if (str.Length > 200)
            {
                //__constant__ int D_DIGIT_INDEXES[200];
                throw new ArgumentException("Hardcoded value in DrawDigitsKernel.cs");
            }

            MyCudaKernel m_drawDigitKernel = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Observers\DrawDigitsKernel");
            CudaDeviceVariable<float> characters = MyMemoryManager.Instance.GetGlobalVariable<float>("CHARACTERS_TEXTURE", MyKernelFactory.Instance.DevCount - 1, LoadDigits);

            MyKernelFactory.Instance.Synchronize();

            m_drawDigitKernel.SetConstantVariable("D_BG_COLOR", bgColor);
            m_drawDigitKernel.SetConstantVariable("D_FG_COLOR", fgColor);
            m_drawDigitKernel.SetConstantVariable("D_IMAGE_WIDTH", imageWidth);
            m_drawDigitKernel.SetConstantVariable("D_IMAGE_HEIGHT", imageHeight);
            m_drawDigitKernel.SetConstantVariable("D_DIGIT_WIDTH", CharacterWidth);
            m_drawDigitKernel.SetConstantVariable("D_DIGIT_SIZE", CharacterSize);
            m_drawDigitKernel.SetConstantVariable("D_DIGITMAP_NBCHARS", CharacterMapNbChars);

            int[] indexes = StringToDigitIndexes(str);
            m_drawDigitKernel.SetConstantVariable("D_DIGIT_INDEXES", indexes);
            m_drawDigitKernel.SetConstantVariable("D_DIGIT_INDEXES_LEN", indexes.Length);

            m_drawDigitKernel.SetupExecution(CharacterSize * indexes.Length);
            m_drawDigitKernel.Run(image, characters.DevicePointer, x, y);
        }
    }
}
