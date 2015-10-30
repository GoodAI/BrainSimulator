using System;
using System.ComponentModel;
using System.Linq;
using System.Drawing;
using System.Drawing.Imaging;

using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using ManagedCuda;
using YAXLib;

using ManagedCuda.VectorTypes;
using GoodAI.Core.Execution;
using System.Text;

//namespace GoodAI.Modules.Vision
namespace HTSLmodule.Worlds
{
    /// <author>GoodAI</author>
    /// <meta>CireNeikual,jv</meta>
    /// <status>Finished</status>
    /// <summary>Presents given sequence of image frames to the output.</summary>
    /// <description>The dataset can be used either default one or custom before the simulation. Here (compared to the ImageDatasetWorld), 
    /// the dataset can be updated at runtime (path to the images and their count). The only requirement is to preserve image resolution 
    /// chosen before the simulation.
    /// 
    /// <h3>Parameters</h3>
    /// <ul>
    ///     <li> <b>UseCustomDataset:</b> If true, the world will attempt to read dataset by given RootFileName, Digits, Extension and NumFrames.</li>
    ///     <li> <b>NumFrames:</b> Number of frames to be loaded from the dataset (starting from 0) and sequentially presented to output.</li>
    ///     <li> <b>RootFileName:</b> Defines path to the file and base part of the name. Name is composed as follows: [RootFineName][numDigits][Extension]. 
    ///     Where items are numbered from 0.</li>
    ///     <li> <b>Digits:</b> How many digits is in the filename? E.g. for 5 it is "RootFilename_00000.png", "RootFilename_00001.png", etc..</li>
    ///     <li> <b>Extension:</b> E.g. ".png"</li>
    /// </ul>
    /// 
    /// Note that the same parameters are in the Load task. This task can be run once for changing the dataset at runtime.
    /// </description>
    public class MyAnimationPredictionWorld : MyWorld
    {
        #region parameters
        [MyBrowsable, Category("File Size"), YAXSerializableField(DefaultValue = 64)]
        public int ImageWidth
        {
            get { return m_iw; }
            set
            {
                if (value > 0)
                {
                    m_iw = value;
                }
            }
        }
        private int m_iw;

        [MyBrowsable, Category("File Size"), YAXSerializableField(DefaultValue = 64)]
        public int ImageHeight
        {
            get { return m_ih; }
            set
            {
                if (value > 0)
                {
                    m_ih = value;
                }
            }
        }
        private int m_ih;

        [MyBrowsable, Category("File"), YAXSerializableField(DefaultValue = false)]
        public bool UseCustomDataset { get; set; }

        [MyBrowsable, Category("File"), YAXSerializableField(DefaultValue = 18)]
        public int NumFrames
        {
            get { return m_numFrames; }
            set
            {
                if (value > 0)
                {
                    m_numFrames = value;
                }
            }
        }
        private int m_numFrames;

        [MyBrowsable, Category("File"), YAXSerializableField(DefaultValue = "userDefinedPath\\NamePrefix_"),
        Description("Path to files including the name prefix")]
        public String RootFileName { get; set; }

        [MyBrowsable, Category("File"), YAXSerializableField(DefaultValue = ".png")]
        public String Extension { get; set; }

        [MyBrowsable, Category("File"), YAXSerializableField(DefaultValue = 5),
        Description("file name has the following form: namePrefix_[numDigits].png. Files are numbered sequentially from 0.")]
        public int Digits
        {
            get { return m_digits; }
            set
            {
                if (value > 0)
                {
                    m_digits = value;
                }
            }
        }
        private int m_digits;

        #endregion

        private int m_defNumFrames = 17;
        private String m_defRootFileName = MyResources.GetMyAssemblyPath() + "\\" + @"\res\animationpredictionworld\SwitchTest_";
        private int m_defDigits = 5;
        private String m_defExtension = ".png";
        private int m_defImageWidth = 64;
        private int m_defImageHeight = 64;

        private int m_currentFrame;

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Image
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        public Bitmap[] m_bitmaps;

        public MyAnimationPredictionLoadTask AnimationPredictionLoadTask { get; private set; }
        public MyAnimationPredictionPresentTask AnimationPredictionPresentTask { get; private set; }

        public override void UpdateMemoryBlocks()
        {
            Image.ColumnHint = ImageWidth;
            Image.Count = ImageWidth * ImageHeight;
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);

            if (UseCustomDataset)
            {
                try
                {
                    this.m_bitmaps = LoadBitmaps(NumFrames, RootFileName, Digits, Extension);
                }
                catch (ArgumentOutOfRangeException e)
                {
                    validator.AddWarning(this, "Loading default dataset, cause: "+ e.Message);
                    UseDefaultBitmaps();
                }
                catch(IndexOutOfRangeException e)
                {
                    validator.AddWarning(this, "Loading the default dataset, cause: "+e.Message);
                    UseDefaultBitmaps();
                }
                catch (Exception)
                {
                    validator.AddWarning(this, "Loading the default dataset, cause: could not read file(s)");
                    UseDefaultBitmaps();
                }
            }
            else
            {
                UseDefaultBitmaps();
            }
        }

        private void UseDefaultBitmaps()
        {
            NumFrames = m_defNumFrames;
            ImageWidth = m_defImageWidth;
            ImageHeight = m_defImageHeight;
            m_currentFrame = 0;
            this.m_bitmaps = LoadBitmaps(NumFrames, m_defRootFileName, m_defDigits, m_defExtension);
        }

        private Bitmap[] LoadBitmaps(int numFrames, String rootFileName, int digits, String extension)
        {
            Bitmap[] bitmaps = new Bitmap[numFrames];
            String fileName = "";

            if (numFrames >= Math.Pow(10, digits))
            {
                throw new ArgumentOutOfRangeException("Number of frames (" + numFrames + ") will not fit in given number of digits (" + digits + ")!");
            }

            for (int i = 0; i < bitmaps.Length; i++)
            {
                fileName = rootFileName + i.ToString().PadLeft(digits, '0') + extension;
                bitmaps[i] = new Bitmap(fileName);

                if (bitmaps[i].Width != ImageWidth || bitmaps[i].Height != ImageHeight)
                {
                    throw new IndexOutOfRangeException("Incorrect width or height of a given image");
                }
            }
            return bitmaps;
        }

        /// <summary>
        /// Tries to reload the images during the simulation. Old bitmaps are preserved if the attempt is unsuccessful and simulation continues.
        /// If loading is OK, task can be disabled again to increase the speed of simulation.
        /// </summary>
        [Description("Reload images."), MyTaskInfo(Disabled = true)]
        public class MyAnimationPredictionLoadTask : MyTask<MyAnimationPredictionWorld>
        {
            [MyBrowsable, Category("File"), YAXSerializableField(DefaultValue = "userDefinedPath\\NamePrefix_"),
            Description("Path to files with file name prefix")]
            public String RootFileName { get; set; }

            [MyBrowsable, Category("File"), YAXSerializableField(DefaultValue = ".png")]
            public String Extension { get; set; }

            [MyBrowsable, Category("File"), YAXSerializableField(DefaultValue = 18)]
            public int NumFrames
            {
                get { return m_numFrames; }
                set
                {
                    if (value > 0)
                    {
                        m_numFrames = value;
                    }
                }
            }
            private int m_numFrames;

            [MyBrowsable, Category("File"), YAXSerializableField(DefaultValue = 5),
            Description("File name has the following form: namePrefix_[numDigits].png. Files are numbered sequentially from 0.")]
            public int Digits
            {
                get { return m_digits; }
                set
                {
                    if (value > 0)
                    {
                        m_digits = value;
                    }
                }
            }
            private int m_digits;
            
            public override void Init(int nGPU)
            {
            }

            public override void Execute()
            {
                try
                {
                    Owner.m_bitmaps = Owner.LoadBitmaps(NumFrames, RootFileName, Digits, Extension);
                }
                catch (ArgumentOutOfRangeException e)
                {
                    MyLog.WARNING.WriteLine("Reload images Task: leaving defautl dataset, cause: " + e.Message);
                    
                }
                catch (IndexOutOfRangeException e)
                {
                    MyLog.WARNING.WriteLine("Reload images Task: leaving the default dataset, cause: " + e.Message);
                    
                }
                catch (Exception)
                {
                    MyLog.WARNING.WriteLine("Reload images Task: leaving the default dataset, cause: could not read file(s)");    
                }
            }
        }

        /// <summary>
        /// Puts the current bitmap onto the output.
        /// </summary>
        [Description("Show images.")]
        public class MyAnimationPredictionPresentTask : MyTask<MyAnimationPredictionWorld>
        {
            [MyBrowsable, Category("Simulation"), YAXSerializableField(DefaultValue = false),
            Description("Continue from the current frame after restarting the simulation?")]
            public bool StartFromFirstFrame { get; set; }

            float[] image;
            byte[] byteArray;

            public override void Init(int nGPU)
            {
                if (StartFromFirstFrame)
                {
                    Owner.m_currentFrame = 0;
                }
            }

            // Could be optimized so that the images are located at GPU, but the dataset may not potentially fit into the GPU memory.
            public override void Execute()
            {
                if (Owner.m_currentFrame >= Owner.m_bitmaps.Count())
                {
                    Owner.m_currentFrame = 0;
                }

                image = new float[Owner.Image.Count];

                for (int x = 0; x < Owner.ImageWidth; x++)
                    for (int y = 0; y < Owner.ImageHeight; y++)
                    {
                        Color c = Owner.m_bitmaps[Owner.m_currentFrame].GetPixel(x, y);

                        image[x + y * Owner.ImageWidth] = 0.333f * (c.R / 255.0f + c.G / 255.0f + c.B / 255.0f);
                    }

                // Create memory block from data
                byteArray = new byte[image.Length * 4];
                Buffer.BlockCopy(image, 0, byteArray, 0, byteArray.Length);
                Owner.Image.Fill(byteArray);

                Owner.m_currentFrame++;
            }
        }
    }
}
