using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.SoundProcessing;
using GoodAI.Modules.SoundProcessing.Features;
using ManagedCuda;
using ManagedCuda.CudaFFT;
using ManagedCuda.VectorTypes;
using System;
using System.ComponentModel;
using System.Drawing.Design;
using System.IO;
using System.Windows.Forms;
using System.Windows.Forms.Design;
using YAXLib;

namespace GoodAI.SoundWorld
{
    /// <author>GoodAI</author>
    /// <meta>mv</meta>
    /// <status>Working</status>
    /// <summary>Provides default or custom dataset audio input in various feature types.</summary>
    /// <description></description>
    public class MySoundWorld : MyWorld
    {
        public enum InputTypeEnum 
        {
            SampleSound,        // Sample sound example for showcase
            UserDefined         // User defined set of audio files
        }
        
        public enum FeatureType 
        {
            Samples,            // Plain samples - suitable for audio file visualisation
            FFT,                // Frequency spectrum computed by Fast Fourier transformation
            MFCC,               // Mel-frequency cepstral coeficients (most common feature for speech recognition)
            LPC,                // Linear predictive coeficients (common feature for speech recognition)
        }
        
        
        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Features
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyOutputBlock(1)]
        public MyMemoryBlock<float> Label
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        [YAXSerializableField]
        protected string m_InputPathAudio;
        [YAXSerializableField]
        protected string m_InputPathCorpus;
        [YAXSerializableField]
        protected FeatureType m_FeaturesType;
        [YAXSerializableField]
        protected InputTypeEnum m_UserInput;

        #region I/O
        [Description("Path to input audio file")]
        [YAXSerializableField(DefaultValue = ""), YAXCustomSerializer(typeof(MyPathSerializer))]
        [MyBrowsable, Category("I/O"), EditorAttribute(typeof(FileNameEditor), typeof(UITypeEditor))]
        public string UserDefinedAudio { 
            get{ return m_InputPathAudio; }
            set
            {
                if (value != "" && Path.GetExtension(value) != ".wav")
                {
                    MyLog.ERROR.WriteLine("Not supported file type!");
                    return;
                }
                
                // if single selected, set inputType to User defined
                m_UserInput = InputTypeEnum.UserDefined;
                // andreset previous corpus selections
                m_InputPathCorpus = "";

                m_InputPathAudio = value;
            }
        }

        [Description("Path to input corpus directory")]
        [YAXSerializableField(DefaultValue = ""), YAXCustomSerializer(typeof(MyPathSerializer))]
        [MyBrowsable, Category("I/O"), EditorAttribute(typeof(FolderNameEditor), typeof(UITypeEditor))]
        public string InputPathCorpus
        {
            get { return m_InputPathCorpus; }
            set
            {
                m_InputPathCorpus = value;
                // andreset previous corpus selections
                m_InputPathAudio = "";
                // if single selected, set inputType to User defined
                m_UserInput = InputTypeEnum.UserDefined;
            }
        }

        [Description("Type of input")]
        [MyBrowsable, Category("I/O")]
        [YAXSerializableField(DefaultValue = InputTypeEnum.SampleSound), YAXElementFor("IO")]
        public InputTypeEnum InputType
        {
            get { return m_UserInput; }
            set
            {
                switch (value)
                {
                    case InputTypeEnum.SampleSound:
                        UserDefinedAudio = "";
                        InputPathCorpus = "";

                        m_UserInput = InputTypeEnum.SampleSound;
                    break;
                    case InputTypeEnum.UserDefined:
                        if (m_InputPathAudio != "" || m_InputPathCorpus != "")
                            m_UserInput = InputTypeEnum.UserDefined;
                        else
                        {
                            MyLog.INFO.WriteLine("First select path to custom audio file/s.");
                            m_UserInput = InputTypeEnum.SampleSound;
                        }
                    break;
                }
            }
        }
        #endregion

        #region Features
        [Description("Number of features to extract")]
        [MyBrowsable, Category("Features")]
        [YAXSerializableField(DefaultValue = 4096), YAXElementFor("Features")]
        public int FeaturesCount  { get; set; }

        [Description("Type of features")]
        [MyBrowsable, Category("Features")]
        [YAXSerializableField(DefaultValue = FeatureType.Samples), YAXElementFor("Features")]
        public FeatureType FeaturesType 
        {
            get { return m_FeaturesType; }
            set
            {
                m_FeaturesType = value;
                switch (value)
                {
                    case FeatureType.Samples:
                        FeaturesCount = 1;
                        break;
                    case FeatureType.FFT:
                        FeaturesCount = 256;
                        break;
                    case FeatureType.MFCC:
                        FeaturesCount = 13;
                        break;
                    case FeatureType.LPC:
                        FeaturesCount = 10;
                        break;
                }
            }
        }
        #endregion


        public MyCUDAGenerateInputTask GenerateInput { get; private set; }

        public override void UpdateMemoryBlocks()
        {
            Features.Count = FeaturesCount;
            Label.Count = '~' - ' ' + 1;
        }

        /// <summary>Reads a sound input from SampleSound or custom source defined by user.</summary>
        [Description("Read sound inputs")]
        public class MyCUDAGenerateInputTask : MyTask<MySoundWorld>
        {
            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 1)]
            public int ExpositionTime { get; set; }

            private WavPlayer m_wavReader;
            private int m_currentCorpusFile = 0;
            
            string[] audio;
            Stream m_stream;
            WavPlayer player;
            private short[] m_InputData;
            private long m_position = 0;
            

            public override void Init(Int32 nGPU)
            {
                // init wavPlayer with memory stream
                m_stream = new MemoryStream();
                player = new WavPlayer(m_stream);
            }

            public override void Execute()
            {
                if (SimulationStep == 0)
                {
                    #region First step init
                    m_position = 0;
                    Owner.Features.Fill(0);
                    Owner.Label.Fill(0);

                    try
                    {   // load input data on simulation start
                        switch (Owner.m_UserInput)
                        {
                            case InputTypeEnum.SampleSound:
                                m_wavReader = new WavPlayer(GoodAI.SoundWorld.Properties.Resources.Digits_en_wav);
                                m_wavReader.m_SamplesPerSec = 32000;
                                m_wavReader.AttachTranscription(GoodAI.SoundWorld.Properties.Resources.Digits_en_txt);
                                
                                m_InputData = m_wavReader.ReadShort(m_wavReader.m_length);
                                break;
                            case InputTypeEnum.UserDefined:
                                // reading corpus files
                                if (Owner.m_InputPathCorpus != "")
                                {
                                    audio = Directory.GetFiles(Owner.m_InputPathCorpus, "*.wav");

                                    m_wavReader = new WavPlayer(audio[m_currentCorpusFile]);
                                    AttachTranscriptFileIfExists(audio[m_currentCorpusFile]);
                                    m_currentCorpusFile = 1;
                                    m_InputData = m_wavReader.ReadShort(m_wavReader.m_length);
                                }
                                else
                                {
                                    m_wavReader = new WavPlayer(Owner.m_InputPathAudio);
                                    AttachTranscriptFileIfExists(Owner.m_InputPathAudio);
                                    m_InputData = m_wavReader.ReadShort(m_wavReader.m_length);
                                }
                                
                                break;
                        }
                    }
                    catch (Exception e)
                    {
                        MyLog.ERROR.WriteLine("Not a valid sound device!");
                    }
                    #endregion
                }

                if (SimulationStep % ExpositionTime == 0)
                {
                    #region Every time step
                    if (m_InputData == null)
                        return;

                    int size = 0;
                    float[] result = new float[Owner.FeaturesCount];

                    // process data according to chosen feature type
                    switch (Owner.FeaturesType)
                    {
                        case FeatureType.Samples:
                            result = PrepareInputs(Owner.FeaturesCount);
                            break;
                        case FeatureType.FFT:
                            // input size must be power of 2 and double sized due to the mirror nature of FFT
                            size = NextPowerOf2(Owner.FeaturesCount * 2);
                            result = PerformFFT(PrepareInputs(size));
                            //result = PerformFFT(GenerateSine(size));  // generate a test sine signal
                            break;
                        case FeatureType.MFCC:
                            result = WindowFunction.Hanning(PrepareInputs(256));
                            result = PerformFFT(result);
                            result = MFCC.Compute(result, player.m_SamplesPerSec, Owner.FeaturesCount);
                            break;
                        case FeatureType.LPC:
                            result = WindowFunction.Hanning(PrepareInputs(256));
                            result = LPC.Compute(result, Owner.FeaturesCount);
                            break;
                    }
                    #endregion

                    // flush processed features into GPU
                    Array.Clear(Owner.Features.Host, 0, Owner.Features.Count);
                    for (int i = 0; i < Owner.FeaturesCount; i++)
                        Owner.Features.Host[i] = result[i];
                    Owner.Features.SafeCopyToDevice();
                }
            }

            // prepare batch for processing
            private float[] PrepareInputs(int count)
            {
                // define overlap
                if (m_position >= count)
                    m_position -= (int)(float)(count * 0.5);

                // Set Label
                if (m_wavReader != null && m_wavReader.HasTranscription)
                {
                    char c = m_wavReader.GetTranscription((int)m_position, (int)m_position + count);
                    int index = StringToDigitIndexes(c);

                    Array.Clear(Owner.Label.Host, 0, Owner.Label.Count);

                    // if unknown character, continue without setting any connection
                    Owner.Label.Host[index] = 1.00f;
                    Owner.Label.SafeCopyToDevice();
                }

                // if input is corpus, cycle files in the set
                float[] result = new float[count];
                if (Owner.InputType == InputTypeEnum.UserDefined && Owner.m_InputPathCorpus != "")
                {
                    bool eof = (m_position + count < m_InputData.Length)?false: true;
                    for (int i = 0; i < count; i++)
                    {
                        result[i] = (float)m_InputData[m_position];
                        m_position = ++m_position % m_InputData.Length;
                    }

                    if (eof)
                    {
                        m_position = 0;
                        m_wavReader = new WavPlayer(audio[m_currentCorpusFile]);
                        AttachTranscriptFileIfExists(audio[m_currentCorpusFile]);
                        if (m_currentCorpusFile + 1 < audio.Length)
                            m_currentCorpusFile++;
                        else
                            m_currentCorpusFile = 0;
                        m_InputData = m_wavReader.ReadShort(m_wavReader.m_length);
                    }
                }
                else // if input is single audio, cycle the file itself
                {
                    for (int i = 0; i < count; i++)
                    {
                        result[i] = (float)m_InputData[m_position];
                        m_position = ++m_position % m_InputData.Length;
                    }
                }

                return result;
            }

            private int StringToDigitIndexes(char str)
            {
                int res = 0;
                int charValue = str;
                if (charValue >= ' ' && charValue <= '~')
                    res = charValue - ' ';
                else
                {
                    if (charValue == '\n')
                        res = '~' - ' ' + 1;
                }
                return res;
            }

            // perform Fast Fourier transform
            private float[] PerformFFT(float[] input)
            {
                int size_real = input.Length;
                int size_complex = (int)Math.Floor(size_real / 2.0) + 1;

                CudaFFTPlanMany fftPlan = new CudaFFTPlanMany(1, new int[] { size_real }, 1, cufftType.R2C);
                
                // size of d_data must be padded for inplace R2C transforms: size_complex * 2 and not size_real
                CudaDeviceVariable<float> fft = new CudaDeviceVariable<float>(size_complex * 2);
                
                // device allocation and host have different sizes, why the amount of data must be given explicitly for copying:
                fft.CopyToDevice(input, 0, 0, size_real * sizeof(float));

                // executa plan
                fftPlan.Exec(fft.DevicePointer, TransformDirection.Forward);

                // output to host as float2
                float2[] output = new float2[size_complex];
                fft.CopyToHost(output);

                // cleanup
                fft.Dispose();
                fftPlan.Dispose();

                // squared magnitude of complex output as a result
                float[] result = new float[output.Length];
                for (int i = 0; i < output.Length; i++)
                    result[i] = (float)Math.Sqrt((output[i].x * output[i].x) + (output[i].y * output[i].y));

                return result;
            }

            private void AttachTranscriptFileIfExists(string audioPath)
            {
                string transcrPath = audioPath.Replace("wav", "txt");
                if (File.Exists(transcrPath))
                    m_wavReader.AttachTranscriptionFile(transcrPath);
            }

            #region Helper methods
            private int NextPowerOf2(int n)
            {
                n--;
                n |= (n >> 16);
                n |= (n >> 8);
                n |= (n >> 4);
                n |= (n >> 2);
                n |= (n >> 1);
                ++n;
                return n;
            }

            private float[] GenerateSine(int size)
            {
                int sampleRate = size;
                float[] buffer = new float[size];
                double amplitude = 0.5 * short.MaxValue;
                double frequency = 1000;
                for (int n = 0; n < buffer.Length; n++)
                    buffer[n] = (float)(amplitude * Math.Sin((2 * Math.PI * n * frequency) / sampleRate));

                return buffer;
            }
            #endregion
        }
    }

    /// <summary>Widow functions for low pass filtering of sound input.</summary>
    public static class WindowFunction
    {
        public static float[] Hanning(float[] signal)
        {
            int frame_size = signal.Length;
            float[] window = new float[frame_size];
            for (int n = 0; n < frame_size; n++)
            {
                window[n] = 0.5f * (float)(1 - Math.Cos(2 * Math.PI * n / (frame_size - 1)));
                window[n] *= signal[n];
            }
            return window;
        }

        public static double[] Hamming(float[] signal)
        {
            int frame_size = signal.Length;
            double[] window = new double[frame_size];
            for (int n = 0; n < frame_size; n++)
            {
                window[n] = 0.54 - 0.46 * Math.Cos(2 * Math.PI * n / (frame_size - 1));
                window[n] *= signal[n];
            }
            return window;
        }

        public static double[] Square(float[] signal)
        {
            int frame_size = signal.Length;
            double[] window = new double[frame_size];
            for (int n = 0; n < frame_size; n++)
            {
                if (0 <= n & n <= frame_size - 1)
                    window[n] = signal[n];
                else
                    window[n] = 0;
            }
            return window;
        }
    }
}
