using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.SoundProcessing.Features;
using ManagedCuda;
using ManagedCuda.CudaFFT;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using YAXLib;

namespace GoodAI.Modules.SoundProcessing
{
    public class SoundWorld : MyWorld
    {
        public enum InputTypeEnum 
        {
            SampleSound,        // Sample sound example for showcase
            Microphone,         // Input from microphone device
            UserDefined         // User defined set of audio files
        }
        
        public enum FeatureType 
        {
            Samples,            // Plain samples - suitable for audio file visualisation
            FFT,                // Frequency spectrum computed by Fast Fourier transformation
            MFCC,               // Mel-frequency cepstral coeficients (most common feature for speech recognition)
            LPC,                // Linear predictive coeficients (common feature for speech recognition)
            CLPC                // Linear predictive cepstral coeficients (common feature for speech recognition)
        }
        
        protected Recorder m_recorder;

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
        protected string m_InputPath;
        [YAXSerializableField]
        protected FeatureType m_FeaturesType;
        [YAXSerializableField]
        protected InputTypeEnum m_UserInput;

        #region I/O
        [Description("Path to input audio files directory")]
        [YAXSerializableField(DefaultValue = ""), YAXCustomSerializer(typeof(MyPathSerializer))]
        [MyBrowsable, Category("I/O"), Editor]
        public string UserDefinedPath { get; set; }

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
                        m_UserInput = InputTypeEnum.SampleSound;
                        break;
                    case InputTypeEnum.Microphone:
                        m_UserInput = InputTypeEnum.Microphone;
                        break;
                    case InputTypeEnum.UserDefined:
                        if (m_InputPath != null)
                            m_UserInput = InputTypeEnum.UserDefined;
                        break;
                }
            }
            
        }

        [Description("Id of microphone device")]
        [MyBrowsable, Category("I/O")]
        [YAXSerializableField(DefaultValue = -1), YAXElementFor("IO")]
        public int MicrophoneDevice { get; set; }

        [Description("Number of seconds for microphone to record")]
        [MyBrowsable, Category("I/O")]
        [YAXSerializableField(DefaultValue = 3), YAXElementFor("IO")]
        public int RecordSeconds { get; set; }
        #endregion

        #region Features
        [Description("Number of features to extract")]
        [MyBrowsable, Category("Features")]
        [YAXSerializableField(DefaultValue = 1), YAXElementFor("Features")]
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
                        FeaturesCount = 12;
                        break;
                    case FeatureType.LPC:
                        FeaturesCount = 10;
                        break;
                    case FeatureType.CLPC:
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
            Label.Count = 1;
        }

        public override void Cleanup()
        {
            base.Cleanup();
            if(m_recorder!= null && m_recorder.is_recording)
                m_recorder.Stop();
        }

        [Description("Read sound inputs")]
        public class MyCUDAGenerateInputTask : MyTask<SoundWorld>
        {
            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 1)]
            public int ExpositionTime { get; set; }

            private WaveReader m_wavReader;
            
            private short[] m_InputData;
            private long m_position = 0;

            public override void Init(Int32 nGPU)
            {
                // do nothing here
            }

            public override void Execute()
            {
                if (SimulationStep == 0)
                {
                    Owner.Features.Fill(0);
                    Owner.Label.Fill(0);

                    try
                    {   // load input data on simulation start
                        switch (Owner.m_UserInput)
                        {
                            case InputTypeEnum.SampleSound:
                                m_wavReader = new WaveReader(BasicNodes.Properties.Resources.Sample, new WaveFormat(44100, 16, 2), -1, Owner.FeaturesCount);
                                m_InputData = m_wavReader.ReadShort(m_wavReader.m_length);
                                break;
                            case InputTypeEnum.Microphone:
                                Owner.m_recorder = new Recorder(new WaveFormat(32000, 16, 2), Owner.MicrophoneDevice, 32000 * Owner.RecordSeconds);
                                Owner.m_recorder.ShortRecording += new ShortRecordingEventHandler(OnRecordShort);
                                Owner.m_recorder.Record();
                                break;
                            case InputTypeEnum.UserDefined:
                                m_wavReader = new WaveReader(Owner.m_InputPath, -1, 4096);
                                m_InputData = m_wavReader.ReadShort(m_wavReader.m_length);
                                break;
                        }
                    }
                    catch (Exception e)
                    {
                        // Not a valid sound device!
                    }
                }

                if (SimulationStep % ExpositionTime == 0)
                {
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
                            WaveFormat m_format = null;
                            if (Owner.m_recorder != null)
                                m_format = Owner.m_recorder.m_format;
                            else if (m_wavReader != null)
                                m_format = m_wavReader.m_format;

                            result = PerformFFT(PrepareInputs(512));
                            result = MFCC.Compute(result, m_format, Owner.FeaturesCount);
                            break;
                        case FeatureType.LPC:
                            result = PrepareInputs(512);
                            result = LPC.Compute(result, Owner.FeaturesCount);
                            break;
                        case FeatureType.CLPC:
                            result = PrepareInputs(512);
                            result = CLPC.Compute(result, Owner.FeaturesCount);
                            break;
                    }

                    // flush processed features into GPU
                    Array.Clear(Owner.Features.Host, 0, Owner.Features.Count);
                    for (int i = 0; i < Owner.FeaturesCount; i++)
                        Owner.Features.Host[i] = result[i];
                    Owner.Features.SafeCopyToDevice();
                }
            }

            public void ExecuteCPU()
            {
                
            }

            
            public void OnRecordShort(short[] input)
            {
                m_InputData = input;
                Owner.m_recorder.Stop();
            }

            // prepare batch for processing
            private float[] PrepareInputs(int count)
            {
                float[] result = new float[count];
                for (int i = 0; i < count; i++)
                {
                    result[i] = (float)m_InputData[m_position];
                    m_position = ++m_position % m_InputData.Length;
                }

                return result;
            }

            // perform Fast Fourier transform
            private float[] PerformFFT(float[] input)
            {
                float[] result = new float[input.Length];
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
                for (int i = 0; i < output.Length; i++)
                    result[i] = (float)Math.Sqrt((output[i].x * output[i].x) + (output[i].y * output[i].y));

                return result;
            }

            #region Helper methods
            private int NextPowerOf2(int n)
            {
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
                double amplitude = 0.25 * short.MaxValue;
                double frequency = 1000;
                for (int n = 0; n < buffer.Length; n++)
                    buffer[n] = (float)(amplitude * Math.Sin((2 * Math.PI * n * frequency) / sampleRate));

                return buffer;
            }
            #endregion
        }
    }
}
