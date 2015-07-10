using AudioLib;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
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
        public enum InputType {Sample, MNIST, Microphone, UserDefined}
        public enum FeatureType {Sample, MFCC, LPC}

        private Player player;
        private Recorder recorder;
        private short[] m_InputData;

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> OutputSamples
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyOutputBlock(1)]
        public MyMemoryBlock<float> OutputFeatures
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }


        [YAXSerializableField]
        protected string m_InputPath;
        [YAXSerializableField]
        protected InputType m_UserInput;

        #region I/O
        [YAXSerializableField(DefaultValue = ""), YAXCustomSerializer(typeof(MyPathSerializer))]
        [MyBrowsable, Category("I/O"), Editor]
        public string UserDefinedPath { get; set; }

        [MyBrowsable, Category("I/O")]
        [YAXSerializableField(DefaultValue = InputType.Sample), YAXElementFor("IO")]
        public InputType UserInput
        {
            get { return m_UserInput; }
            set
            {
                switch (value)
                {
                    case InputType.Sample:
                        player = new Player(BasicNodes.Properties.Resources.Sample, new WaveFormat(44100, 16, 2), -1, 4096);
                        m_InputData = player.ReadShort(player.m_length);
                        break;
                    case InputType.MNIST:
                        throw new NotImplementedException();
                        //break;
                    case InputType.Microphone:
                        recorder = new Recorder(new WaveFormat(32000, 32, 1), -1, 4096);
                        break;
                    case InputType.UserDefined:
                        if (m_InputPath != null)
                        {
                            player = new Player(m_InputPath, -1, 4096);
                            m_InputData = player.ReadShort(player.m_length);
                        }
                        break;
                }
            }
            
        }
        #endregion

        #region Features
        [MyBrowsable, Category("Features")]
        [YAXSerializableField(DefaultValue = 10), YAXElementFor("Features")]
        public int FeaturesCount {get; set; }

        [MyBrowsable, Category("Features")]
        [YAXSerializableField(DefaultValue = InputType.Sample), YAXElementFor("Features")]
        public FeatureType FeaturesType { get; set; }
        #endregion


        public MyCUDAGenerateInputTask GenerateInput { get; private set; }

        public override void UpdateMemoryBlocks()
        {
            OutputSamples.Count = 1; // simple stream of float samples
            OutputFeatures.Count = FeaturesCount;
        }

        [Description("Read text inputs")]
        public class MyCUDAGenerateInputTask : MyTask<SoundWorld>
        {
            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 1)]
            public int ExpositionTime { get; set; }

            public override void Init(Int32 nGPU)
            {
                // do nothing here
            }

            public override void Execute()
            {
                if (Owner.UserDefinedPath.Length > 0)
                {
                    if (SimulationStep == 0)
                    {
                        Owner.OutputSamples.Fill(0);
                    }

                    if (SimulationStep % ExpositionTime == 0)
                    {
                        // convert character into digit index
                        int id = (int)SimulationStep % Owner.m_InputData.Length;
                        float sample = (float)Owner.m_InputData[id];
                        
                        Array.Clear(Owner.OutputSamples.Host, 0, Owner.OutputSamples.Count);
                        Owner.OutputSamples.Host[0] = sample;
                        Owner.OutputSamples.SafeCopyToDevice();
                    }
                }
            }

            public void ExecuteCPU()
            {
                for (int i = 0; i < Owner.UserDefinedPath.Length; i++)
                {
                    char c = Owner.UserDefinedPath[(int)SimulationStep];
                    int index = StringToDigitIndexes(c);

                    Array.Clear(Owner.OutputSamples.Host, 0, Owner.OutputSamples.Count);
                    // if unknown character, continue without setting any connction
                    Owner.OutputSamples.Host[index] = 1.00f;
                }
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
        }
    }
}
