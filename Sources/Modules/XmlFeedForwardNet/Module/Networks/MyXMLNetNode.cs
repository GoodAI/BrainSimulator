using XmlFeedForwardNet.Tasks;
using  XmlFeedForwardNet.Utils;
using BrainSimulator.Memory;
using BrainSimulator.Utils;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;
using System.Windows.Forms.Design;
using System.Drawing.Design;

namespace  XmlFeedForwardNet.Networks
{
    public class MyXMLNetNode : MyAbstractFeedForwardNode
    {
        [YAXSerializableField(DefaultValue = ""), YAXCustomSerializer(typeof(MyPathSerializer))]
        [MyBrowsable, Category("\tBuild"), Description("XML file describing the architecture of the network")]
        [Editor(typeof(FileNameEditor), typeof(UITypeEditor))]
        public string BuildFile { get; set; }

        private FileInfo m_buildLastFile;

        public int XMLColumnHint = 0;

        public int FeatureLayerPosition { get; set; }

        private MyMemoryBlock<float> lastDataInput = null;

        /****************************
        *           INPUTS
        ****************************/

        [MyInputBlock(0)]
        public MyMemoryBlock<float> DataInput
        {
            get { return GetInput(0); }
        }

        /****************************
        *          OUTPUTS
        ****************************/

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        /****************************
        *          METHODS
        ****************************/

        public CUdeviceptr CurrentSampleOutputPtr
        {
            get
            {
                if (Output != null && LastLayer != null)
                    return Output.GetDevicePtr(this, (int)(LastLayer.Output.Count * m_currentSamplePosition));
                else
                    return new CUdeviceptr(0);
            }
        }

        protected override void updateInput()
        {
            base.updateInput();
            FileInfo fi = new FileInfo(BuildFile);
            if (DataInput != lastDataInput)
            {
                lastDataInput = DataInput;
                ParamsChanged = true;
            }
            if (m_buildLastFile == null || m_buildLastFile.LastWriteTime != fi.LastWriteTime || !fi.FullName.Equals(m_buildLastFile.FullName))
            {
                m_buildLastFile = fi;
                ParamsChanged = true;
            }
        }

        protected override void Build()
        {
            base.Build();

            MyXMLNetworkBuilder builder = new MyXMLNetworkBuilder(this);
            builder.Build(BuildFile);

        }

        protected override void Dimension()
        {
            FeatureLayerPosition = -1;

            base.Dimension();

            Output.Count = LastLayer.Output.Count * ForwardSamplesPerStep;

            ParamsChanged = false;
        }

        public override void UpdateMemoryBlocks()
        {
            try
            {
                base.UpdateMemoryBlocks();
            }
            catch (MyXMLNetworkBuilder.MyXMLBuilderException e)
            {
                ParamsChanged = false;
                MyLog.ERROR.WriteLine(e.Message);
            }
        }
    }
}
