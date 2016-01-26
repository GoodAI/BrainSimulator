using GoodAI.Core.Execution;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace GoodAI.Modules.School.Worlds
{

    public class SchoolWorldAdapter : MyWorld //, IMyCustomExecutionPlanner
    {
        [MyInputBlock]
        public MyMemoryBlock<float> ActionInput
        {
            get { return GetInput(0); }
        }

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Visual
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyOutputBlock(1)]
        public MyMemoryBlock<float> Audio
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        [MyOutputBlock(2)]
        public MyMemoryBlock<float> Text
        {
            get { return GetOutput(2); }
            set { SetOutput(2, value); }
        }

        [MyOutputBlock(3)]
        public MyMemoryBlock<float> Data
        {
            get { return GetOutput(3); }
            set { SetOutput(3, value); }
        }

        [MyBrowsable, Category("World Sizes")]
        [YAXSerializableField(DefaultValue = 10000)]
        public int VisualSize { get; set; }

        [MyBrowsable, Category("World Sizes")]
        [YAXSerializableField(DefaultValue = 1000)]
        public int AudioSize { get; set; }

        [MyBrowsable, Category("World Sizes")]
        [YAXSerializableField(DefaultValue = 100)]
        public int TextSize { get; set; }

        [MyBrowsable, Category("World Sizes")]
        [YAXSerializableField(DefaultValue = 200)]
        public int DataSize { get; set; }

        // for serialization
        private string CurrentWorldType
        {
            get
            {
                if (CurrentWorld != null)
                {
                    return CurrentWorldType.ToString();
                }
                else
                {
                    return String.Empty;
                }
            }
            set 
            {
                if (String.IsNullOrEmpty(value))
                {
                    CurrentWorld = null;
                }
                else
                {
                    CurrentWorld = (IWorldAdapter)Type.GetType(value);
                }
            }
        }

        [MyBrowsable, Category("Awsome")]
        [YAXDontSerialize, TypeConverter(typeof(IWorldAdapterConverter))]
        public IWorldAdapter CurrentWorld { get; set; }
        
        public override void UpdateMemoryBlocks()
        {
            Visual.Count = VisualSize;
            Audio.Count = AudioSize;
            Text.Count = TextSize;
            Data.Count = DataSize;
        }

    }
}
