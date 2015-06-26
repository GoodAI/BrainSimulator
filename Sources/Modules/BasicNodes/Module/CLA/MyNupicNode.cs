using BrainSimulator.Memory;
using BrainSimulator.Nodes;
using BrainSimulator.Task;
using BrainSimulator.Utils;
using System;
using System.Collections.Generic;
using System.IO;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;


namespace BrainSimulator.CLA
{        
    /// <author>Josef Strunc</author>
    /// <status>Working</status>
    /// <summary>Node wrapping up the Nupic Core library (https://github.com/numenta/nupic.core)</summary>
    /// <description>Nupic is open source implementation of Numenta's cortical learning algorithm.
    /// This node uses three specific parts of the Nupic Core library (through the wrapper library NupicCoreSoloWrapper.dll):
    /// <ul>
    /// <li>SpatialPooler class,</li>
    /// <li>Cells4 class (temporeal pooler implementation) and</li>
    /// <li>FasfClaClassifier class</li>
    /// </ul>
    /// </description>
    public class MyNupicNode : MyWorkingNode
    {

        [MyBrowsable, Category("\t Load Properties From File")]
        [YAXSerializableField(DefaultValue = "propertiesFile.txt"), YAXElementFor("Structure")]
        [EditorAttribute(typeof(System.Windows.Forms.Design.FileNameEditor), typeof(System.Drawing.Design.UITypeEditor))]
        public string PropertiesFileName { get; set; }

        // ------------------------------- Spatial Pooler Parameters -----------------------------------
        [MyBrowsable, Category("B) Region")]
        [YAXSerializableField(DefaultValue = 32u), YAXElementFor("Structure")]
        public uint Region2dSideSize
        {
            get
            {
                return 0; // m_regionSize;
            }
            set
            {
                m_regionSize = value;
                m_numColumns = m_regionSize * m_regionSize;
                m_numCells = m_numColumns * CellsPerColumn;
            }
        }

        [MyBrowsable, Category("B) Region")]
        [YAXSerializableField(DefaultValue = 1024u), YAXElementFor("Structure")]
        public uint NumColumns
        {
            get
            {
                return m_numColumns;
            }
            private set
            {
                m_numColumns = value;
                // the number of columns should be square of region side size
                m_regionSize = (uint)Math.Sqrt(m_numColumns);
            }
        }

        [MyBrowsable, Category("B) Region")]
        [YAXSerializableField(DefaultValue = 1), YAXElementFor("Structure")]
        public int Seed { get; set; }

        // ------------------------------- Temporal Pooler Specific Parameters -----------------------------------
        [MyBrowsable, Category("C) Temporal Pooler")]
        [YAXSerializableField(DefaultValue = 4u), YAXElementFor("Structure")]
        public uint CellsPerColumn
        {
            get
            {
                return m_numCellsPerColumn;
            }
            set
            {
                m_numCellsPerColumn = value;
                m_numCells = NumColumns * m_numCellsPerColumn;
            }
        }

        [MyBrowsable, Category("B) Region")]
        [YAXSerializableField(DefaultValue = 4096u), YAXElementFor("Structure")]
        public uint NumCells
        {
            get
            {
                return m_numCells;
            }
            private set
            {
                m_numCells = value;
            }
        }

        [MyBrowsable, Category("C) Temporal Pooler")]
        [YAXSerializableField(DefaultValue = 1u), YAXElementFor("Structure")]
        public uint ActivationThreshold { get; set; }

        [MyBrowsable, Category("C) Temporal Pooler")]
        [YAXSerializableField(DefaultValue = 1u), YAXElementFor("Structure")]
        public uint MinThreshold { get; set; }

        [MyBrowsable, Category("C) Temporal Pooler")]
        [YAXSerializableField(DefaultValue = 1u), YAXElementFor("Structure")]
        public uint NewSynapseCount { get; set; }

        [MyBrowsable, Category("C) Temporal Pooler")]
        [YAXSerializableField(DefaultValue = 1u), YAXElementFor("Structure")]
        public uint SegmentUpdateValidDuration { get; set; }

        [MyBrowsable, Category("C) Temporal Pooler")]
        [YAXSerializableField(DefaultValue = 0.5f), YAXElementFor("Structure")]
        public float PermanenceInitial { get; set; }

        [MyBrowsable, Category("C) Temporal Pooler")]
        [YAXSerializableField(DefaultValue = 0.8f), YAXElementFor("Structure")]
        public float PermanenceConnected { get; set; }

        [MyBrowsable, Category("C) Temporal Pooler")]
        [YAXSerializableField(DefaultValue = 1.0f), YAXElementFor("Structure")]
        public float PermanenceMax { get; set; }

        [MyBrowsable, Category("C) Temporal Pooler")]
        [YAXSerializableField(DefaultValue = 0.1f), YAXElementFor("Structure")]
        public float PermanenceDec { get; set; }

        [MyBrowsable, Category("C) Temporal Pooler")]
        [YAXSerializableField(DefaultValue = 0.1f), YAXElementFor("Structure")]
        public float PermanenceInc { get; set; }

        [MyBrowsable, Category("C) Temporal Pooler")]
        [YAXSerializableField(DefaultValue = 0.1f), YAXElementFor("Structure")]
        public float GlobalDecay { get; set; }

        [MyBrowsable, Category("C) Temporal Pooler")]
        [YAXSerializableField(DefaultValue = false), YAXElementFor("Structure")]
        public bool DoPooling { get; set; }

        [MyBrowsable, Category("C) Temporal Pooler")]
        [YAXSerializableField(DefaultValue = false), YAXElementFor("Structure")]
        public bool CheckSynapseConsistency { get; set; }


        // -------------------- classifier properties ------------------------
        [MyBrowsable, Category("C) Classifier")]
        [YAXSerializableField(DefaultValue = 10), YAXElementFor("Structure")]
        public int NumBuckets {
            get { return m_numBuckets; }
            set {
                m_numBuckets = value;
            }
        }

        [MyBrowsable, Category("C) Classifier")]
        [YAXSerializableField(DefaultValue = "1"), YAXElementFor("Structure")]
        public string PredictedStepsList
        {
            get { return m_predictedStepsString; }
            set
            {
                m_predictedStepsString = value;
                if (m_predictedStepsString != null)
                {
                    m_predictedStepsArray = Array.ConvertAll(m_predictedStepsString.Split(','), int.Parse);
                }
            }
        }
        [MyBrowsable, Category("C) Classifier")]
        public int[] PredictedStepsArray
        {
            get { return m_predictedStepsArray; }
            private set {}
        }

        // dependent variables:
        private uint m_numColumns;
        private uint m_regionSize;
        private uint m_numCells;
        private uint m_numCellsPerColumn;
        
        private int m_numBuckets;

        private string m_predictedStepsString;
        private int[] m_predictedStepsArray;

        private String m_previousPropertiesFileName = "";

        //[MyBrowsable, Category("D) Cell")]
        //[YAXSerializableField(DefaultValue = 20), YAXElementFor("Structure")]
        //public int DISTAL_CONNECTIONS_PER_SEGMENT { get; set; }

        //[MyBrowsable, Category("D) Cell")]
        //[YAXSerializableField(DefaultValue = 1), YAXElementFor("Structure")]
        //public int MAX_CONN_PER_SEGMENT { get; private set; }

        // INPUT PARAMETERS ---------------------------------------------------
        [MyBrowsable, Category("A) Input")]
        [YAXSerializableField(DefaultValue = 1u), YAXElementFor("Structure")]
        public uint InputSize { get; private set; }

        [MyBrowsable, Category("A) Input")]
        [YAXSerializableField(DefaultValue = 1u), YAXElementFor("Structure")]
        public uint InputWidth { get; private set; }

        [MyBrowsable, Category("A) Input")]
        [YAXSerializableField(DefaultValue = 1u), YAXElementFor("Structure")]
        public uint InputHeight { get; private set; }


        // INPUT / OUTPUT -----------------------------------------------------
        [MyInputBlock]
        public MyMemoryBlock<float> Input
        {
            get { return GetInput(0); }
        }

        [MyInputBlock]
        public MyMemoryBlock<float> Label
        {
            get { return GetInput(1); }
        }

        [MyOutputBlock]
        public MyMemoryBlock<float> ActiveColumns
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        // TpOutput is OR of active cells with predictive cells
        [MyOutputBlock]
        public MyMemoryBlock<float> TpOutput
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        [MyOutputBlock]
        public MyMemoryBlock<float> ClassifierOutput
        {
            get { return GetOutput(2); }
            set { SetOutput(2, value); }
        }

        [MyOutputBlock]
        public MyMemoryBlock<float> ClassifierBestPredictions
        {
            get { return GetOutput(3); }
            set { SetOutput(3, value); }
        }

        // MEMORY BLOCKS ----------------------------------------------------
        public MyMemoryBlock<int> ActiveCellsIndices { get; private set; }

        // TASKS -----------------------------------------------------------
        public MySpTask SpTask { get; protected set; }
        public MyTpTask TpTask { get; protected set; }
        public MyClassifierTask ClassifierTask { get; protected set; }


        public override void UpdateMemoryBlocks()
        {
            InputSize = Input == null ? 1 : (uint)Input.Count;

            InputWidth = Input == null ? 1 : (uint)Input.ColumnHint;
            InputHeight = (uint)Math.Ceiling((float)InputSize / (float)InputWidth);

            ActiveColumns.Count = (int)NumColumns;
            ActiveColumns.ColumnHint = (int)Region2dSideSize;
            TpOutput.Count = (int)NumColumns * (int)CellsPerColumn;
            TpOutput.ColumnHint = (int)NumColumns;
            ClassifierOutput.Count = NumBuckets * m_predictedStepsArray.Length;
            ClassifierOutput.ColumnHint = NumBuckets;

            ClassifierBestPredictions.Count = PredictedStepsArray.Length;
            ClassifierBestPredictions.ColumnHint = 1;

            ActiveCellsIndices.Count = (int)NumCells;

            // TEMPORAL POOLER

            // STATS
            //Anomaly.Count = 1;
            //InputError.Count = 1;

            // COLUMNS

            // CELLS

            // SEGMENTS

            // CONNECTIONS

            //RandomNumbersOnGPU.Count = SEGMENTS * DISTAL_CONNECTIONS_PER_SEGMENT;

            if (!String.IsNullOrEmpty(PropertiesFileName) && !PropertiesFileName.Equals(m_previousPropertiesFileName))
            {
                LoadProperties(PropertiesFileName);
            }
            m_previousPropertiesFileName = PropertiesFileName;
        }

        private void LoadProperties(String filePath)
        {
            byte[] fileContentBuffer;
            long fileSize;
            try
            {
                fileSize = new FileInfo(filePath).Length;
                fileContentBuffer = new byte[fileSize];

                BinaryReader reader = new BinaryReader(File.OpenRead(filePath));
                reader.Read(fileContentBuffer, 0, (int)fileSize);
                reader.Close();
            }
            catch (Exception e)
            {
                MyLog.WARNING.WriteLine("Loading Nupic properties from file " + filePath + " failed : " + e.Message);
                return;
            }

            if (fileSize > 0)
            {
                int numOfParams = Enum.GetValues(typeof(NupicProperty)).Length;
                Dictionary<NupicProperty, float> propertyValues = new Dictionary<NupicProperty, float>(numOfParams);
                float value;
                string parametersString = System.Text.Encoding.UTF8.GetString(fileContentBuffer);
                int paramValueStart;
                int paramValueLength;
                string parameterName;
                string nextParameterName;

                foreach (NupicProperty p in Enum.GetValues(typeof(NupicProperty)))
                {
                    parameterName = Enum.GetName(typeof(NupicProperty), p);
                    nextParameterName = Enum.GetName(typeof(NupicProperty), p + 1);
                    // index of the first character of value of the parameter; the parameters name is followed by 3 characters "': "
                    paramValueStart = parametersString.IndexOf(parameterName) + parameterName.Length + 3;
                    // index of next comma
                    paramValueLength = parametersString.Substring(paramValueStart, parametersString.Length - paramValueStart).IndexOfAny(new char[] { ',', '}' });
                    value = (float)Convert.ToDouble(parametersString.Substring(paramValueStart, paramValueLength));
                    propertyValues.Add(p, value);
                }


                propertyValues.TryGetValue(NupicProperty.columnCount, out value);
                NumColumns = (uint)value;

                propertyValues.TryGetValue(NupicProperty.maxBoost, out value);
                SpTask.MaxBoost = value;

                propertyValues.TryGetValue(NupicProperty.numActiveColumnsPerInhArea, out value);
                SpTask.NumActiveColumnsPerInhibitionArea = (uint)value;

                propertyValues.TryGetValue(NupicProperty.potentialPct, out value);
                SpTask.PotentialPercentage = value;

                propertyValues.TryGetValue(NupicProperty.seed, out value);
                Seed = (int)value;

                propertyValues.TryGetValue(NupicProperty.synPermActiveInc, out value);
                SpTask.SynapsePermanenceActiveInc = value;

                propertyValues.TryGetValue(NupicProperty.synPermConnected, out value);
                SpTask.SynapsePermanenceConnected = value;

                propertyValues.TryGetValue(NupicProperty.synPermInactiveDec, out value);
                SpTask.SynapsePermanenceInactiveDec = value;

                propertyValues.TryGetValue(NupicProperty.activationThreshold, out value);
                ActivationThreshold = (uint)value;

                propertyValues.TryGetValue(NupicProperty.cellsPerColumn, out value);
                CellsPerColumn = (uint)value;

                propertyValues.TryGetValue(NupicProperty.globalDecay, out value);
                GlobalDecay = value;

                propertyValues.TryGetValue(NupicProperty.initialPerm, out value);
                PermanenceInitial = value;

                //swarmParametersValues.TryGetValue(SwarmParameter.maxAge, out value);
                //swarmParametersValues.TryGetValue(SwarmParameter.maxSegmentsPerCell, out value);
                //swarmParametersValues.TryGetValue(SwarmParameter.maxSynapsesPerSegment, out value);

                propertyValues.TryGetValue(NupicProperty.minThreshold, out value);
                MinThreshold = (uint)value;

                propertyValues.TryGetValue(NupicProperty.newSynapseCount, out value);
                NewSynapseCount = (uint)value;

                //swarmParametersValues.TryGetValue(SwarmParameter.pamLength, out value);

                propertyValues.TryGetValue(NupicProperty.permanenceDec, out value);
                PermanenceDec = value;

                propertyValues.TryGetValue(NupicProperty.permanenceInc, out value);
                PermanenceInc = value;

                MyLog.INFO.WriteLine("The Nupic parameters were succesfully loaded from file " + filePath);
            }
            else
            {
                MyLog.INFO.WriteLine("The file " + filePath + " is empty.");
            }
        }

        private enum NupicProperty
        {
            columnCount,
            //globalInhibition,
            inputWidth,
            maxBoost,
            numActiveColumnsPerInhArea,
            potentialPct,
            seed,
            //spVerbosity,
            //spatialImp,
            synPermActiveInc,
            synPermConnected,
            synPermInactiveDec,
            //tpEnable,
            activationThreshold,
            cellsPerColumn,
            //columnCount,
            globalDecay,
            initialPerm,
            //inputWidth,
            maxAge,
            maxSegmentsPerCell,
            maxSynapsesPerSegment,
            minThreshold,
            newSynapseCount,
            //outputType,
            pamLength,
            permanenceDec,
            permanenceInc
            //seed,
            //temporalImp,
            //verbosity
        }

    }
}
