using BrainSimulator;
using BrainSimulator.Nodes;
using BrainSimulator.Task;
using BrainSimulator.Utils;

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

using nupic.algorithms.Cells4;
using nupic.algorithms.cla_classifier;
using nupic.algorithms.spatial_pooler;
using YAXLib;

namespace BrainSimulator.CLA
{

    // TODO: update parameters provided in Task's parameters (dynamicaly changed) - call the Managed Nupic API

    //// SPATIAL POOLER
    [Description("Spatial Pooler"), MyTaskInfo(OneShot = false)]
    public class MySpTask : MyTask<MyNupicNode>
    {

        [MyBrowsable, Category("C) Column")]
        [YAXSerializableField(DefaultValue = 16u), YAXElementFor("Structure")]
        public uint PotentialRadius { get; set; }

        [MyBrowsable, Category("C) Column")]
        [YAXSerializableField(DefaultValue = 0.5f), YAXElementFor("Structure")]
        public float PotentialPercentage { get; set; }

        [MyBrowsable, Category("B) Region")]
        [YAXSerializableField(DefaultValue = true), YAXElementFor("Structure")]
        public bool LearningEnabled { get; set; }

        [MyBrowsable, Category("B) Region")]
        [YAXSerializableField(DefaultValue = true), YAXElementFor("Structure")]
        public bool GlobalInhibition { get; set; }

        [MyBrowsable, Category("B) Region")]
        [YAXSerializableField(DefaultValue = 10u), YAXElementFor("Structure")]
        public uint NumActiveColumnsPerInhibitionArea { get; set; }

        [MyBrowsable, Category("B) Region")]
        [YAXSerializableField(DefaultValue = -1.0f), YAXElementFor("Structure")]
        public float LocalAreaDensity { get; set; }

        [MyBrowsable, Category("B) Region")]
        [YAXSerializableField(DefaultValue = 0u), YAXElementFor("Structure")]
        public uint StimulusThreshold { get; set; }

        // inhibition radius
        // iteration num
        // iteration learn num
        // wrap around
        // update period
        //syn perm trim threshold

        [MyBrowsable, Category("B) Region")]
        [YAXSerializableField(DefaultValue = 0.1f), YAXElementFor("Structure")]
        public float SynapsePermanenceActiveInc { get; set; }

        [MyBrowsable, Category("B) Region")]
        [YAXSerializableField(DefaultValue = 0.01f), YAXElementFor("Structure")]
        public float SynapsePermanenceInactiveDec { get; set; }

        // syn perm below stimulus inc

        [MyBrowsable, Category("B) Region")]
        [YAXSerializableField(DefaultValue = 0.1f), YAXElementFor("Structure")]
        public float SynapsePermanenceConnected { get; set; }

        [MyBrowsable, Category("B) Region")]
        [YAXSerializableField(DefaultValue = 0.001f), YAXElementFor("Structure")]
        public float MinPercentageOverlapDutyCycles { get; set; }

        [MyBrowsable, Category("B) Region")]
        [YAXSerializableField(DefaultValue = 0.001f), YAXElementFor("Structure")]
        public float MinPercentageActiveDutyCycles { get; set; }

        [MyBrowsable, Category("B) Region")]
        [YAXSerializableField(DefaultValue = 1000u), YAXElementFor("Structure")]
        public uint DutyCyclePeriod { get; set; }

        [MyBrowsable, Category("B) Region")]
        [YAXSerializableField(DefaultValue = 10.0f), YAXElementFor("Structure")]
        public float MaxBoost { get; set; }

        // boost factors
        // overlap duty cycles
        // active duty cycles
        // min overlap duty cycles
        // min active duty cycles


        public override void Init(int nGPU)
        {
            UInt32[] inputDim = new UInt32[] { Owner.InputWidth, Owner.InputHeight };
            UInt32[] colDim = new UInt32[] { Owner.Region2dSideSize, Owner.Region2dSideSize };
            m_sp = new ManagedSpatialPooler(
                inputDim,
                colDim,
                PotentialRadius,
                PotentialPercentage,
                GlobalInhibition,
                LocalAreaDensity,
                NumActiveColumnsPerInhibitionArea,
                StimulusThreshold,
                SynapsePermanenceInactiveDec,
                SynapsePermanenceActiveInc,
                SynapsePermanenceConnected,
                MinPercentageOverlapDutyCycles,
                MinPercentageActiveDutyCycles,
                DutyCyclePeriod,
                MaxBoost,
                Owner.Seed,
                0u,
               true);
            m_input = new uint[Owner.InputSize];
            m_output = new uint[Owner.NumColumns];
            m_inputPtr = GCHandle.Alloc(m_input, GCHandleType.Pinned).AddrOfPinnedObject();
            m_outputPtr = GCHandle.Alloc(m_output, GCHandleType.Pinned).AddrOfPinnedObject();
        }

        public override void Execute()
        {



            if (Owner.Input != null)
            {
                Owner.Input.SafeCopyToHost();
                ConvertArrayToUint(Owner.Input.Host, ref m_input);
                m_sp.compute(m_inputPtr, LearningEnabled, m_outputPtr);

                for (int i = 0; i < Owner.NumColumns; i++)
                {
                    Owner.ActiveColumns.Host[i] = (float)m_output[i];
                }
                Owner.ActiveColumns.SafeCopyToDevice();
            }
        }

        public static void ConvertArrayToUint(float[] from, ref uint[] to)
        {
            for (int i = 0; i < from.Length; i++)
            {
                to[i] = (uint)(from[i]);
            }
        }

        private ManagedSpatialPooler m_sp;
        private uint[] m_input;
        private uint[] m_output;
        private IntPtr m_inputPtr;
        private IntPtr m_outputPtr;
    }

    //// TEMPORAL POOLER
    [Description("Temporal Pooler"), MyTaskInfo(OneShot = false)]
    public class MyTpTask : MyTask<MyNupicNode>
    {
        [MyBrowsable, Category("Temporal Pooler")]
        [YAXSerializableField(DefaultValue = true), YAXElementFor("Structure")]
        public bool LearningEnabled { get; set; }

        [MyBrowsable, Category("Temporal Pooler")]
        [YAXSerializableField(DefaultValue = true), YAXElementFor("Structure")]
        public bool InferenceEnabled { get; set; }


        public override void Init(int nGPU)
        {
            m_tp = new ManagedCells4(
                Owner.NumColumns,
                Owner.CellsPerColumn,
                Owner.ActivationThreshold,
                Owner.MinThreshold,
                Owner.NewSynapseCount,
                Owner.SegmentUpdateValidDuration,
                Owner.PermanenceInitial,
                Owner.PermanenceConnected,
                Owner.PermanenceMax,
                Owner.PermanenceDec,
                Owner.PermanenceInc,
                Owner.GlobalDecay,
                Owner.DoPooling,
                Owner.Seed,
                true,   // INIT_FROM_CPP - when false -> crash
                Owner.CheckSynapseConsistency);

            //m_tp = new ManagedCells4(Owner.COLUMNS, Owner.CELLS_PER_COLUMN,
            //    12, 8, 15, 5, 0.5f, 0.8f, 1.0f, 0.1f, 0.1f, 0.0f, false, 42, true, false);
            m_input = new float[Owner.NumColumns];
            m_output = new float[Owner.NumCells];  // remove the intermediate m_output and use Owner.TpOutput.Host directly?
            m_activeColumnsPtr = GCHandle.Alloc(m_input, GCHandleType.Pinned).AddrOfPinnedObject();
            m_outputPtr = GCHandle.Alloc(m_output, GCHandleType.Pinned).AddrOfPinnedObject();
        }

        public override void Execute()
        {
            if (Owner.ActiveColumns.OnHost)
            {
                Array.Copy(Owner.ActiveColumns.Host, 0, m_input, 0, Owner.NumColumns);

                // ActiveColumns is already on the Host when SpatialPooler
                // task had run, no need to copy it here from the Device
                m_tp.compute(m_activeColumnsPtr, m_outputPtr, InferenceEnabled, LearningEnabled);

                Array.Copy(m_output, 0, Owner.TpOutput.Host, 0, Owner.NumCells);
                Owner.TpOutput.SafeCopyToDevice();
            }
        }

        private ManagedCells4 m_tp;
        private IntPtr m_activeColumnsPtr;
        private IntPtr m_outputPtr;
        private float[] m_output;
        private float[] m_input;
    }

    // CLASSIFIER
    [Description("CLA Classifier"), MyTaskInfo(OneShot = false)]
    public class MyClassifierTask : MyTask<MyNupicNode>
    {

        [MyBrowsable, Category("CLA Classifier")]
        [YAXSerializableField(DefaultValue = true), YAXElementFor("Structure")]
        public bool LearningEnabled { get; set; }

        [MyBrowsable, Category("CLA Classifier")]
        [YAXSerializableField(DefaultValue = true), YAXElementFor("Structure")]
        public bool InferenceEnabled { get; set; }

        // label input can represent category or continuous number
        [MyBrowsable, Category("CLA Classifier")]
        [YAXSerializableField(DefaultValue = false), YAXElementFor("Structure")]
        public bool LabelRepresentsCategory { get; set; }

        [MyBrowsable, Category("CLA Classifier")]
        [YAXSerializableField(DefaultValue = 0), YAXElementFor("Structure")]
        public float BucketsMinValue
        {
            get { return m_bucketsMin; }
            set
            {
                m_bucketsMin = value;
            }
        }

        [MyBrowsable, Category("CLA Classifier")]
        [YAXSerializableField(DefaultValue = 9), YAXElementFor("Structure")]
        public float BucketsMaxValue
        {
            get { return m_bucketsMax; }
            set
            {
                m_bucketsMax = value;
            }
        }

        [MyBrowsable, Category("CLA Classifier")]
        [YAXSerializableField(DefaultValue = 0), YAXElementFor("Structure")]
        public float BucketsStep
        {
            get { return m_bucketsStep; }
            private set { }
        }

        public override void Init(int nGPU)
        {
            m_bucketsStep = (m_bucketsMax - m_bucketsMin) / (Owner.NumBuckets - 1);
            m_recordNum = 0;
            m_classifier = new ManagedFastClaClassifier(Owner.PredictedStepsArray, 0.1, 0.1, 0);
        }

        public override void Execute()
        {
            if (Owner.Label != null)
            {
                Owner.Label.SafeCopyToHost();
                Owner.TpOutput.SafeCopyToHost();

                // get tp output bit indices
                List<uint> tpOutputBitIndices = new List<uint>();
                for (uint i = 0; i < Owner.NumCells; i++)
                {
                    if (Owner.TpOutput.Host[i] > 0.9f)  // detecting ones in float array
                    {
                        tpOutputBitIndices.Add(i);
                    }
                }
                uint[] tpOutputBitIndicesArray = tpOutputBitIndices.ToArray();

                Owner.ActiveCellsIndices.Fill(0);
                Owner.ActiveCellsIndices.SafeCopyToHost();
                for (int i = 0; i < tpOutputBitIndicesArray.Length; i++)
                {
                    Owner.ActiveCellsIndices.Host[i] = (int)tpOutputBitIndicesArray[i];
                }

                uint bucketIndex = 0;
                if (Owner.Label.Host[0] > m_bucketsMin)
                {
                    bucketIndex = (uint)Math.Abs((Owner.Label.Host[0] - m_bucketsMin) / m_bucketsStep);
                }

                m_predictions = m_classifier.fastCompute(m_recordNum, tpOutputBitIndicesArray, bucketIndex,
                Owner.Label.Host[0], LabelRepresentsCategory, LearningEnabled, InferenceEnabled);


                double[] valuesInBuckets;
                m_predictions.TryGetValue(-1, out valuesInBuckets); // -1 is special key for actual values in buckets; TODO: remove the magic const

                int bestBucketIndex = 0;
                double[] probabilities;
                for (int i = 0; i < Owner.PredictedStepsArray.Length; i++)
                {
                    if (m_predictions.TryGetValue(Owner.PredictedStepsArray[i], out probabilities))
                    {
                        double maxProb = 0;
                        for (int j = 0; j < probabilities.Length; j++)
                        {
                            Owner.ClassifierOutput.Host[probabilities.Length * i + j] = (float)probabilities[j];
                            if (probabilities[j] > maxProb)
                            {
                                maxProb = probabilities[j];
                                bestBucketIndex = j;
                            }
                        }

                        if (valuesInBuckets != null)
                        {
                            Owner.ClassifierBestPredictions.Host[i] = (float)valuesInBuckets[bestBucketIndex];
                        }
                        else
                        {
                            Owner.ClassifierBestPredictions.Host[i] = 0.0f;
                        }
                    }
                }

                Owner.ActiveCellsIndices.SafeCopyToDevice();
                Owner.ClassifierOutput.SafeCopyToDevice();
                Owner.ClassifierBestPredictions.SafeCopyToDevice();
            }

            m_recordNum++;
        }

        private ManagedFastClaClassifier m_classifier;
        private SortedDictionary<int, double[]> m_predictions;
        private uint m_recordNum;
        private float m_bucketsStep;
        private float m_bucketsMin;
        private float m_bucketsMax;
    }


}
