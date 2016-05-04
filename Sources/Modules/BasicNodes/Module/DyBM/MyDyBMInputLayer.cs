using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using YAXLib;
using System.Linq;
using GoodAI.Modules.DyBM.Tasks;

namespace GoodAI.Modules.DyBM
{
    public class MyQueue : Queue<float>
    {
        public MyQueue(int capacity)
        :base(capacity)
        {
            for (int i = 0; i < capacity; i++)
                this.Enqueue(0);
        }

        public float Head()
        {
            return this.ToArray()[0];
        }

        public float Tail()
        {
            return this.ToArray()[this.Count - 1];
        }

        public int HeadIndex()
        {
            return 0;
        }

        public int TailIndex()
        {
            return this.Count - 1;
        }
    }

    public class MyDyBMInputLayer : MyWorkingNode
    {
        // Inputs and Outputs
        /// <summary>
        /// Input of the Layer
        /// </summary>
        [MyInputBlock(0)]
        public MyMemoryBlock<float> Input 
        { 
            get { return GetInput(0); } 
        }

        /// <summary>
        /// Output of the Layer
        /// </summary>
        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        // Public properties
        /// <summary>
        /// Slope of LTD eligibility trace 
        /// </summary>
        [MyBrowsable, Category("Properties")]
        [YAXSerializableField(DefaultValue = "0.5"), YAXElementFor("Properties")]
        public string Slope_μ 
        {
            get {
                string value = "";

                for (int n = 0; n < slope_μ.Count; n++)
                    value += slope_μ[n].ToString() + " ";

                value = value.Remove(value.Length - 1, 1);
                return value;
                }

            set {
                    slope_μ = (value.Length > 0)
                    ? value.Trim().Split(',', ' ').Select(a => float.Parse(a, CultureInfo.InvariantCulture)).ToList()
                    : null;

                    for (int i = 0; i < slope_μ.Count; i++)
                    {
                        slope_μ[i] = slope_μ[i] <= 0 ? 0.1f : slope_μ[i];
                        slope_μ[i] = slope_μ[i] >= 1 ? 0.9f : slope_μ[i];
                        
                    }
                }
        }
        public List<float> slope_μ;

        /// <summary>
        /// Slope of LTP eligibility trace
        /// </summary>
        [MyBrowsable, Category("Properties")]
        [YAXSerializableField(DefaultValue = "0.5"), YAXElementFor("Properties")]
        public string Slope_λ 
        {
            get
            {
                string value = "";

                for (int n = 0; n < slope_λ.Count; n++)
                    value += slope_λ[n].ToString() + " ";

                value = value.Remove(value.Length - 1, 1);
                return value;
            }

            set
            {
                slope_λ = (value.Length > 0)
                  ? value.Trim().Split(',', ' ').Select(a => float.Parse(a, CultureInfo.InvariantCulture)).ToList()
                  : null;

                for (int i = 0; i < slope_λ.Count; i++)
                {
                    slope_λ[i] = slope_λ[i] <= 0 ? 0.1f : slope_λ[i];
                    slope_λ[i] = slope_λ[i] >= 1 ? 0.9f : slope_λ[i];
                    
                }
            }
        }
        public List<float> slope_λ;

        /// <summary>
        /// Interval of axonal conduction delays. The actual conduction delay of each FIFO queue is sampled from this interval.
        /// </summary>
        [MyBrowsable, Category("Properties")]
        [YAXSerializableField(DefaultValue = 9), YAXElementFor("Properties")]
        public int DelayInterval { get; set; }

        /// <summary>
        /// Initial Learning rate η0 of the Layer
        /// </summary>
        [MyBrowsable, Category("Properties")]
        [YAXSerializableField(DefaultValue = 3.0f), YAXElementFor("Properties")]
        public float MaximumLearningRate { get; set; }

        /// <summary>
        /// Temperature of the Layer
        /// </summary>
        [MyBrowsable, Category("Properties")]
        [YAXSerializableField(DefaultValue = 1.0f), YAXElementFor("Properties")]
        public float Temperature_τ { get; set; }

        /// <summary>
        /// Number of iterations for log-likelihood collection until learning rate update
        /// </summary>
        [MyBrowsable, Category("Properties")]
        [YAXSerializableField(DefaultValue = 3), YAXElementFor("Properties")]
        public int LearningRateCorection_M { get; protected set; }

        // Observable MemoryBlocks
        /// <summary>
        /// One FIFO queue for each synapse between neurons
        /// </summary>
        public MyQueue[] Fifo_x { get; protected set; }
        /// <summary>
        /// Each neurons bias. Large positive bias makes neuron most likely to spike.
        /// </summary>
        public MyMemoryBlock<float> Bias_b { get; protected set; }
        /// <summary>
        /// Energy of each neuron at time t
        /// </summary>
        public MyMemoryBlock<float> Energy_E { get; protected set; }
        /// <summary>
        /// Conduction delay of each synapse
        /// </summary>
        public MyMemoryBlock<float> Delay_d { get; protected set; }
        /// <summary>
        /// One Long-Term Potentiation weight for each synapse
        /// </summary>
        public MyMemoryBlock<float> Weight_u { get; protected set; }
        /// <summary>
        /// One Long-Term Depresion weight for each synapse
        /// </summary>
        public MyMemoryBlock<float> Weight_v { get; protected set; }
        /// <summary>
        /// Long-Term Potentiation eligibility trace
        /// </summary>
        public MyMemoryBlock<float> Trace_α { get; protected set; }
        /// <summary>
        /// Long-Term Depresion eligibility trace, only considering spikes within period of Conduction delay
        /// </summary>
        public MyMemoryBlock<float> Trace_β { get; protected set; }
        /// <summary>
        /// Long-Term Depresion eligibility trace, considering spikes after Conduction delay
        /// </summary>
        public MyMemoryBlock<float> Trace_γ { get; protected set; }
        /// <summary>
        /// FIFO traces for each synapse
        /// </summary>
        public MyMemoryBlock<float> FIFO_trace { get; protected set; }
        /// <summary>
        /// Adjustable learning rate for each parameter θ using AdaGrad.
        /// </summary>
        public MyMemoryBlock<float> LearningRate_η { get; protected set; }
        /// <summary>
        /// Log-likelihood
        /// </summary>
        public MyMemoryBlock<float> LogLikelihood { get; protected set; }
        /// <summary>
        /// Log-likelihood derivate for Learning rate update
        /// </summary>
        public MyMemoryBlock<float> Derivative_Δ { get; protected set; }
        
        // Public parameters
        /// <summary>
        /// Number of neurons within the Layer
        /// </summary>
        public int Neurons { get; protected set; }
        /// <summary>
        /// Number of synapses within the Layer
        /// </summary>
        public int Synapses { get; protected set; }
        /// <summary>
        /// Number of eligibility traces for each synapse based on parameters
        /// </summary>
        public int Traces_K { get; protected set; }
        /// <summary>
        /// Number of eligibility traces for each neuron based on parameters
        /// </summary>
        public int Traces_L { get; protected set; }
        private Random rand = new Random();

        // Tasks
        //public MyDyBMReconstructTask ReconstructTask { get; protected set; }
        public MyDyBMLearningTask LearningTask { get; protected set; }
        

        /// <summary>
        /// Constructor of DyBMInputLayer class
        /// </summary>
        public MyDyBMInputLayer() { }

        public override void UpdateMemoryBlocks()
        {
            if(Input != null)
            {
                // Number of Neurons on simulation start is equal to number of inputs
                Neurons = Input.Count;

                // Fully connected network
                Synapses = Neurons * Neurons;

                // Set num. of outputs equal to num. of neurons
                Output.Count = Neurons;

                // Init numbers of eligibility traces according to parameters
                Traces_K = slope_λ.Count;
                Traces_L = slope_μ.Count;

                // Init Memory blocks
                Bias_b.Count = Neurons;
                Energy_E.Count = Neurons;
                Delay_d.Count = Synapses;
                Weight_u.Count = Synapses * Traces_K;
                Weight_v.Count = Synapses * Traces_L;
                Trace_α.Count = Synapses * Traces_K;
                Trace_β.Count = Synapses * Traces_L;
                Trace_γ.Count = Neurons * Traces_L;
                FIFO_trace.Count = Synapses * DelayInterval;
                LogLikelihood.Count = 1;
                LearningRate_η.Count = 2 + Synapses; // θ -> b,u and all vij
                Derivative_Δ.Count = 2 + Synapses;   // θ -> b,u and all vij

                // Create N FIFO queues and initialize their conduction delays
                Fifo_x = new MyQueue[Synapses];
                for (int n = 0; n < Synapses; n++)
                {
                    int capacity = rand.Next(1, DelayInterval);
                    Fifo_x[n] = new MyQueue(capacity);
                }
            }
        }
    }
}
