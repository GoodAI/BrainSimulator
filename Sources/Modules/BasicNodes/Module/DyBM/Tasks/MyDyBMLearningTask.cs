using GoodAI.BasicNodes.DyBM;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.DyBM.Tasks
{
    /// <summary>
    /// Layer learning task
    /// </summary>
    [Description("DyBM Learning"), MyTaskInfo(OneShot = false)]
    public class MyDyBMLearningTask : MyTask<MyDyBMInputLayer>
    {
        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = true), YAXElementFor("Structure")]
        public bool UpdateWeights { get; set; }

        /// <summary>
        /// Initialize all parameters to 0 -> meaning that previous sequences were all 0
        /// </summary>
        /// <param name="nGPU"></param>
        public override void Init(int nGPU)
        {
            if (Owner.Delay_d != null)
            {
                MyRandom rand = new MyRandom();
                Owner.Delay_d.SafeCopyToHost();
                Owner.Bias_b.SafeCopyToHost();
                Owner.Weight_u.SafeCopyToHost();
                Owner.Weight_v.SafeCopyToHost();
                Owner.LearningRate_η.SafeCopyToHost();

                // Generated interval
                for (int s = 0; s < Owner.Neurons; s++)
                {
                    // Randomize bias
                    Owner.Bias_b.Host[s] = (float)rand.NextDouble(0, 0.1f);
                }

                for (int s = 0; s < Owner.Synapses; s++)
                {
                    // Init conduction delays
                    Owner.Delay_d.Host[s] = Owner.Fifo_x[s].Count;
                    // Randomize weights
                    Owner.Weight_u.Host[s] = (float)rand.NextDouble(0, 0.1f);
                    Owner.Weight_v.Host[s] = (float)rand.NextDouble(0, 0.1f);
                }

                for (int i = 0; i < 4; i++)
                {
                    // Init learning rate
                    Owner.LearningRate_η.Host[i] = 1;
                }

                Owner.Delay_d.SafeCopyToDevice();
                Owner.Bias_b.SafeCopyToDevice();
                Owner.Weight_u.SafeCopyToDevice();
                Owner.Weight_v.SafeCopyToDevice();
                Owner.LearningRate_η.SafeCopyToDevice();
            }
        }

        /// <summary>
        /// Execute one learning step of the layer through all neurons
        /// </summary>
        public override void Execute()
        {
            int L, K, M, m;
            float head, tail;
            float sum_α1, sum_β1, sum_γ1, sum_α, sum_β, sum_γ, logPθ;
            float[] αij, βij, γi, γj, uij, vij, vji, η, Δ, μ, λ, Eθ, Pθ;

            int Neurons = Owner.Neurons;
            float τ = Owner.Temperature_τ;
            Pθ = new float[Neurons];

            K = Owner.Traces_K;
            L = Owner.Traces_L;
            M = Owner.LearningRateCorection_M;
            Eθ = Owner.Energy_E.Host;
            μ = Owner.slope_μ.ToArray();
            λ = Owner.slope_λ.ToArray();

            Owner.Output.SafeCopyToHost();
            Owner.Bias_b.SafeCopyToHost();
            Owner.Energy_E.SafeCopyToHost();
            Owner.Weight_u.SafeCopyToHost();
            Owner.Weight_v.SafeCopyToHost();
            Owner.Trace_α.SafeCopyToHost();
            Owner.Trace_β.SafeCopyToHost();
            Owner.Trace_γ.SafeCopyToHost();
            Owner.FIFO_trace.SafeCopyToHost();
            Owner.LogLikelihood.SafeCopyToHost();
            Owner.LearningRate_η.SafeCopyToHost();
            Owner.Derivative_Δ.SafeCopyToHost();

            Owner.LogLikelihood.Host[0] = 0;

            #region Take the input of the Layer
            Owner.Input.SafeCopyToHost();
            // Input of the at time t-1
            float[] x = Owner.Input.Host;
            // Expected input at time t
            float[] X = new float[Neurons];
            #endregion

            #region Feed-Forward pass for all neurons
            logPθ = 0;
            double expE, expP;
            for (int j = 0; j < Neurons; j++)
            {
                #region Compute sums of weighted eligidible traces
                sum_α1 = 0;
                sum_β1 = 0;
                sum_γ1 = 0;

                sum_α = 0;
                sum_β = 0;
                sum_γ = 0;

                // Process all synapses of the neuron
                for (int i = 0; i < Neurons; i++)
                {
                    // Acquire Pointers to subarrays relevant for this synapse at time t-1
                    αij = new float[K];
                    Array.Copy(Owner.Trace_α.Host, (j * Neurons) + (i * K), αij, 0, K);
                    βij = new float[K];
                    Array.Copy(Owner.Trace_β.Host, (j * Neurons) + (i * L), βij, 0, L);
                    γi = new float[L];
                    Array.Copy(Owner.Trace_γ.Host, i, γi, 0, L);
                    uij = new float[K];
                    Array.Copy(Owner.Weight_u.Host, (j * Neurons) + (i * K), uij, 0, K);
                    vij = new float[L];
                    Array.Copy(Owner.Weight_v.Host, (j * Neurons) + (i * L), vij, 0, L);
                    vji = new float[L];
                    Array.Copy(Owner.Weight_v.Host, (i * Neurons) + (j * L), vji, 0, L);

                    for (int k = 0; k < K; k++)
                    {
                        // Weight sum for Energy computation
                        sum_α1 += uij[k] * αij[k] * 1;
                        sum_α += uij[k] * αij[k] * x[j];
                    }

                    for (int l = 0; l < L; l++)
                    {
                        // Weight sum γj at time t-1 for Energy computation
                        sum_β1 += vij[l] * βij[l] * 1;
                        sum_β += vij[l] * βij[l] * x[j];
                        
                        sum_γ1 += vji[l] * γi[l]  * 1;
                        sum_γ += vji[l] * γi[l] * x[j];
                    }
                }
                #endregion

                #region Compute Energy, Log-likelihood and next Expected input
                // Compute Energy of each neuron
                Eθ[j] = (-Owner.Bias_b.Host[j] * 1) - sum_α1 + sum_β1 + sum_γ1;
                Owner.Energy_E.Host[j] = Eθ[j];
                // Compute probability of neuron has observed energy given its history
                Pθ[j] = (-Owner.Bias_b.Host[j] * x[j]) - sum_α + sum_β + sum_γ;

                // Compute next input estimate at time t denoted as Pθ
                expE = (float)Math.Exp(Math.Pow(-τ, -1) * Eθ[j]);
                X[j] = (float)expE / (1.0f + (float)expE);
                
                expP = Math.Exp(Math.Pow(-τ, -1) * Pθ[j]);
                Pθ[j] = (float)expP / (1.0f + (float)expE);

                // Compute a log likelihood
                logPθ = (float)Math.Log10(Pθ[j] == 0 ? 1 : Pθ[j]);
                Owner.LogLikelihood.Host[0] += logPθ;
                #endregion
            }
            #endregion

            #region Update Weights, Bias and Eligidible traces
            // Load learning rates
            m = 0;

            η = new float[Owner.LearningRate_η.Count];
            Δ = new float[Owner.LearningRate_η.Count];
            for (int i = 0; i < (Neurons * Neurons) + 2; i++)
            {
                η[i] = Owner.LearningRate_η.Host[i];
                Δ[i] = Owner.Derivative_Δ.Host[i];
            }

            for (int j = 0; j < Neurons; j++)
            {
                // Acquire γj at time t-1
                γj = new float[L];
                Array.Copy(Owner.Trace_γ.Host, (j * L), γj, 0, L);

                // For all synapses of each neuron
                for (int i = 0; i < Neurons; i++)
                {
                    // Index of the synapse
                    int id = ((j * Neurons) + i) / 2 + 2;
                    int id_K = (j * Neurons) + (i * K);
                    int id_L = (j * Neurons) + (i * L);

                    // Acquire Pointers to subarrays relevant for this synapse at time t-1
                    αij = new float[K];
                    Array.Copy(Owner.Trace_α.Host, id_K, αij, 0, K);
                    βij = new float[K];
                    Array.Copy(Owner.Trace_β.Host, id_L, βij, 0, L);
                    uij = new float[K];
                    Array.Copy(Owner.Weight_u.Host, id_K, uij, 0, K);
                    vij = new float[L];
                    Array.Copy(Owner.Weight_v.Host, id_L, vij, 0, L);

                    if (UpdateWeights)
                    {
                        #region Update Bias and Weights
                        
                        // Update bias value
                        Δ[0] += (float)Math.Pow( (x[j] - X[j]), 2);
                        Owner.Bias_b.Host[j] += η[0] * (x[j] - X[j]);

                        // Update LTP weight
                        for (int k = 0; k < K; k++)
                        {
                            Δ[1] += (float)Math.Pow( (x[j] - X[j]) * αij[k], 2);
                            Owner.Weight_u.Host[id_K] += η[1] * (x[j] - X[j]) * αij[k];
                        }

                        // Update both parts of LTD weight
                        for (int l = 0; l < L; l++)
                        {
                            Δ[id + 0] += (float)Math.Pow( (X[j] - x[j]) * βij[l], 2);
                            Δ[id + 1] += (float)Math.Pow( (X[i] - x[i]) * γj[l], 2);

                            Owner.Weight_v.Host[id_L] += η[id + 0] * (X[j] - x[j]) * βij[l];
                            Owner.Weight_v.Host[id_L] += η[id + 1] * (X[i] - x[i]) * γj[l];
                        }
                        #endregion
                    }

                    // Define head and tail of each FIFO queue
                    tail = Owner.Fifo_x[i].Tail();
                    head = Owner.Fifo_x[i].Head();

                    #region α - Feed forward connections
                    // Update α eligibility trace
                    for (int k = 0; k < K; k++)
                    {
                        αij[k] = λ[k] * (αij[k] + head);
                        Owner.Trace_α.Host[id_K + k] = αij[k];
                    }
                    #endregion

                    #region β - Only remembered recurrent connections
                    // FIFO indexes
                    int _head = Owner.Fifo_x[i].HeadIndex();
                    int _tail = Owner.Fifo_x[i].TailIndex();

                    // Update β eligibility trace - reset every iteration
                    for (int l = 0; l < L; l++)
                    {
                        // Update β eligibility trace
                        βij[l] = 0;
                        for (int s = (_head + 1); s <= _tail; s++)
                        {
                            float x_fifo = Owner.Fifo_x[i].ToArray()[s];
                            βij[l] += (float)Math.Pow(μ[l], s) * x_fifo;
                        }
                        Owner.Trace_β.Host[id_L + l] = βij[l];
                    }
                    #endregion
                }

                #region γ - Recurrent connections
                // Update γj for time t
                for (int l = 0; l < L; l++)
                {
                    γj[l] = μ[l] * (γj[l] + x[j]);
                    Owner.Trace_γ.Host[(j * L) + l] = γj[l];
                }
                #endregion

                #region Update learning rate usind AdaGrad
                if (m++ >= M)
                {
                    float max = Owner.MaximumLearningRate;

                    for (int i = 0; i < (Neurons * Neurons) + 2; i++)
                    {
                        Δ[i] = Δ[i] == 0 ? 1 : Δ[i];
                        η[i] = 1 / ((float)Math.Sqrt(Δ[i]));
                        η[i] = η[i] > max ? max : η[i];

                        Δ[i] = 0;
                    }
                    
                    m = 0;
                }

                for (int i = 0; i < (Neurons * Neurons) + 2; i++)
                {
                    Owner.LearningRate_η.Host[i] = η[i];
                    Owner.Derivative_Δ.Host[i] = Δ[i];
                }
                #endregion
            }
            #endregion

            #region Manage FIFO queues
            for (int j = 0; j < Neurons; j++)
            {
                for (int i = 0; i < Neurons; i++)
                {
                    // Index of the synapse
                    int idx = j * Neurons + i;

                    // Insert new value to the queue
                    Owner.Fifo_x[idx].Enqueue(x[j]);
                    // Discard the head value of the queue
                    Owner.Fifo_x[idx].Dequeue();
                    Array.Copy(Owner.Fifo_x[idx].ToArray(), 0, Owner.FIFO_trace.Host, idx * Owner.DelayInterval, Owner.Fifo_x[idx].Count);
                }
            }
            #endregion

            // Copy Expected input to the output of the Layer
            X.CopyTo(Owner.Output.Host, 0);

            Owner.Output.SafeCopyToDevice();
            Owner.Trace_α.SafeCopyToDevice();
            Owner.Trace_β.SafeCopyToDevice();
            Owner.Trace_γ.SafeCopyToDevice();
            Owner.Bias_b.SafeCopyToDevice();
            Owner.Energy_E.SafeCopyToDevice();
            Owner.Weight_u.SafeCopyToDevice();
            Owner.Weight_v.SafeCopyToDevice();
            Owner.FIFO_trace.SafeCopyToDevice();
            Owner.LogLikelihood.SafeCopyToDevice();
            Owner.LearningRate_η.SafeCopyToDevice();
            Owner.Derivative_Δ.SafeCopyToDevice();
        }
    }

}
