using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.ComponentModel;

namespace GoodAI.Modules.DyBM.Tasks
{
    /// <summary>
    /// Layer learning task
    /// </summary>
    [Description("DyBM Learning"), MyTaskInfo(OneShot = false)]
    public class MyDyBMLearningTask : MyTask<MyDyBMInputLayer>
    {
        /// <summary>
        /// Initialize all parameters to 0 -> meaning that previous sequences were all 0
        /// </summary>
        /// <param name="nGPU"></param>
        public override void Init(int nGPU)
        {
            if (Owner.Delay_d != null)
            {
                Random rand = new Random();
                Owner.Delay_d.SafeCopyToHost();
                Owner.Bias_b.SafeCopyToHost();
                Owner.Weight_u.SafeCopyToHost();
                Owner.Weight_v.SafeCopyToHost();

                // Generated interval
                float min = 0.01f;
                float max = 0.1f;

                for (int s = 0; s < Owner.Neurons; s++)
                {
                    // Randomize bias
                    Owner.Bias_b.Host[s] = (float)rand.NextDouble() * (max - min) + min;
                }

                for (int s = 0; s < Owner.Synapses; s++)
                {
                    // Init conduction delays
                    Owner.Delay_d.Host[s] = Owner.Fifo_x[s].Count;
                    // Randomize weights
                    Owner.Weight_u.Host[s] = (float)rand.NextDouble() * (max - min) + min;
                    Owner.Weight_v.Host[s] = (float)rand.NextDouble() * (max - min) + min;
                }

                Owner.Delay_d.SafeCopyToDevice();
                Owner.Bias_b.SafeCopyToDevice();
                Owner.Weight_u.SafeCopyToDevice();
                Owner.Weight_v.SafeCopyToDevice();
            }
        }

        /// <summary>
        /// Execute one learning step of the layer through all neurons
        /// </summary>
        public override void Execute()
        {
            int L, K, M, m;
            float head, tail;
            float sum_α, sum_β, sum_γ, sum_αP, sum_βP, sum_γP, logPθ;
            float[] αij, βij, γj, uij, vij, μ, λ, Eθ, Pθ;

            int Neurons = Owner.Neurons;
            float τ = Owner.Temperature_τ;
            float η = Owner.LearningRate_η;
            float Δ = Owner.Derivative_Δ;
            Pθ = new float[Neurons];

            m = 0;
            K = Owner.Traces_K;
            L = Owner.Traces_L;
            M = Owner.LearningRateCorection_M;
            Eθ = Owner.Energy_E.Host;
            μ = Owner.slope_μ.ToArray();
            λ = Owner.slope_λ.ToArray();

            Owner.Previous_x.SafeCopyToHost();
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

            Owner.LogLikelihood.Host[0] = 0;

            #region Take the input of the Layer
            Owner.Input.SafeCopyToHost();
            // Input of the at time t-1
            float[] x = Owner.Previous_x.Host; 
            // Expected input at time t
            float[] X = new float[Neurons];
            #endregion

            #region Feed-Forward pass for all neurons
            logPθ = 0;
            for (int j = 0; j < Neurons; j++)
            {
                #region Compute sums of weighted eligidible traces
                sum_α = 0;
                sum_β = 0;
                sum_γ = 0;

                sum_αP = 0;
                sum_βP = 0;
                sum_γP = 0;

                // Process all synapses of the neuron
                for (int i = 0; i < Neurons; i++)
                {
                    // Index of the synapse
                    int id = (j * Neurons) + i;
                    int id_K = (j * Neurons) + (i * K);
                    int id_L = (j * Neurons) + (i * L);

                    // Define head and tail of each FIFO queue
                    tail = Owner.Fifo_x[id].Tail();
                    head = Owner.Fifo_x[id].Head();

                    // Acquire Pointers to subarrays relevant for this synapse at time t-1
                    αij = new float[K];
                    Array.Copy(Owner.Trace_α.Host, id_K, αij, 0, K);
                    βij = new float[K];
                    Array.Copy(Owner.Trace_β.Host, id_L, βij, 0, L);
                    γj = new float[L];
                    Array.Copy(Owner.Trace_γ.Host, i, γj, 0, L);
                    uij = new float[K];
                    Array.Copy(Owner.Weight_u.Host, id_K, uij, 0, K);
                    vij = new float[L];
                    Array.Copy(Owner.Weight_v.Host, id_L, vij, 0, L);

                    for (int k = 0; k < K; k++)
                    {
                        // Weight sum for Energy computation
                        sum_α += uij[k] * αij[k] * 1;
                        sum_αP += uij[k] * αij[k] * Owner.Input.Host[j];
                    }

                    for (int l = 0; l < L; l++)
                    {
                        // Weight sum γj at time t-1 for Energy computation
                        sum_β += vij[l] * βij[l] * 1;
                        sum_γ += vij[l] * γj[l]  * 1;

                        sum_βP += vij[l] * βij[l] * Owner.Input.Host[j];
                        sum_γP += vij[l] * γj[l] * Owner.Input.Host[j];
                    }
                }
                #endregion

                #region Compute Energy, Log-likelihood and next Expected input
                // Compute Energy of each neuron
                Eθ[j] = (-Owner.Bias_b.Host[j] * 1) - sum_α + sum_β + sum_γ;
                Owner.Energy_E.Host[j] = Eθ[j];

                // Compute next input estimate at time t denoted as Pθ
                float exp = (float)Math.Exp(Math.Pow(-τ, -1) * Eθ[j]);
                X[j] = exp / (1.0f + exp);

                // Compute a log likelihood
                Pθ[j] = (-Owner.Bias_b.Host[j] * Owner.Input.Host[j]) - sum_αP + sum_βP + sum_γP;

                Pθ[j] = (float)(Math.Pow(-0.0001f, -1) * Pθ[j]) / (1.0f + (float)(Math.Pow(-0.0001f, -1) * Eθ[j]));
                logPθ = (float)Math.Log10(Pθ[j] == 0 ? 1 : Pθ[j]);
                Owner.LogLikelihood.Host[0] += logPθ;
                #endregion
            }
            #endregion

            #region Update Weights, Bias and Eligidible traces
            float[] xt = Owner.Input.Host;

            for (int j = 0; j < Neurons; j++)
            {
                // Update bias value
                Owner.Bias_b.Host[j] += η * (xt[j] - X[j]);
                
                // Acquire γj at time t-1
                γj = new float[L];
                int idx = (j * L);
                Array.Copy(Owner.Trace_γ.Host, idx, γj, 0, L);

                // For all synapses of each neuron
                for (int i = 0; i < Neurons; i++)
                {
                    // Index of the synapse
                    int id = (j * Neurons) + i;
                    int id_K = (j * Neurons) + (i * K);
                    int id_L = (j * Neurons) + (i * L);

                    // Define head and tail of each FIFO queue
                    tail = Owner.Fifo_x[id].Tail();
                    head = Owner.Fifo_x[id].Head();

                    // Acquire Pointers to subarrays relevant for this synapse at time t-1
                    αij = new float[K];
                    Array.Copy(Owner.Trace_α.Host, id_K, αij, 0, K);
                    βij = new float[K];
                    Array.Copy(Owner.Trace_β.Host, id_L, βij, 0, L);
                    uij = new float[K];
                    Array.Copy(Owner.Weight_u.Host, id_K, uij, 0, K);
                    vij = new float[L];
                    Array.Copy(Owner.Weight_v.Host, id_L, vij, 0, L);

                    #region Update Weights
                    // Update LTP weight
                    for (int k = 0; k < K; k++)
                    {
                        Owner.Weight_u.Host[id_K] += η * (xt[j] - X[j]) * αij[k];
                    }

                    // Update first part of LTD weight
                    for (int l = 0; l < L; l++)
                    {
                        Owner.Weight_v.Host[id_L] += η * (X[j] - xt[j]) * βij[l];
                    }

                    // Update second part of LTD weight
                    for (int l = 0; l < L; l++)
                    {
                        Owner.Weight_v.Host[id_L] += η * (X[j] - xt[j]) * γj[l];
                    }

                    /* Update learning rate
                    if (m++ < M)
                    {
                        Δ += logPθ * logPθ;
                    }
                    else
                    {
                        Owner.LearningRate_η = η / ((float)Math.Sqrt(Δ > 0 ? Δ : 1));
                        Δ = 0;
                        m = 0;
                    }*/
                    #endregion

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
                    int _head = Owner.Fifo_x[id].HeadIndex();
                    int _tail = Owner.Fifo_x[id].TailIndex();

                    // Update β eligibility trace - reset every iteration
                    for (int l = 0; l < L; l++)
                    {
                        // Update β eligibility trace
                        βij[l] = 0;
                        for (int s = (_head + 1); s <= _tail; s++)
                        {
                            float x_fifo = Owner.Fifo_x[id].ToArray()[s];
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
                    γj[l] = μ[l] * (γj[l] + xt[j]);
                    Owner.Trace_γ.Host[idx + l] = γj[l];
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
            // Copy current input to be the previous input in next step
            Owner.Input.Host.CopyTo(Owner.Previous_x.Host, 0);

            Owner.Previous_x.SafeCopyToDevice();
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
        }
    }

}
