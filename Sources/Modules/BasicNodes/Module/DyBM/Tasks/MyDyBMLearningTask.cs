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

                for (int s = 0; s < Owner.Neurons; s++)
                {
                    // Randomize bias
                    Owner.Bias_b.Host[s] = (float)rand.Next(1, 10) / 100;
                }

                for (int s = 0; s < Owner.Synapses; s++)
                {
                    // Init conduction delays
                    Owner.Delay_d.Host[s] = Owner.Fifo_x[s].Count;
                    // Randomize weights
                    Owner.Weight_u.Host[s] = (float)rand.Next(1, 10) / 100;
                    Owner.Weight_v.Host[s] = (float)rand.Next(1, 10) / 100;
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
            float[] αij, βij, γj, uij, vij, Eθ, Ȇθ, μ, λ;

            int Neurons = Owner.Neurons;
            float τ = Owner.Temperature_τ;
            float η = Owner.LearningRate_η;
            float Δ = Owner.Derivative_Δ;
            
            m = 0;
            M = Owner.LearningRateCorection_M;
            K = Owner.Traces_K;
            L = Owner.Traces_L;
            Eθ = Owner.Energy_E.Host;

            Owner.Output.SafeCopyToHost();
            Owner.Trace_α.SafeCopyToHost();
            Owner.Trace_β.SafeCopyToHost();
            Owner.Trace_γ.SafeCopyToHost();
            Owner.Bias_b.SafeCopyToHost();
            Owner.Energy_E.SafeCopyToHost();
            Owner.Weight_u.SafeCopyToHost();
            Owner.Weight_v.SafeCopyToHost();


            #region Take the input of the Layer
            Owner.Input.SafeCopyToHost();
            // Input of the at time t
            float[] x = Owner.Input.Host;
            // Expected input at time t
            float[] X = new float[Neurons];
            #endregion

            #region Feed-Forward pass for all neurons
            for (int j = 0; j < Neurons; j++)
            {
                // Define head and tail of each FIFO queue
                tail = Owner.Fifo_x[j].Tail();
                head = Owner.Fifo_x[j].Head();
                
                μ = Owner.slope_μ.ToArray();
                λ = Owner.slope_λ.ToArray();

                // Compute eligibility traces
                float sum_α = 0;
                float sum_β = 0;
                float sum_γ = 0;

                γj = new float[L];
                Array.Copy(Owner.Trace_γ.Host, (j * L), γj, 0, L);
                
                #region γ - Recurrent connections
                for (int l = 0; l < L; l++)
                {
                    γj[l] = μ[l] * (γj[l] + tail);
                    Owner.Trace_γ.Host[(j * L) + l] = γj[l];
                }
                #endregion

                #region Process all synapses of each neuron
                for (int i = 0; i < Neurons; i++)
                {
                    // Index of the synapse
                    int idx = (j * Neurons) + (i * K);

                    // Pointers to subarrays relevant for this synapse
                    αij = new float[K];
                    Array.Copy(Owner.Trace_α.Host, idx, αij, 0, K);
                    βij = new float[K];
                    Array.Copy(Owner.Trace_β.Host, idx, βij, 0, K);
                    uij = new float[K];
                    Array.Copy(Owner.Weight_u.Host, idx, uij, 0, K);
                    vij = new float[L];
                    Array.Copy(Owner.Weight_v.Host, idx, vij, 0, L);

                    #region α - Feed forward connections
                    for (int k = 0; k < K; k++)
                    {
                        // Update α eligibil;ity trace
                        float d = Owner.Delay_d.Host[(j * Neurons) + i];
                        αij[k] = λ[k] * (αij[k] + head);
                        Owner.Trace_α.Host[idx] = αij[k];

                        // Weight sum for Energy computation
                        sum_α += uij[k] * αij[k] * x[j];
                    }
                    #endregion

                    #region β - Only remembered recurrent connections
                    // FIFO indexes
                    int _tail = 0;
                    int _head = Owner.Fifo_x[i].Count - 1;

                    // Update β eligibility trace - reset every iteration
                    for (int l = 0; l < L; l++)
                    {
                        βij[l] = 0;
                        for (int s = _head; s > _tail; s--)
                        {
                            float xi = Owner.Fifo_x[i].ToArray()[s];
                            βij[l] += (float)Math.Pow(μ[l], -s) * xi;
                        }

                        Owner.Trace_β.Host[idx] = βij[l];

                        // Weight sums for Energy computation
                        sum_β += vij[l] * βij[l] * x[j];
                        sum_γ += vij[l] * μ[l] * x[j];
                    }
                    #endregion
                }
                #endregion

                #region Compute Energy, Log-likelihood and next Expected input
                // Compute Energy of each neuron
                Eθ[j] = -Owner.Bias_b.Host[j] * x[j] - sum_α + sum_β + sum_γ;
                Owner.Energy_E.Host[j] = Eθ[j];

                // Compute next input estimate denoted as Pθ
                X[j] = (float)(Math.Exp(Math.Pow(-τ, -1) * Eθ[j]) /
                  (1 + (float)(Math.Exp(Math.Pow(-τ, -1) * Eθ[j]) )));
                
                if (float.IsNaN(X[j]))
                    X[j] = 0;

                // Compute a log likelihood
                float logPθ = (float)Math.Log10(X[j]);
                #endregion

                #region Update bias and all synapses of each neuron
                // Update bias value
                Owner.Bias_b.Host[j] += η * (x[j] - X[j]);
                
                // Update all synapses
                for (int i = 0; i < Neurons; i++)
                {
                    // Index of the synapse
                    int idx = (j * Neurons) + (i * K);

                    // Pointers to subarrays relevant for this synapse
                    αij = new float[K];
                    Array.Copy(Owner.Trace_α.Host, idx, αij, 0, K);
                    βij = new float[K];
                    Array.Copy(Owner.Trace_β.Host, idx, βij, 0, K);

                    // Update learning rate
                    if (m < M)
                    {
                        Δ += logPθ * logPθ;
                    }
                    else
                    {
                        Owner.LearningRate_η = η / ((float)Math.Sqrt(Δ));
                        Δ = 0;
                        m = 1;
                    }

                    // Update LTP weight
                    for (int k = 0; k < K; k++)
                    {
                        Owner.Weight_u.Host[idx] += η * (x[j] - X[j]) * αij[k];
                    }

                    // Update first part of LTD weight
                    for (int l = 0; l < L; l++)
                    {
                        Owner.Weight_v.Host[idx] += η * (X[j] - x[j]) * βij[l];
                    }

                    // Update second part of LTD weight
                    for (int l = 0; l < L; l++)
                    {
                        Owner.Weight_v.Host[idx] += η * (X[j] - x[i]) * γj[l];
                    }
                }
                #endregion
            }
            #endregion

            #region Manage FIFO queues
            for (int j = 0; j < Neurons; j++)
            {
                // Insert new value to the queue
                Owner.Fifo_x[j].Enqueue(x[j]);
                // Discard the head value of the queue
                Owner.Fifo_x[j].Dequeue();
            }
            #endregion

            X.CopyTo(Owner.Output.Host, 0);
            

            Owner.Output.SafeCopyToDevice();
            Owner.Trace_α.SafeCopyToDevice();
            Owner.Trace_β.SafeCopyToDevice();
            Owner.Trace_γ.SafeCopyToDevice();
            Owner.Bias_b.SafeCopyToDevice();
            Owner.Energy_E.SafeCopyToDevice();
            Owner.Weight_u.SafeCopyToDevice();
            Owner.Weight_v.SafeCopyToDevice();
        }
    }
}
