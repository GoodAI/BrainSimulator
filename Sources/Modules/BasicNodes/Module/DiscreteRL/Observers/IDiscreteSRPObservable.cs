using GoodAI.Modules.Harm;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.BasicNodes.DiscreteRL.Observers
{
    /// <summary>
    /// Interface for HARM
    /// </summary>
    interface IDiscreteSRPObservable : IDiscretePolicyObservable
    {
        /// <summary>
        /// return StochasticReturnPredictor given by its index
        /// </summary>
        /// <param name="ind"></param>
        /// <returns></returns>
        MyStochasticReturnPredictor GetPredictorNo(int ind);
    }
}
