using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Platform.Core.Observers
{
    /// <author>GoodAI</author>
    /// <meta>jv</meta>
    /// <status>working</status>
    /// <summary>
    /// An observer which is able to produce custom peek labels (values shown when clicking onto the MemoryBlock).
    /// </summary>
    public interface ICustomPeekLabelProducingObserver
    {
        /// <summary>
        /// Given the absolute coordinates of the click in the observer, return some custom text to be shown on the right corner.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns>Custom String to be shown on the top right corner, null if the label should not be shown.</returns>
        String GetPeekLabelAt(int x, int y);
    }
}
