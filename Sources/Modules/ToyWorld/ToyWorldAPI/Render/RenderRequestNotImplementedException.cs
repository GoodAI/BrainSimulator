using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.ToyWorld.Control
{
    /// <summary>
    /// Thrown when requesting an unknown IRenderRequest or IAvatarRenderRequest from an IGameController.
    /// </summary>
    public class RenderRequestNotImplementedException : Exception
    {
        /// <summary>
        /// Initializes a new instance of the RenderRequestNotImplementedException class with the specified message.
        /// </summary>
        /// <param name="message">The message that describes the error.</param>
        public RenderRequestNotImplementedException(string message)
            : base(message)
        { }
    }
}
