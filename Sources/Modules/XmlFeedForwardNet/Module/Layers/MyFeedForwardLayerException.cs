using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace XmlFeedForwardNet.Layers
{
    public class MyFeedForwardLayerException : Exception
    {
        public MyFeedForwardLayerException(string msg) : base(msg) { }
    }
}
