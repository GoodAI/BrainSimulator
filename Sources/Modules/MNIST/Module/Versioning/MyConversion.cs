using GoodAI.Core.Configuration;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;
using System.Xml.XPath;

namespace MNIST.Versioning
{
    public class MyConversion : MyBaseConversion
    {
        public override int CurrentVersion
        {
            get { return 1; }
        }
    }
}
