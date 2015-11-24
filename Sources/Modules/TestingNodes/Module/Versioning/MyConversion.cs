using GoodAI.Core.Configuration;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Xml.Linq;

namespace GoodAI.Modules.Versioning
{
    public class MyConversion : MyBaseConversion
    {
        public override int CurrentVersion { get { return 1; } }
    }
}
