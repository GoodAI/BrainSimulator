using GoodAI.Core.Configuration;

namespace GoodAI.Modules.Versioning
{
    public class MyConversion : MyBaseConversion
    {
        public override int CurrentVersion
        {
            get { return 1; }
        }
    }
}
