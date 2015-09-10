using GoodAI.Core.Configuration;

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
