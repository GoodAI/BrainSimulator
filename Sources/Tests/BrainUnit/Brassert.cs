using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Testing.BrainUnit
{
    public class BrassertFailedExc : Exception
    {
        public BrassertFailedExc(string message) : base(message) { }
    }

    public static class Brassert
    {
        public static void Fail(string message = "")
        {
            throw new BrassertFailedExc(message);
        }

        public static void True(bool condition, string message = "")
        {
            if (!condition)
                throw new BrassertFailedExc(message);
        }
    }
}
