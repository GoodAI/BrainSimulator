using System;
using System.Diagnostics.Contracts;

namespace Utils
{
    public static class MyContract
    {
        public static void Requires<TException>(bool condition, string message = "") where TException : Exception, new()
        {
#if CONTRACTS_FULL
            Contract.Requires<TException>(condition);
#else
            if (!condition) throw new TException();
#endif
        }
    }
}
