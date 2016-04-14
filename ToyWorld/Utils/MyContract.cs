using System;
using System.Diagnostics.Contracts;

namespace Utils
{
    /// <summary>
    /// Used for implementing Exception version of Reguires Code Contract for people without Code Contracts installed
    /// </summary>
    public static class MyContract
    {
        public static void Requires<TException>(bool condition, string message = "") where TException : Exception, new()
        {
#if CONTRACTS_FULL
            // TODO: Message is missing because CC wants it to be known before build (it has to be static or string literal)
            Contract.Requires<TException>(condition);
#else
            if (!condition) throw new TException();
#endif
        }
    }
}
