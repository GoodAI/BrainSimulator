using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Testing.BrainUnit;

namespace GoodAI.Tests.BrainTestRunner
{
    internal class TestReporter
    {
        private int m_passCount = 0;
        private int m_failCount = 0;

        public void StartTest(BrainTest test)
        {
            PrintLine("RUN", test);
        }

        public void AddInvalidTest(BrainTest test, InvalidTestException e)
        {
            Evaluate(passed: false);
            PrintLine("invalid test", test, e.Message, ConsoleColor.Red);
        }

        public void AddFail(BrainTest test, Exception e)
        {
            Evaluate(passed: test.ExpectedToFail);

            if (test.ExpectedToFail)
                PrintLine("      XF", test, "(Expected to fail.)", ConsoleColor.DarkGray);
            else
                PrintLine("    FAIL", test, e.Message, ConsoleColor.Red);
        }

        public void AddPass(BrainTest test)
        {
            Evaluate(passed: !test.ExpectedToFail);

            if (test.ExpectedToFail)
                PrintLine(" Unexpected OK!", test, "Expected to fail!", ConsoleColor.Red);
            else
                PrintLine("      OK", test);
        }

        public void AddCrash(BrainTest test, Exception e)
        {
            Evaluate(passed: false);
            PrintLine("CRASH!", test, e.Message, ConsoleColor.Red);
        }

        public void Conclude()
        {
            if (m_failCount + m_passCount == 0)
            {
                Console.WriteLine("(No tests run)");
            }
            else if (m_failCount == 0)
            {
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine("\n>>> PASS <<<  ({0} tests run)\n", m_passCount);
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("\n>>> FAIL! <<<  ({0} out of {1} tests failed)\n",
                    m_failCount, m_passCount + m_failCount);
            }

            Console.ResetColor();
        }

        private void Evaluate(bool passed)
        {
            if (passed)
                m_passCount++;
            else
                m_failCount++;

            // TODO: measure elapsed time
        }

        private void PrintLine(string what, BrainTest test, string message = "", ConsoleColor color = 0)
        {
            if (color != 0)
                Console.ForegroundColor = color;

            Console.Write(" {0, -10} ", what);
            Console.ResetColor();

            Console.WriteLine("{0}{1}", test.Name, string.IsNullOrEmpty(message) ? "" : ": " + message);
        }
    }
}
