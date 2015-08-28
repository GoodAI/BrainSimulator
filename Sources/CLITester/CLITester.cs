using CLIWrapper;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CLITester
{
    class CLITester
    {
        static void Main(string[] args)
        {
            BSCLI CLI = new BSCLI(MyLogLevel.WARNING);
            CLI.OpenProject(@"C:\Users\michal.vlasak\Desktop\bindTest2.brain");
            float iterations = 250;

            List<Tuple<int, int, float, float>> results = new List<Tuple<int, int, float, float>>();
            //CLI.Set(6, "OutputSize", 32);

            for (int symbolSize = 512; symbolSize <= 8192; symbolSize *= 2)
            {
                for (int binds = 20; binds <= 50; binds += 5)
                {
                    float okSum = 0;
                    //float wrongSum = 0;
                    CLI.Set(7, "Binds", binds);
                    CLI.Set(7, "SymbolSize", symbolSize);
                    for (int i = 0; i < iterations; ++i)
                    {
                        CLI.Run(1, 10);
                        float okDot = CLI.GetValues(8)[0];
                        //float wrongDot = CLI.GetValues(9)[0];
                        okSum += okDot;
                        //wrongSum += wrongDot;
                        CLI.Stop();
                        if ((i + 1) % 10 == 0)
                        {
                            MyLog.WARNING.Write('.');
                        }
                    }
                    MyLog.WARNING.WriteLine();
                    float wrongSum = 1;
                    MyLog.WARNING.WriteLine("Results:" + symbolSize + "@" + binds + " => " + okSum / iterations + " / " + wrongSum / iterations);
                    results.Add(new Tuple<int, int, float, float>(symbolSize, binds, okSum / iterations, wrongSum / iterations));
                }
            }

            File.WriteAllLines(@"C:\Users\michal.vlasak\Desktop\results.txt", results.Select(n => n.ToString().Substring(1, n.ToString().Length - 2)));

            CLI.Quit();
            Console.ReadLine();
        }
    }
}
