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
    class CLIProgram
    {
        static void Main(string[] args)
        {
            BSCLI CLI = new BSCLI(MyLogLevel.WARNING);
            CLI.OpenProject(@"C:\Users\michal.vlasak\Desktop\Breakout.brain");
            CLI.Quit();
            Console.ReadLine();
            return;
        }
    }
}

/* Last bind test
 * BSCLI CLI = new BSCLI(MyLogLevel.WARNING);
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
 * /


/* Binding test 2
 * BSCLI CLI = new BSCLI(MyLogLevel.WARNING);
            CLI.OpenProject(@"C:\Users\michal.vlasak\Desktop\bindTest2.brain");
            float iterations = 250;

            List<Tuple<int, int, float, float>> results = new List<Tuple<int, int, float, float>>();
            //CLI.Set(6, "OutputSize", 32);

            for (int symbolSize = 512; symbolSize <= 8192; symbolSize *= 2) {
                for (int binds = 20; binds <= 50; binds += 5) {
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

            File.WriteAllLines(@"C:\Users\michal.vlasak\Desktop\results.txt", results.Select(n => n.ToString().Substring(1, n.ToString().Length-2)));

            CLI.Quit();
            Console.ReadLine();
 */

/* Bind/ubnind accuracy test
 *      static void Main(string[] args){
 *          BSCLI CLI = new BSCLI(MyLogLevel.WARNING);
            CLI.OpenProject(@"C:\Users\michal.vlasak\Desktop\bindingTest.brain");

            int[] checks = { 116, 117, 118, 322, 363 };

            foreach(int checkId in checks){
                MyLog.WARNING.WriteLine(checkId);
                for (int i = 4; i <= 1024; i *= 2)
                {
                    int symbolSize = i;
                    List<MyNode> cbs = CLI.GetNodesOfType(typeof(MyCodeBook));
                    foreach (MyNode n in cbs)
                    {
                        CLI.Set(n.Id, "SymbolSize", symbolSize);
                    }
                    int oks = okWith(CLI, checkId);
                    MyLog.WARNING.WriteLine(symbolSize + "-" + oks + "/22");
                }
            }
            
            CLI.Quit();
            Console.ReadLine();
        }

        static int okWith(BSCLI CLI, int nodeId) {
            int round = 0;
            int oks = 0;
            foreach (var code in Enum.GetValues(typeof(BrainSimulator.VSA.MyCodeVector)))
            {
                if (round == 0) {
                    round++;
                    continue;
                }
                CLI.Set(33, typeof(MyCodeBook.MyCodeVectorsTask), "CodeVector", code.ToString());
                CLI.Run(10, 0);
                oks += check(round, nodeId, CLI) ? 1 : 0;

                CLI.Stop();
                round++;
            }
            return oks;
        }

        static bool check(int correctId, int nodeId, BSCLI CLI) {
            float[] res = CLI.GetValues(nodeId);
            float max = res[0];
            int maxIdx = 0;
            //from 1 becose 0 is empty
            for(int i = 1; i < res.Length; ++i)
            {
                if (res[i] > max) {
                    max = res[i];
                    maxIdx = i;
                }
            }
            return correctId == maxIdx ? true : false;
        }
 */

/*Timespan test - for old version with simple PlotResults
 * 
            uint bootstrapTime = 10;
            uint expositionTime = 10;
            uint delay = 50;
            uint symbols = 1;
            string filename = "output";
            //ask time is static value, because learning will be turned of, so it doesn't matter
            
            BSCLI CLI = new BSCLI();
            CLI.OpenProject(@"C:\Users\michal.vlasak\Desktop\capacityTest.brain");
            
            
            CLI.Load(10, true);
            CLI.Save(10, false);

            for (uint i = 1; i <= 4; ++i) {
                for (uint k = 100; k <= 1000; k+=200) {
                    for (uint j = 1; j <= 5; j+=1) {
                        expositionTime = j;
                        delay = k;
                        symbols = i;
                        string pattern = "-" + bootstrapTime + "," + expositionTime + ",0,0,-" + delay + ",10,0,0";
                        CLI.Set(9, "SymbolCount", symbols);
                        CLI.Set(10, typeof(MySelfOrganizingMap.MyAdaptationTask), "LearningFactor", 0.1);
                        CLI.Set(11, typeof(MyMultiplexerNode.MyRoutingTask), "Pattern", pattern);
                        CLI.Set(11, typeof(MyMultiplexerNode.MyRoutingTask), "Rotate", false);
                        int valId = CLI.TrackValue(15, 100, 2);

                        CLI.Run(bootstrapTime + expositionTime + delay, 50);

                        CLI.Set(10, typeof(MySelfOrganizingMap.MyAdaptationTask), "LearningFactor", 0);

                        CLI.Run(50, 50);

                        filename = symbols + "-" + expositionTime + " after " + delay;
                        CLI.PlotResults(valId, "C:/Users/michal.vlasak/Desktop/tests/" + filename + ".png", filename + " @ " + pattern);
                        MyLog.INFO.WriteLine("Next run");
                        CLI.Stop();
                    }
                }    
            }
            CLI.Quit();
            Console.ReadLine();
 */

/*Capacity test
for (uint i = 2; i <= 1024; i*=2) {
                for (uint j = 1; j <= 10001; j+=500) {
                    uint symbolSize = i;
                    uint symbolCount = j;

                    CLI.Set(9, "SymbolSize", symbolSize);
                    CLI.Set(19, "SymbolSize", symbolSize);
                    CLI.Set(7, "OutputSize", symbolSize);
                    CLI.Set(9, "SymbolCount", symbolCount-1);

                    int correct = CLI.TrackValue(15, 100, 1);
                    int second = CLI.TrackValue(19, 100, 1);
                    int random = CLI.TrackValue(18, 100, 1);

                    CLI.Run(100, 50);

                    string filename = symbolSize + "-" + symbolCount;
                    int[] ids = {correct, random, second};
                    string[] titles = { filename, "random" };
                    CLI.PlotResults(ids, "C:/Users/michal.vlasak/Desktop/tests/capacity/" + filename + ".png", titles);
                    MyLog.INFO.WriteLine("Next run");
                    CLI.Stop();
                }
            }
*/