using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.BrainSimulator.Properties;
using GoodAI.Core.Configuration;
using GoodAI.Core.Utils;

namespace GoodAI.BrainSimulator.UserSettings
{
    internal static class ToolBarNodes
    {
        public static void InitDefaultToolBar(Settings settings, Dictionary<Type, MyNodeConfig> knownNodes)
        {
            //PrintToolBarNodes(settings);  // uncomment to get a text dump that can be used in ListDefaultToolBarNodes() below

            //if (Debugger.IsAttached)
            //    settings.Reset();

            if (settings.ToolBarNodes != null)
                return;

            settings.ToolBarNodes = new StringCollection();

            if (settings.QuickToolBarNodes == null)
                settings.QuickToolBarNodes = new StringCollection();

            var nodesToAdd = new HashSet<string>(ListDefaultToolBarNodes());

            var quickBarNodesToAdd = new HashSet<string>(ListDefaultQuickBarNodes());

            foreach (string nodeName in knownNodes.Values
                .Where(nodeConfig => nodeConfig.CanBeAdded && (nodeConfig.NodeType != null))
                .Select(nodeConfig => nodeConfig.NodeType.Name)
                .Where(nodeName => nodesToAdd.Contains(nodeName)))
            {
                settings.ToolBarNodes.Add(nodeName);

                if (quickBarNodesToAdd.Contains(nodeName))
                    settings.QuickToolBarNodes.Add(nodeName);
            }
        }

        private static IEnumerable<string> ListDefaultToolBarNodes()
        {
            return new[]
            {
                "MyAbsoluteValue",
                "MyAccumulator",
                "MyConditionalGroup",
                "MyCSharpNode",
                "MyFork",
                "MyGenerateInput",
                "MyHiddenLayer",
                "MyJoin",
                "MyLoopGroup",
                "MyNeuralNetworkGroup",
                "MyNodeGroup",
                "MyOutputLayer",
                "MyPolynomialFunction",
                "MyPythonNode",
                "MyRandomNode",
                "MyUserInput"
            };
        }

        private static IEnumerable<string> ListDefaultQuickBarNodes()
        {
            return new[]
            {
                "MyFork",
                "MyJoin",
                "MyNodeGroup",
                "MyPolynomialFunction"
            };
        }

        /// <summary>
        /// Helper method for creating default toolbar based on current settings. (Unused in normal program run.)
        /// </summary>
        private static void PrintToolBarNodes(Settings settings)
        {
            if (settings.ToolBarNodes == null)
                return;

            string tempFile = Path.Combine(Path.GetTempPath(), @"toolbarnodes.txt");
            IEnumerable<string> nodesList = settings.ToolBarNodes.Cast<string>().ToList()
                .OrderBy(name => name)
                .Select(nodeName => string.Format("                \"{0}\",", nodeName));

            File.WriteAllLines(tempFile, nodesList);

            MyLog.DEBUG.WriteLine("ToolBarNodes dumped to '{0}'", tempFile);
        }
    }
}
