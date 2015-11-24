using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Execution;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Testing.BrainUnit;

namespace GoodAI.Tests.BrainTestRunner
{
    internal sealed class BrainUnitNodeTest : BrainTest
    {
        private MyNode m_brainUnitNode;

        public BrainUnitNodeTest(string brainUnitNodeProjectPath)
        {
            BrainFileName = brainUnitNodeProjectPath;
        }

        public override string Name
        {
            get
            {
                return string.Format("BrainUnitNode test: '{0}'", Path.GetFileName(BrainFileName));
            }
        }

        public override void Check(IBrainScan b)
        {
            if (m_brainUnitNode == null)
                throw new InvalidOperationException("Test not initialized, BrainUnitNode instance not set.");

            MethodInfo checkMethod = m_brainUnitNode.GetType().GetMethod("Check");

            try
            {
                checkMethod.Invoke(m_brainUnitNode, new object[0]);
            }
            catch (TargetInvocationException e)
            {
                Exception innerException = e.InnerException ?? e;
                throw innerException;
            }
        }

        public void Initialize(MyProjectRunner projectRunner)
        {
            try
            {
                projectRunner.OpenProject(BrainFileName);

                List<MyNode> brainUnitNodes = projectRunner.Filter(node => (node.GetType().Name == "BrainUnitNode"));
                if (brainUnitNodes.Count != 1)
                    throw new InvalidTestException("Exactly 1 occurrence of BrainUnitNode required.");  // TODO: allow more

                m_brainUnitNode = brainUnitNodes[0];

                PropertyInfo maxStepCountProperty = m_brainUnitNode.GetType().GetProperty("MaxStepCount", typeof(int));

                MaxStepCount = (int)maxStepCountProperty.GetValue(m_brainUnitNode);
            }
            catch (Exception e)
            {
                MyLog.ERROR.WriteLine(
                    "Failed to instantiate test {0}: {1}", Path.GetFileName(BrainFileName), e.Message);
                throw;
            }
        }
    }
}
