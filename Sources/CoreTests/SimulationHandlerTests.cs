using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Execution;
using NUnit.Framework;
using Rhino.Mocks;

namespace CoreTests
{
    [TestFixture]
    public class SimulationHandlerTests
    {
        [Test]
        public void SimulationPropertySetterTest()
        {
            var handler = new MySimulationHandler(new BackgroundWorker());

            var simulation = MockRepository.GenerateStub<MySimulation>();
            // This should not throw, it's the first simulation.
            handler.Simulation = simulation;

            Assert.Throws<InvalidOperationException>(() => handler.Simulation = simulation);
        }
    }
}
