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
            var simulation = MockRepository.GenerateStub<MySimulation>();
            var handler = new MySimulationHandler(simulation)
            {
                Simulation = simulation
            };

            // This should not throw, it's the first simulation.

            Assert.Throws<InvalidOperationException>(() => handler.Simulation = simulation);
        }
    }
}
