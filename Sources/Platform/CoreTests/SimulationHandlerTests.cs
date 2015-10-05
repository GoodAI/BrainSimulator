using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Execution;
using Rhino.Mocks;
using Xunit;

namespace CoreTests
{
    public class SimulationHandlerTests
    {
        [Fact]
        public void SimulationPropertySetterTest()
        {
            var simulation = MockRepository.GenerateStub<MySimulation>();
            var handler = new MySimulationHandler(simulation);

            // This should not throw, it's the first simulation.

            Assert.Throws<InvalidOperationException>(() => handler.Simulation = simulation);
        }
    }
}
