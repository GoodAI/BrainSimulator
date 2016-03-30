using World.ToyWorldCore;
using Xunit;

namespace ToyWorldTests.World
{
    class AtlasTests
    {
        private Atlas m_atlas;
        public AtlasTests()
        {
            m_atlas = new Atlas();
        }

        [Fact]
        public void TestGetLayer()
        {

            m_atlas.GetLayer(LayerType.Background);
        }
    }
}
