using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Project;
using Xunit;

namespace CoreTests
{
    public class ProjectLoaderTests
    {
        private static string GetTempFileNameWithExt(string extension)
        {
            return Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString()) + extension;
        }

        private static string GetDataPath()
        {
            return Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        }

        [Fact]
        public void SavesSomething()
        {
            string tempFileName = GetTempFileNameWithExt(".brain");

            ProjectLoader.SaveProject(tempFileName, "foobar", null);

            Assert.True(File.Exists(tempFileName));
        }

        [Fact]
        public void SavesSomethingZipped()
        {
            string tempFileName = GetTempFileNameWithExt(".brainz");

            ProjectLoader.SaveProject(tempFileName, "foobar", GetDataPath());

            Assert.True(File.Exists(tempFileName));            
        }

        [Fact]
        public void SaveRequiresThirdParamForBrainz()
        {
            string tempFileName = GetTempFileNameWithExt(".brainz");

            Assert.Throws<ArgumentNullException>(() =>
                ProjectLoader.SaveProject(tempFileName, "foobar", null));
        }

        [Fact]
        public void LoadsSomething()
        {
            string tempFileName = GetTempFileNameWithExt(".brain");
            File.WriteAllText(tempFileName, "something");

            string content = ProjectLoader.LoadProject(tempFileName, dataStoragePath: null);

            Assert.Equal("something", content);
        }

        [Fact]
        public void SavesAndLoadsZippedBrain()
        {
            string tempFileName = GetTempFileNameWithExt(".brainz");

            ProjectLoader.SaveProject(tempFileName, "something_zipped", GetDataPath());

            string content = ProjectLoader.LoadProject(tempFileName, GetDataPath());

            Assert.Equal("something_zipped", content);
        }
    }
}
