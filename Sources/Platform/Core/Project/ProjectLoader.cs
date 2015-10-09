using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using GoodAI.Core.Utils;

namespace GoodAI.Core.Project
{
    // This does not work with MyProject types, only with strings.
    // TODO(Premek): Work with MyProject types
    public static class ProjectLoader
    {
        private const string BrainFileName = "Project.brain";
    
        public static void SaveProject(string fileName, string fileContent, string dataStoragePath)
        {
            if (fileName == null) throw new ArgumentNullException("fileName");
            if (fileContent == null) throw new ArgumentNullException("fileContent");

            if (fileName.EndsWith(".brainz"))
            {
                if (dataStoragePath == null)
                    throw new ArgumentNullException("dataStoragePath", "required for .brainz");

                if (!Directory.Exists(dataStoragePath))
                {
                    MyLog.WARNING.WriteLine("No state data found, saving only zipped project file.");
                    Directory.CreateDirectory(dataStoragePath);
                }

                File.WriteAllText(Path.Combine(dataStoragePath, BrainFileName), fileContent);

                if (File.Exists(fileName))
                {
                    File.Delete(fileName);
                }

                ZipFile.CreateFromDirectory(dataStoragePath, fileName, CompressionLevel.Optimal, false);
            }
            else // We are saving just the project definition aka .brain file
            {
                File.WriteAllText(fileName, fileContent);
            }
        }

        public static string LoadProject(string fileName, string dataStoragePath)
        {
            if (fileName == null) throw new ArgumentNullException("fileName");

            string brainFileName = fileName;

            if (fileName.EndsWith(".brainz"))
            {
                if (dataStoragePath == null)
                    throw new ArgumentNullException("dataStoragePath", "required for .brainz");

                if (Directory.Exists(dataStoragePath))
                {
                    Directory.Delete(dataStoragePath, true);
                }

                ZipFile.ExtractToDirectory(fileName, dataStoragePath);
                brainFileName = Path.Combine(dataStoragePath, BrainFileName);
            }

            return File.ReadAllText(brainFileName);
        }
    }
}
