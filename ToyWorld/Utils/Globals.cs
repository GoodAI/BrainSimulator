using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Utils
{
    public static class Globals
    {
        // Name of the application (NOTE that Utils, World, Rendering and Runner 
        // should be part of the ENGINE, which could be used by different Games....)
        public const string AppName = "Toy World";
        public const string TestFileLocation = @".\TestFiles\";

        public static string GetDllDirectory()
        {
            return System.IO.Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
        }

    }
}
