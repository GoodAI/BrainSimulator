using System;
using System.Drawing;
using System.IO;
using System.Reflection;

namespace GoodAI.Core.Utils
{
    public static class MyResources
    {
        public static string GetMyAssemblyPath()
        {            
            string location = Assembly.GetCallingAssembly().Location;
            return location.Substring(0, location.LastIndexOf(Path.DirectorySeparatorChar));
        }

        public static string GetEntryAssemblyPath()
        {
            // Static initialization must not crash when called from tests!
            Assembly assembly = Assembly.GetEntryAssembly() ?? Assembly.GetCallingAssembly();

            string location = assembly.Location;
            return location.Substring(0, location.LastIndexOf(Path.DirectorySeparatorChar));
        }

        public static string PathToResourceName(string path)
        {
            return String.Join(".", path.Split(new[] { Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar }, StringSplitOptions.RemoveEmptyEntries));
        }

        public static Image GetImage(string resourceName, string resourceDir = "res")
        {
            return GetImageFromAssembly(Assembly.GetExecutingAssembly(), resourceName, resourceDir);
        }

        public static Image GetImageFromAssembly(Assembly assembly, string resourceName, string resourceDir = "res")
        {
            try
            {
                Image im;
                using (Stream stream = assembly.GetManifestResourceStream(assembly.GetName().Name + "." + resourceDir + "." + resourceName))
                {
                    im = System.Drawing.Image.FromStream(stream, true);
                }

                return im;
            }
            catch (Exception e)
            {
                MyLog.WARNING.WriteLine("Image resource '" + resourceName + "' load failed from assembly " + assembly.GetName().Name + "."); 
                return null;
            }
        }

        public static string GetTextFromAssembly(string resourceName, string resourceDir = "conf")
        {
            return GetTextFromAssembly(Assembly.GetExecutingAssembly(), resourceName, resourceDir);
        }

        public static string GetTextFromAssembly(Assembly assembly, string resourceName, string resourceDir = "conf")
        {
            try
            {
                string content;
//                var l = assembly.GetManifestResourceNames();
                using (Stream stream = assembly.GetManifestResourceStream(assembly.GetName().Name + "." + resourceDir + "." + resourceName))
                using (StreamReader reader = new StreamReader(stream))
                {
                    content = reader.ReadToEnd();
                }
                return content;
            }
            catch (Exception e)
            {
                MyLog.WARNING.WriteLine("Text resource '" + resourceName + "' load failed.");
                return string.Empty;
            }
        }

        public static Stream GetPTXStream(string resourceName, string resourceDir = "ptx")
        {
            return GetPTXStreamFromAssembly(Assembly.GetExecutingAssembly(), resourceName, resourceDir);
        }

        public static Stream GetPTXStreamFromAssembly(Assembly assembly, string resourceName, string resourceDir = "ptx")
        {
            try

            {
                return assembly.GetManifestResourceStream(assembly.GetName().Name + "." + resourceDir + "." + PathToResourceName(resourceName) + ".ptx");
            }
            catch (Exception e)
            {
                MyLog.ERROR.WriteLine("PTX resource '" + resourceName + "' load failed from assembly " + assembly.GetName().Name + ".");
                return null;
            }                
        }
    }
}