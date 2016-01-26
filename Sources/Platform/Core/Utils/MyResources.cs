using System;
using System.Drawing;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
using GoodAI.Core.Configuration;

namespace GoodAI.Core.Utils
{
    public static class MyResources
    {
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static string GetMyAssemblyPath()
        {
            return GetAssemblyDirectory(Assembly.GetCallingAssembly());
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static string GetEntryAssemblyPath()
        {
            // Static initialization must not crash when called from tests!
            Assembly assembly = Assembly.GetEntryAssembly() ?? Assembly.GetCallingAssembly();

            return GetAssemblyDirectory(assembly);
        }

        /// <summary>
        /// This works even if the assembly is run from a temporary directory (e.g. in unit tests)
        /// </summary>
        /// @see http://stackoverflow.com/questions/52797/how-do-i-get-the-path-of-the-assembly-the-code-is-in
        public static string GetAssemblyDirectory(Assembly assembly)
        {
            var uri = new UriBuilder(assembly.CodeBase);

            return Path.GetDirectoryName(Uri.UnescapeDataString(uri.Path));
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

        public static bool TryGetTextFromAssembly(Assembly assembly, string resourceName, out string content,
            string resourceDir = "conf")
        {
            try
            {
                using (
                    Stream stream =
                        assembly.GetManifestResourceStream(assembly.GetName().Name + "." + resourceDir + "." +
                                                           resourceName))
                using (StreamReader reader = new StreamReader(stream))
                {
                    content = reader.ReadToEnd();
                }
            }
            catch
            {
                content = string.Empty;
                return false;
            }

            return true;
        }

        public static string GetTextFromAssembly(Assembly assembly, string resourceName, string resourceDir = "conf")
        {
            string content;
            // Moved the log from warning to debug - the dll could contain a UI extension.
            if (!TryGetTextFromAssembly(assembly, resourceName, out content, resourceDir))
                MyLog.DEBUG.WriteLine("Text resource '" + resourceName + "' load failed.");

            return content;
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