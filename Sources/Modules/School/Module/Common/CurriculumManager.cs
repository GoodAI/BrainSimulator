using System;
using System.Collections.Generic;
using System.Linq;
using GoodAI.Modules.School.Worlds;
using System.ComponentModel;
using System.IO;
using GoodAI.Core.Utils;
using GoodAI.Modules.School.Common;
using YAXLib;

namespace GoodAI.School.Common
{
    public class CurriculumManager
    {
        public static IEnumerable<Type> GetAvailableWorlds()
        {
            Type interfaceType = typeof(IWorldAdapter);
            IEnumerable<Type> types = AppDomain.CurrentDomain.GetAssemblies()
                .SelectMany(s => s.GetTypes())
                .Where(p => interfaceType.IsAssignableFrom(p)
                    && !p.IsInterface
                    && !p.IsAbstract
                    && WorldIsAdmitted(p));

            return types;
        }

        public static List<Type> GetTasksForWorld(Type worldType)
        {
            List<Type> result = new List<Type>();
            foreach (KeyValuePair<Type, List<Type>> entry in LearningTaskFactory.KnownLearningTasks)
            {
                Type learningTaskType = entry.Key;
                List<Type> worldTypes = entry.Value;

                bool isBrowsable = true;
                object[] browsableAttrs = learningTaskType.GetCustomAttributes(typeof(BrowsableAttribute), true);
                if (browsableAttrs.Length > 0)
                    isBrowsable = (browsableAttrs[0] as BrowsableAttribute).Browsable;


                if (ContainsType(worldTypes, worldType) && worldTypes.Contains(worldType) && isBrowsable)
                    result.Add(learningTaskType);


            }

            return result;
        }

        public static void SavePlanDesign(PlanDesign design, string filePath)
        {
            string dummy;
            SavePlanDesign(design, filePath, out dummy);
        }

        public static void SavePlanDesign(PlanDesign design, string filePath, out string xmlResult)
        {
            YAXSerializer serializer = new YAXSerializer(typeof(PlanDesign));

            xmlResult = serializer.Serialize(design);
            File.WriteAllText(filePath, xmlResult);
            MyLog.Writer.WriteLine(MyLogLevel.INFO, "School project saved to: " + filePath);
        }

        public static PlanDesign LoadPlanDesign(string filePath)
        {
            string dummy;
            return LoadPlanDesign(filePath, out dummy);
        }

        public static PlanDesign LoadPlanDesign(string filePath, out string xmlCurr)
        {
            xmlCurr = null;

            YAXSerializer serializer = new YAXSerializer(typeof(PlanDesign));
            if (string.IsNullOrEmpty(filePath))
                return null;

            try { xmlCurr = File.ReadAllText(filePath); }
            catch (IOException e)
            {
                MyLog.WARNING.WriteLine("Unable to read file " + filePath + " during curriculum loading. " + e.Message);
                return null;
            }

            try
            {
                PlanDesign plan = (PlanDesign)serializer.Deserialize(xmlCurr);
                return plan;
            }
            catch (YAXException e)
            {
                MyLog.WARNING.WriteLine("Unable to deserialize data from " + filePath + " during curriculum loading. " + e.Message);
                return null;
            }
        }

        private static bool ContainsType(List<Type> worldTypes, Type selectedWorldType)
        {
            return worldTypes.Any(type => selectedWorldType == type || selectedWorldType.IsSubclassOf(type));
        }

        // True if the world should appear in the GUI
        // Used to suppress worlds that should not be used
        private static bool WorldIsAdmitted(Type p)
        {
            return p != typeof(PlumberWorld);
        }
    }
}
