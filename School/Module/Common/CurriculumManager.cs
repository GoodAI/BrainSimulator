using System;
using System.Collections.Generic;
using System.Linq;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;

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
                if (ContainsType(worldTypes, worldType) && worldTypes.Contains(worldType))
                    result.Add(learningTaskType);
            }

            return result;
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
