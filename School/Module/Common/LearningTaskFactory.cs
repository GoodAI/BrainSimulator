using GoodAI.Modules.School.Worlds;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace GoodAI.Modules.School.Common
{
    public enum AbilityNameEnum
    {
        SimplestPatternDetection,
    }

    public class LearningTaskFactory
    {
        private static Dictionary<Type, List<Type>> m_knownLearningTasks = null;

        // Dictionary with all known LTs and worlds they support
        public static Dictionary<Type, List<Type>> KnownLearningTasks
        {
            get
            {
                if (m_knownLearningTasks == null)
                {
                    //TODO: check multi-level inheritance if there will be any in future
                    IEnumerable<Type> tasks = Assembly.GetAssembly(typeof(AbstractLearningTask<>))
                        .GetTypes()
                        .Where(x => x.BaseType != null &&
                            x.BaseType.IsGenericType &&
                            x.BaseType.GetGenericTypeDefinition() == typeof(AbstractLearningTask<>));

                    m_knownLearningTasks = new Dictionary<Type, List<Type>>();
                    foreach (Type taskType in tasks)
                    {
                        List<Type> supportedWorlds = GetSupportedWorlds(taskType);
                        m_knownLearningTasks.Add(taskType, supportedWorlds);
                    }
                }

                return m_knownLearningTasks;
            }
        }

        public static ILearningTask CreateLearningTask(Type learningTaskType, SchoolWorld w)
        {
            //ConstructorInfo c = learningTaskType.GetConstructor(new[] { typeof(SchoolWorld) });
            //return (ILearningTask)c.Invoke(new[] { w });
            return (ILearningTask)Activator.CreateInstance(learningTaskType, w);
        }

        public static ILearningTask CreateLearningTask(Type taskType, Type worldType)
        {
            // check if task type is valid
            if (!KnownLearningTasks.ContainsKey(taskType))
                return null;

            // check if the world is valid for this task
            if (!KnownLearningTasks[taskType].Contains(worldType))
                return null;

            // everything is OK - create the task
            SchoolWorld world = (SchoolWorld)Activator.CreateInstance(worldType);
            ILearningTask task = (ILearningTask)Activator.CreateInstance(taskType, world);
            return task;
        }

        public static Type GetGenericType(Type taskType)
        {
            Type baseClass = taskType;
            do
            {
                baseClass = baseClass.BaseType;
            }
            while (baseClass.GetGenericArguments().Length == 0);

            return baseClass.GetGenericArguments()[0];
        }

        public static List<Type> GetSupportedWorlds(Type taskType)
        {
            //TODO: check multi-level inheritance if there will be any in future
            // get generic parameter
            Type genericType = GetGenericType(taskType);
            // look up all derived classes of this type
            List<Type> results = Assembly.GetAssembly(typeof(SchoolWorld))
                .GetTypes()
                .Where(x => x.BaseType == genericType)
                .ToList();
            results.Add(genericType);
            // remove abstract classes
            results = results.Where(x => !x.IsAbstract).ToList();

            return results;
        }
    }
}
