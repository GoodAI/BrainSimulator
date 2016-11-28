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
    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AllFields)]
    public class LTDesign
    {
        [YAXSerializeAs("Enabled"), YAXAttributeForClass, YAXSerializableField(DefaultValue = true)]
        public bool Enabled { get; private set; }

        [YAXSerializeAs("TaskType"), YAXSerializableField(DefaultValue = "")]
        public string TaskType { get; private set; }

        [YAXSerializeAs("WorldType"), YAXSerializableField(DefaultValue = "")]
        public string WorldType { get; private set; }

        public LTDesign()
        {
        }

        public LTDesign(string taskType, string worldType, bool isEnabled)
        {
            TaskType = taskType;
            WorldType = worldType;
            Enabled = isEnabled;
        }

        public ILearningTask AsILearningTask(SchoolWorld world = null)
        {
            if (!Enabled)
                return null;    //there is no placeholder for empty task, therefore null
            ILearningTask task;
            if (world != null)
                task = LearningTaskFactory.CreateLearningTask(Type.GetType(TaskType), world);
            else
                task = LearningTaskFactory.CreateLearningTask(Type.GetType(TaskType));
            task.RequiredWorldType = Type.GetType(WorldType);
            return task;
        }
    }

    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AllFields)]
    public class CurriculumDesign
    {
        [YAXSerializeAs("Tasks")]
        public List<LTDesign> Tasks { get; private set; }

        [YAXSerializeAs("Enabled"), YAXAttributeForClass, YAXSerializableField(DefaultValue = true)]
        public bool Enabled { get; private set; }

        [YAXSerializeAs("Name"), YAXAttributeForClass, YAXSerializableField(DefaultValue = "")]
        public string Name { get; private set; }

        [YAXSerializeAs("Description"), YAXSerializableField(DefaultValue = "")]
        public string Description { get; private set; }

        public CurriculumDesign(List<LTDesign> tasks, bool isEnabled, string name, string description)
        {
            Tasks = tasks;
            Enabled = isEnabled;
            Name = name;
            Description = description;
        }

        public static explicit operator SchoolCurriculum(CurriculumDesign design)
        {
            SchoolCurriculum curriculum = new SchoolCurriculum();
            if (!design.Enabled)
                return curriculum;

            design.Tasks.
                Select(x => x.AsILearningTask()).
                Where(x => x != null).
                ToList().
                ForEach(x => curriculum.Add(x));

            return curriculum;
        }

        public SchoolCurriculum AsSchoolCurriculum(SchoolWorld world)
        {
            SchoolCurriculum curriculum = new SchoolCurriculum();
            if (!Enabled)
                return curriculum;

            Tasks.
                Select(x => x.AsILearningTask(world)).
                Where(x => x != null).
                ToList().
                ForEach(x => curriculum.Add(x));

            return curriculum;
        }
    }

    // mediator between view (CurriculumNode) and model (SchoolCurriculum) - is also used for serialization
    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AllFields)]
    public class PlanDesign
    {
        [YAXSerializeAs("Curricula")]
        public List<CurriculumDesign> Curricula { get; set; }

        public PlanDesign(List<CurriculumDesign> curricula)
        {
            Curricula = curricula;
        }

        public static explicit operator SchoolCurriculum(PlanDesign design)
        {
            SchoolCurriculum result = new SchoolCurriculum();
            foreach (CurriculumDesign curr in design.Curricula)
                result.Add((SchoolCurriculum)curr);

            return result;
        }

        public SchoolCurriculum AsSchoolCurriculum(SchoolWorld world)
        {
            SchoolCurriculum result = new SchoolCurriculum();
            foreach (CurriculumDesign curr in Curricula)
                result.Add(curr.AsSchoolCurriculum(world));

            return result;
        }
    }

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
                MyLog.WARNING.WriteLine("Unable to read file " + filePath);
                return null;
            }

            try
            {
                PlanDesign plan = (PlanDesign)serializer.Deserialize(xmlCurr);
                return plan;
            }
            catch (YAXException e)
            {
                MyLog.WARNING.WriteLine("Unable to deserialize data from " + filePath);
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
