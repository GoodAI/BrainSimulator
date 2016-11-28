using Aga.Controls.Tree;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using YAXLib;

namespace GoodAI.School.GUI
{
    #region UI classes

    public class CurriculumNode : Node
    {
        public CurriculumNode()
        {
            IsChecked = true;
        }

        public string Description { get; set; }

        public static CurriculumNode FromCurriculumDesign(CurriculumDesign design)
        {
            CurriculumNode node = new CurriculumNode { Text = design.Name, IsChecked = design.Enabled, Description = design.Description };
            design.Tasks.Where(x => LearningTaskNode.FromLTDesign(x) != null).ToList().ForEach(x => node.Nodes.Add(LearningTaskNode.FromLTDesign(x)));
            return node;
        }

        public static List<CurriculumNode> FromPlanDesign(PlanDesign design)
        {
            return design.Curricula.Select(FromCurriculumDesign).ToList();
        }

        public static PlanDesign ToPlanDesign(List<CurriculumNode> nodes)
        {
            List<CurriculumDesign> curricula = nodes.Select(x => (CurriculumDesign)x).ToList();
            return new PlanDesign(curricula);
        }

        public static explicit operator CurriculumDesign(CurriculumNode node)
        {
            List<LTDesign> tasks = node.Nodes.
                Where(x => x is LearningTaskNode).
                Select(x => (LTDesign)(x as LearningTaskNode)).
                ToList();
            bool isEnabled = node.IsChecked;
            string name = node.Text;
            string description = node.Description;
            return new CurriculumDesign(tasks, isEnabled, name, description);
        }
    }

    public class LearningTaskNode : Node
    {
        public Type TaskType { get; private set; }
        public Type WorldType { get; private set; }

        public bool IsActive { get; set; }
        public int Progress { get; set; }
        public TrainingResult Status { get; set; }
        public int Steps { get; set; }
        public float Time { get; set; }

        public override string Text
        {
            get
            {
                DisplayNameAttribute displayNameAttTask = TaskType.GetCustomAttributes(typeof(DisplayNameAttribute), true).FirstOrDefault() as DisplayNameAttribute;
                DisplayNameAttribute displayNameAttWorld = WorldType.GetCustomAttributes(typeof(DisplayNameAttribute), true).FirstOrDefault() as DisplayNameAttribute;

                string taskDisplayName = displayNameAttTask != null ? displayNameAttTask.DisplayName : TaskType.Name;
                string worldDisplayName = displayNameAttWorld != null ? displayNameAttWorld.DisplayName : WorldType.Name;

                return taskDisplayName + " (" + worldDisplayName + ")";
            }
        }

        public LearningTaskNode(Type taskType, Type worldType)
        {
            TaskType = taskType;
            WorldType = worldType;
            IsChecked = true;
        }

        public static LearningTaskNode FromLTDesign(LTDesign design)
        {
            Type taskType = Type.GetType(design.TaskType);
            Type worldType = Type.GetType(design.WorldType);
            if (taskType == null || worldType == null)  //unable to reconstruct types from serialized strings
                return null;
            return new LearningTaskNode(taskType, worldType) { IsChecked = design.Enabled };
        }

        public static explicit operator LTDesign(LearningTaskNode node)
        {
            string taskType = node.TaskType.AssemblyQualifiedName;
            string worldType = node.WorldType.AssemblyQualifiedName;
            bool isEnabled = node.IsChecked;
            return new LTDesign(taskType, worldType, isEnabled);
        }

        public override bool Equals(object obj)
        {
            if (!(obj is LearningTaskNode))
                return false;

            return TaskType == ((LearningTaskNode)obj).TaskType && WorldType == ((LearningTaskNode)obj).WorldType;
        }
    }

    public class LevelNode
    {
        public int Level { get; set; }

        public string Text
        {
            get
            {
                return "Level " + Level;
            }
        }

        public LevelNode(int level)
        {
            Level = level;
        }
    }

    public class AttributeNode
    {
        private readonly string m_annotation;
        public string Name { get; set; }
        public string Value { get; set; }
        private Type Type { get; set; }

        public AttributeNode(string name)
        {
            Name = name;
        }

        public AttributeNode(TSHintAttribute attribute, float value)
        {
            Name = attribute.Name;

            Type = attribute.TypeOfValue;
            if (Type == typeof(Single) || Type == typeof(Double))
                Value = value.ToString("F");
            else
                Value = Convert.ChangeType(value, Type).ToString();

            m_annotation = attribute.Annotation;
        }

        public override bool Equals(object obj)
        {
            return Name == (obj as AttributeNode).Name;
        }

        // Annotation is not public property because DataGridView automaticly generates columns for
        // all public properties, while this should not be column
        public string GetAnotation()
        {
            return m_annotation;
        }
    }

    #endregion UI classes

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
}
