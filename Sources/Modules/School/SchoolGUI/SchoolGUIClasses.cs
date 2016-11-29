using Aga.Controls.Tree;
using GoodAI.Modules.School.Common;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using GoodAI.School.Common;

namespace GoodAI.School.GUI
{
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
}
