using Aga.Controls.Tree;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.Collections.Generic;
using System.Linq;
using YAXLib;

namespace GoodAI.School.GUI
{
    #region UI classes

    public class SchoolTreeNode : Node
    {
        public bool Enabled { get; set; }

        public SchoolTreeNode()
        {
            Enabled = true;
        }
    }

    public class CurriculumNode : SchoolTreeNode
    {
        public string Description { get; set; }
    }

    public class LearningTaskNode : SchoolTreeNode
    {
        public Type TaskType { get; private set; }
        public Type WorldType { get; private set; }
        public int Steps { get; set; }
        public float Time { get; set; }
        public string Status { get; set; }

        public override string Text
        {
            get
            {
                return TaskType.Name + " (" + WorldType.Name + ")";
            }
        }

        public LearningTaskNode(Type taskType, Type worldType)
        {
            TaskType = taskType;
            WorldType = worldType;
        }
    }

    #endregion UI classes

    // mediator between view (CurriculumNode) and model (SchoolCurriculum) - is also used for serialization
    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AllFields)]
    public class PlanDesign
    {
        [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AllFields)]
        private class LTDesign
        {
            [YAXSerializeAs("Enabled"), YAXAttributeForClass, YAXSerializableField(DefaultValue = true)]
            private readonly bool m_enabled;

            [YAXSerializeAs("TaskType"), YAXSerializableField(DefaultValue = "")]
            private readonly string m_taskType;

            [YAXSerializeAs("WorldType"), YAXSerializableField(DefaultValue = "")]
            private readonly string m_worldType;

            public LTDesign()
            {
            }

            public LTDesign(LearningTaskNode node)
            {
                m_taskType = node.TaskType.AssemblyQualifiedName;
                m_worldType = node.WorldType.AssemblyQualifiedName;
                m_enabled = node.Enabled;
            }

            public static explicit operator LearningTaskNode(LTDesign design)
            {
                Type taskType = Type.GetType(design.m_taskType);
                Type worldType = Type.GetType(design.m_worldType);
                if (taskType == null || worldType == null)  //unable to reconstruct types from serialized strings
                    return null;
                return new LearningTaskNode(taskType, worldType) { Enabled = design.m_enabled };
            }

            public ILearningTask AsILearningTask(SchoolWorld world = null)
            {
                if (!m_enabled)
                    return null;    //there is no placeholder for empty task, therefore null
                ILearningTask task;
                if (world != null)
                    task = LearningTaskFactory.CreateLearningTask(Type.GetType(m_taskType), world);
                else
                    task = LearningTaskFactory.CreateLearningTask(Type.GetType(m_taskType));
                task.RequiredWorld = Type.GetType(m_worldType);
                return task;
            }
        }

        [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AllFields)]
        private class CurriculumDesign
        {
            [YAXSerializeAs("Tasks")]
            private readonly List<LTDesign> m_tasks;

            [YAXSerializeAs("Enabled"), YAXAttributeForClass, YAXSerializableField(DefaultValue = true)]
            private readonly bool m_enabled;

            [YAXSerializeAs("Name"), YAXAttributeForClass, YAXSerializableField(DefaultValue = "")]
            private readonly string m_name;

            [YAXSerializeAs("Description"), YAXSerializableField(DefaultValue = "")]
            private readonly string m_description;

            public CurriculumDesign()
            {
            }

            public CurriculumDesign(CurriculumNode node)
            {
                m_tasks = node.Nodes.
                    Where(x => x is LearningTaskNode).
                    Select(x => new LTDesign(x as LearningTaskNode)).
                    ToList();
                m_enabled = node.Enabled;
                m_name = node.Text;
                m_description = node.Description;
            }

            public static explicit operator CurriculumNode(CurriculumDesign design)
            {
                CurriculumNode node = new CurriculumNode { Text = design.m_name, Enabled = design.m_enabled, Description = design.m_description };

                design.m_tasks.Where(x => (LearningTaskNode)x != null).ToList().ForEach(x => node.Nodes.Add((LearningTaskNode)x));

                return node;
            }

            public static explicit operator SchoolCurriculum(CurriculumDesign design)
            {
                SchoolCurriculum curriculum = new SchoolCurriculum();
                if (!design.m_enabled)
                    return curriculum;

                design.m_tasks.
                    Select(x => x.AsILearningTask()).
                    Where(x => x != null).
                    ToList().
                    ForEach(x => curriculum.Add(x));

                return curriculum;
            }

            public SchoolCurriculum AsSchoolCurriculum(SchoolWorld world)
            {
                SchoolCurriculum curriculum = new SchoolCurriculum();
                if (!m_enabled)
                    return curriculum;

                m_tasks.
                    Select(x => x.AsILearningTask(world)).
                    Where(x => x != null).
                    ToList().
                    ForEach(x => curriculum.Add(x));

                return curriculum;
            }
        }

        [YAXSerializeAs("Curricula")]
        private List<CurriculumDesign> m_curricula;

        public PlanDesign()
        {
        }

        public PlanDesign(List<CurriculumNode> nodes)
        {
            m_curricula = nodes.Select(x => new CurriculumDesign(x)).ToList();
        }

        public static explicit operator List<CurriculumNode>(PlanDesign design)
        {
            return design.m_curricula.Select(x => (CurriculumNode)x).ToList();
        }

        public static explicit operator SchoolCurriculum(PlanDesign design)
        {
            SchoolCurriculum result = new SchoolCurriculum();
            foreach (CurriculumDesign curr in design.m_curricula)
                result.Add((SchoolCurriculum)curr);

            return result;
        }

        public SchoolCurriculum AsSchoolCurriculum(SchoolWorld world)
        {
            SchoolCurriculum result = new SchoolCurriculum();
            foreach (CurriculumDesign curr in m_curricula)
                result.Add(curr.AsSchoolCurriculum(world));

            return result;
        }
    }
}
