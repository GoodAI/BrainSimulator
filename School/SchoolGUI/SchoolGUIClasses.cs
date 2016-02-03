using Aga.Controls.Tree;
using GoodAI.Modules.School.Common;
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
        public SchoolTreeNode() { Enabled = true; }
    }

    public class CurriculumNode : SchoolTreeNode { }

    public class LearningTaskNode : SchoolTreeNode
    {
        public Type TaskType { get; private set; }
        public Type WorldType { get; private set; }
        // for data grid
        public int Steps { get; set; }
        public float Time { get; set; }
        public string Status { get; set; }

        public LearningTaskNode(Type taskType, Type worldType)
        {
            TaskType = taskType;
            WorldType = worldType;
        }

        public override string Text
        {
            get
            {
                return TaskType.Name + " (" + WorldType.Name + ")";
            }
        }
    }

    #endregion

    // mediator between view (CurriculumNode) and model (SchoolCurriculum) - is also used for serialization
    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AllFields)]
    public class PlanDesign
    {
        [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AllFields)]
        private class LTDesign
        {
            [YAXSerializeAs("Enabled"), YAXAttributeForClass]
            private readonly bool m_enabled;
            [YAXSerializeAs("TaskType")]
            private readonly string m_taskType;
            [YAXSerializeAs("WorldType")]
            private readonly string m_worldType;

            public LTDesign() { }

            public LTDesign(LearningTaskNode node)
            {
                m_taskType = node.TaskType.AssemblyQualifiedName;
                m_worldType = node.WorldType.AssemblyQualifiedName;
                m_enabled = node.Enabled;
            }

            public static explicit operator LearningTaskNode(LTDesign design)
            {
                LearningTaskNode node = new LearningTaskNode(Type.GetType(design.m_taskType), Type.GetType(design.m_worldType));
                node.Enabled = design.m_enabled;
                return node;
            }
        }

        [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AllFields)]
        private class CurriculumDesign
        {
            [YAXSerializeAs("Tasks")]
            private readonly List<LTDesign> m_tasks;
            [YAXSerializeAs("Enabled"), YAXAttributeForClass]
            private readonly bool m_enabled;
            [YAXSerializeAs("Name"), YAXAttributeForClass]
            private readonly string m_name;

            public CurriculumDesign() { }

            public CurriculumDesign(CurriculumNode node)
            {
                m_tasks = node.Nodes.
                    Where(x => x is LearningTaskNode).
                    Select(x => new LTDesign(x as LearningTaskNode)).
                    ToList();
                m_enabled = node.Enabled;
                m_name = node.Text;
            }

            public static explicit operator CurriculumNode(CurriculumDesign design)
            {
                CurriculumNode node = new CurriculumNode();
                node.Text = design.m_name;
                node.Enabled = design.m_enabled;

                design.m_tasks.ForEach(x => node.Nodes.Add((LearningTaskNode)x));

                return node;
            }
        }

        [YAXSerializeAs("Curricula")]
        private List<CurriculumDesign> m_curricula;

        public PlanDesign() { }

        public PlanDesign(List<CurriculumNode> nodes)
        {
            m_curricula = nodes.Select(x => new CurriculumDesign(x)).ToList();
        }

        public static explicit operator List<CurriculumNode>(PlanDesign design)
        {
            return design.m_curricula.Select(x => (CurriculumNode)x).ToList();
        }
    }
}
