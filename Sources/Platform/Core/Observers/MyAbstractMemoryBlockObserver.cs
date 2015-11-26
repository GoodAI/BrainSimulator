using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using System;

namespace GoodAI.Core.Observers
{
    public abstract class MyAbstractMemoryBlockObserver : MyObserver<MyAbstractMemoryBlock>
    {
        private int m_lastCount = 0;

        protected override void PrepareExecution()
        {
            if (Target.Count != m_lastCount)
            {
                OnCountChanged();
                m_lastCount = Target.Count;
            }
        }

        protected virtual void OnCountChanged()
        {
            Reset();
        }

        public override string GetTargetName(MyNode declaredOwner)
        {
            if (declaredOwner == Target.Owner)
            {
                return Target.Owner.Name + ": " + Target.Name;
            }
            else
            {
                return declaredOwner.Name + " (" + Target.Owner.Name + "): " + Target.Name;
            }
        }

        protected override string CreateTargetIdentifier()
        {
            if (Target != null)
            {
                return Target.Owner.Id + "#" + Target.Name;
            }
            else return String.Empty;
        }

        public override void RestoreTargetFromIdentifier(MyProject project)
        {
            if (TargetIdentifier != null)
            {
                string[] split = TargetIdentifier.Split('#');
                if (split.Length == 2)
                {
                    MyWorkingNode node = (MyWorkingNode)project.GetNodeById(int.Parse(split[0]));

                    if (node != null)
                    {
                        Target = MyMemoryManager.Instance.GetMemoryBlockByName(node, split[1]);
                    }
                }
            }
        }
    }
}
