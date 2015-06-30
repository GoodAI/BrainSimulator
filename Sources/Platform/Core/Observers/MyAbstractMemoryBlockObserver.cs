using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Core.Observers
{
    public abstract class MyAbstractMemoryBlockObserver : MyObserver<MyAbstractMemoryBlock>
    {
        public override string GetTargetName(MyNode declaredOwner)
        {
            if (declaredOwner == Target.Owner)
            {
                return Target.Owner.Name + " - " + Target.Name;
            }
            else
            {
                return declaredOwner.Name + " (" + Target.Owner.Name + ") - " + Target.Name;
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
