using GoodAI.BasicNodes.Harm.Obsrvers;
using GoodAI.Core;
using GoodAI.Core.Observers;
using GoodAI.Core.Utils;
using GoodAI.Modules.Harm;
using ManagedCuda;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using System.ComponentModel;
using System.Drawing;
using YAXLib;

namespace GoodAI.Modules.Observers
{
    /// <author>GoodAI</author>
    /// <meta>df,jv</meta>
    /// <status>Working</status>
    /// <summary>
    /// Observes valdata stored in the QMatrix.
    /// </summary>
    /// <typeparam name="T">Node which uses DiscreteQLearnin to be observed</typeparam>
    public abstract class MyAbstractQLearningObserver<T> : AbstractPolicyLearnerObserver<T> where T : MyAbstractDiscreteQLearningNode
    {
        [MyBrowsable, Category("Mode"),
        Description("Observe utility colors which are already scaled by the current motivation.")]
        [YAXSerializableField(DefaultValue = true)]
        public bool ShowCurrentMotivations { get; set; }
    }
}
