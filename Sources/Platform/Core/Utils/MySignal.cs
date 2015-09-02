using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;

namespace GoodAI.Core.Signals
{
    public abstract class MySignal
    {
        private static Dictionary<Type, long> SIGNAL_TABLE;
        private static Dictionary<string, Type> SIGNAL_NAMES;
        
        static MySignal() 
        {
            SIGNAL_TABLE = new Dictionary<Type, long>();
            SIGNAL_NAMES = new Dictionary<string, Type>();            
        }

        public string DefaultName
        {
            get { return MyProject.ShortenNodeTypeName(GetType()); }
        }

        public static MySignal CreateSignalByDefaultName(string defaultName) 
        {
            if (SIGNAL_NAMES.ContainsKey(defaultName))
            {
                return (MySignal)Activator.CreateInstance(SIGNAL_NAMES[defaultName]);
            }
            else
            {
                MyLog.WARNING.WriteLine("Unknown signal: " + defaultName);
                return null;
            }
        }

        protected MySignal()
        {
            Type myType = GetType();

            if (!SIGNAL_TABLE.ContainsKey(myType))
            {
                if (SIGNAL_TABLE.Count < 64)
                {
                    SIGNAL_TABLE[myType] = 1 << SIGNAL_TABLE.Count;

                    if (!SIGNAL_NAMES.ContainsKey(DefaultName))
                    {
                        SIGNAL_NAMES.Add(DefaultName, myType);
                    }
                }
                else
                {
                    throw new InvalidOperationException("Too many signal types in the system");
                }
            }

            m_mask = SIGNAL_TABLE[myType];
        }

        public MyNode Owner { get; set; }
        public string Name { get; set; }

        private long m_mask;

        public virtual long Mask 
        {
            get { return m_mask; }
        }

        public void Raise()
        {
            Owner.RiseSignalMask |= Mask;
            Owner.DropSignalMask &= ~Mask;
        }

        public void Drop()
        {
            Owner.DropSignalMask |= Mask;
            Owner.RiseSignalMask &= ~Mask;
        }

        public void Keep()
        {
            Owner.DropSignalMask &= ~Mask;
            Owner.RiseSignalMask &= ~Mask;
        }

        public bool IsIncomingRised()
        {
            return (Owner.IncomingSignals & Mask) > 0;
        }

        public bool IsIncomingRised(MyNode node)
        {
            return (node.IncomingSignals & Mask) > 0;
        }

        public bool IsRised()
        {
            return (Owner.RiseSignalMask & Mask) > 0;
        }

        public bool IsDropped()
        {
            return (Owner.DropSignalMask & Mask) > 0;
        }

        public class MySignalTypeConverter : StringConverter
        {
            public override bool GetStandardValuesSupported(ITypeDescriptorContext context)
            {
                return true;
            }

            public override StandardValuesCollection GetStandardValues(ITypeDescriptorContext context)
            {
                List<string> standardValues = new List<string>();
                standardValues.Add("<none>");
                standardValues.AddRange(SIGNAL_NAMES.Keys);

                return new StandardValuesCollection(standardValues);
            }
        }
    }

    public class MyProxySignal : MySignal
    {
        public MySignal Source { get; set; }

        public override long Mask
        {
            get
            {
                if (Source != null)
                {
                    return Source.Mask;
                }
                else
                {
                    return base.Mask;
                }
            }
        }
    }

    public class MySignalStrategy
    {
        public static long SumAllIncomingSignals(MyNode target)
        {
            long result = 0;

            for (int i = 0; i < target.InputBranches; i++)
            {
                MyAbstractMemoryBlock mb = target.GetAbstractInput(i);

                if (mb != null)
                {                    
                    result |= mb.Owner.OutgoingSignals;                    
                }
            }

            return result;
        }

        public static long UseFirstInputSignal(MyNode target)
        {
            if (target.InputBranches == 0)
                return 0;

            long result = 0;

            MyAbstractMemoryBlock mb = target.GetAbstractInput(0);

            if (mb != null)
            {
                result |= mb.Owner.OutgoingSignals;
            }

            return result;
        }
    }
}
