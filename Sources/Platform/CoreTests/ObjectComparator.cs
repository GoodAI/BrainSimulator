using System;
using System.Collections.Generic;
using System.Configuration;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace CoreTests
{
    /// <summary>
    /// A basic semantic object comparator. Only provides shallow compare right now.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class ObjectComparator<T>
    {
        class Comparison
        {
            public string MemberName { get; set; }
            public object First { get; set; }
            public object Second { get; set; }

            public override string ToString()
            {
                return string.Format("{0}: {1} vs {2}", MemberName, First, Second);
            }

            public bool CheckEqual()
            {
                if (First == Second)
                    return true;

                return First != null ? First.Equals(Second) : Second.Equals(First);
            }
        }

        private string[] m_ignore;

        public ObjectComparator(params string[] ignore)
        {
            m_ignore = ignore;
        }

        public void AssertEqual(T first, T second)
        {
            PropertyInfo[] properties = typeof (T).GetProperties(BindingFlags.Instance | BindingFlags.Public);
            // TODO(HonzaS): Add fields.

            var failedMembers = new List<Comparison>();

            foreach (PropertyInfo member in properties)
            {
                if (m_ignore.Contains(member.Name))
                    continue;

                var comparison = new Comparison
                {
                    MemberName = member.Name,
                    First = member.GetValue(first),
                    Second = member.GetValue(second)
                };

                if (!comparison.CheckEqual())
                    failedMembers.Add(comparison);
            }

            if (!failedMembers.Any()) return;

            foreach (Comparison failedMember in failedMembers)
            {
                Console.WriteLine(failedMember);
            }
            Assert.True(false, "Objects were not equal.");
        }
    }
}
