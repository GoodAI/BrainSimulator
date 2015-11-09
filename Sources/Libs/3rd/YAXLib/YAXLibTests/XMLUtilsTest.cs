// Copyright 2009 - 2010 Sina Iravanian - <sina@sinairv.com>
//
// This source file(s) may be redistributed, altered and customized
// by any means PROVIDING the authors name and all copyright
// notices remain intact.
// THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED. USE IT AT YOUR OWN RISK. THE AUTHOR ACCEPTS NO
// LIABILITY FOR ANY DATA DAMAGE/LOSS THAT THIS PRODUCT MAY CAUSE.
//-----------------------------------------------------------------------

using NUnit.Framework;

using YAXLib;
using System.Xml.Linq;

namespace YAXLibTests
{
    /// <summary>
    /// Summary description for XMLUtilsTest
    /// </summary>
    [TestFixture]
    public class XMLUtilsTest
    {
        [Test]
        public void CanCreateLocationTest()
        {
            var elem = new XElement("Base", null);

            Assert.That(XMLUtils.CanCreateLocation(elem, "level1/level2"), Is.True);
            var created = XMLUtils.CreateLocation(elem, "level1/level2");
            Assert.That(created.Name.ToString(), Is.EqualTo("level2"));
            Assert.That(XMLUtils.LocationExists(elem, "level1/level2"), Is.True);
            created = XMLUtils.CreateLocation(elem, "level1/level3");
            Assert.That(created.Name.ToString(), Is.EqualTo("level3"));
            Assert.That(XMLUtils.LocationExists(elem, "level1/level3"), Is.True);
        }
    }
}
