using GoodAI.Core.Observers;
using YAXLib;

namespace GoodAI.Core.Utils
{
    [YAXSerializeAs("Location")]
    public class MyLocation
    {
        [YAXAttributeForClass]
        public float X { get; set; }
        [YAXAttributeForClass]
        public float Y { get; set; }        
    }

    [YAXSerializeAs("Size")]
    public class MySize
    {
        [YAXAttributeForClass]
        public float Width { get; set; }
        [YAXAttributeForClass]
        public float Height { get; set; }
    }

    [YAXSerializeAs("CameraData")]
    public class MyCameraData
    {
        [YAXAttributeForClass]
        public MyAbstractObserver.ViewMethod CameraType { get; set; }

        [YAXAttributeForClass]
        public float X { get; set; }
        [YAXAttributeForClass]
        public float Y { get; set; }
        [YAXAttributeForClass]
        public float Z { get; set; }
    }

    [YAXSerializeAs("LayoutProperties")]
    public class MyLayout
    {
        [YAXSerializableField(DefaultValue = 1), YAXAttributeForClass]
        public float Zoom { get; set; } 

        [YAXSerializableField]
        public MyLocation Translation { get; set; }

        public MyLayout()
        {
            Zoom = 1;
            Translation = new MyLocation();
        }
    }
}
