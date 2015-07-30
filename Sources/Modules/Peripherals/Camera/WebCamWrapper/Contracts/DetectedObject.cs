using System.Drawing;
using System.Runtime.Serialization;

namespace Touchless.Vision.Contracts
{
    [DataContract]
    public class DetectedObject
    {
        private static readonly object SyncObject = new object();
        private static int _nextId = 1;

        public DetectedObject()
        {
            Id = NextId();
        }

        [DataMember]
        public int Id { get; private set; }

        [IgnoreDataMember]
        public Bitmap Image { get; set; }

        //[DataMember]
        //public byte[] ImageData
        //{
        //    get
        //    {
        //        byte[] data = null;
        //        if (this.Image != null)
        //        {
        //            MemoryStream memoryStream = new MemoryStream();
        //            this.Image.Save(memoryStream, ImageFormat.Bmp);
        //            memoryStream.Flush();
        //            data = memoryStream.ToArray();
        //            memoryStream.Close();
        //        }

        //        return data;


        //    }
        //}

        [DataMember]
        public virtual Point Position { get; set; }

        public void AssignNewId()
        {
            Id = NextId();
        }


        private static int NextId()
        {
            int result;
            lock (SyncObject)
            {
                result = _nextId;
                _nextId++;
            }

            return result;
        }
    }
}