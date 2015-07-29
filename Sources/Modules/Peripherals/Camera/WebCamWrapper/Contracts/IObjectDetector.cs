using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Collections.ObjectModel;

namespace Touchless.Vision.Contracts
{
    public interface IObjectDetector : ITouchlessAddIn
    {
        event Action<IObjectDetector, DetectedObject, Frame> NewObject;
        event Action<IObjectDetector, DetectedObject, Frame> ObjectMoved;
        event Action<IObjectDetector, DetectedObject, Frame> ObjectRemoved;
        event Action<IObjectDetector, Frame, ReadOnlyCollection<DetectedObject>> FrameProcessed;
        
        ReadOnlyCollection<DetectedObject> DetectObjects(Frame frame);
    }
}
