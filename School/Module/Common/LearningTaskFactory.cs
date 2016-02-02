using GoodAI.Modules.School.LearningTasks;
using GoodAI.Modules.School.Worlds;

namespace GoodAI.Modules.School.Common
{
    public enum AbilityNameEnum
    {
        SimplestPatternDetection,
    }

    public enum LearningTaskNameEnum
    {
        DebuggingTask,
        DetectWhite,
        DetectBlackAndWhite,
        DetectShape,
        ApproachTarget,
        SimpleSizeDetection,
        SimpleDistanceDetection,
        MovingTarget,
        MovingTargetD,
        Obstacles,
        MultipleTargetsSequence,
        OneDApproach,
        DetectColor,
        CooldownAction,
        DetectAngle,
        DetectShapeColor,
        CopyAction,
        CopySequence,
        DetectDifference,
        DetectSimilarity,
        CompareLayouts,
        ClassComposition
    }

    public class LearningTaskFactory
    {
        public static ILearningTask CreateLearningTask(LearningTaskNameEnum learningTaskName, SchoolWorld w)
        {
            switch (learningTaskName)
            {
                case LearningTaskNameEnum.DetectWhite:
                    return new LTDetectWhite(w);
                case LearningTaskNameEnum.ApproachTarget:
                    return new LTApproach(w);
                case LearningTaskNameEnum.SimpleSizeDetection:
                    return new LTSimpleSize(w);
                case LearningTaskNameEnum.SimpleDistanceDetection:
                    return new LTSimpleDistance(w);
                case LearningTaskNameEnum.DebuggingTask:
                    return new LTDebugging(w);
                case LearningTaskNameEnum.MovingTarget:
                    return new LTMovingTarget(w);
                case LearningTaskNameEnum.MovingTargetD:
                    return new LTMovingTargetD(w);
                case LearningTaskNameEnum.Obstacles:
                    return new LTObstacles(w);
                case LearningTaskNameEnum.OneDApproach:
                    return new LT1DApproach(w);
                case LearningTaskNameEnum.DetectColor:
                    return new LTDetectColor(w);
                case LearningTaskNameEnum.CooldownAction:
                    return new LTActionWCooldown(w);
                case LearningTaskNameEnum.DetectShape:
                    return new LTDetectShape(w);
                case LearningTaskNameEnum.DetectShapeColor:
                    return new LTDetectShapeColor(w);
                case LearningTaskNameEnum.DetectBlackAndWhite:
                    return new LTDetectBlackAndWhite(w);
                case LearningTaskNameEnum.DetectAngle:
                    return new LTSimpleAngle(w);
//                case LearningTaskNameEnum.ShapeGroups:
//                    return new LTShapeGroups(w);
                case LearningTaskNameEnum.CopyAction:
                    return new LTCopyAction(w);
                case LearningTaskNameEnum.CopySequence:
                    return new LTCopySequence(w);
                case LearningTaskNameEnum.DetectDifference:
                    return new LTDetectDifference(w);
                case LearningTaskNameEnum.DetectSimilarity:
                    return new LTDetectSimilarity(w);
                case LearningTaskNameEnum.CompareLayouts:
                    return new LTCompareLayouts(w);
                case LearningTaskNameEnum.ClassComposition:
                    return new LTClassComposition(w);
                default:
                    return null;
            }
        }
    }

}
