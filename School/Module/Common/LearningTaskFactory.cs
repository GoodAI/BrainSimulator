using GoodAI.Modules.School.LearningTasks;
using GoodAI.Modules.School.Worlds;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

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
        
        private static IEnumerable<Type> m_knownLearningTasks = null;

        public static IEnumerable<Type> KnownLearningTasks
        {
            get
            {
                if (m_knownLearningTasks == null)
                {
                    //TODO: check multi-level inheritance if there will be any in future
                    m_knownLearningTasks = Assembly.GetAssembly(typeof(AbstractLearningTask<>))
                        .GetTypes()
                        .Where(x => x.BaseType != null &&
                            x.BaseType.IsGenericType &&
                            x.BaseType.GetGenericTypeDefinition() == typeof(AbstractLearningTask<>));
                }
                return m_knownLearningTasks;
            }
        }

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
                case LearningTaskNameEnum.MultipleTargetsSequence:
                    return new LTMultipleTargetsSequence(w);
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

        public static List<Type> GetSupportedWorlds(Type taskType)
        {
            //TODO: check multi-level inheritance if there will be any in future
            // get generic parameter
            Type genericType = taskType.BaseType.GetGenericArguments()[0];
            // look up all derived classes of this type
            List<Type> results = Assembly.GetAssembly(typeof(AbstractSchoolWorld))
                .GetTypes()
                .Where(x => x.BaseType == genericType)
                .ToList();
            results.Add(genericType);
            // remove abstract classes
            results = results.Where(x => !x.IsAbstract).ToList();

            return results;
        }
    }
}
