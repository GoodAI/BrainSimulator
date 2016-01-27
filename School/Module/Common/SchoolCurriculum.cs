using System.Collections;
using System.Collections.Generic;

namespace GoodAI.Modules.School.Common
{
    public enum CurriculumType
    {
        TrainingCurriculum,
        DebuggingCurriculum,
        TemporaryCurriculumSimon, // TODO: remove temporaries when curriculum is configurable and saveable from GUI
        TemporaryCurriculumOrest,
        TemporaryCurriculumMichal,
        TemporaryCurriculumMartinPoliak,
        TemporaryCurriculumMartinStransky,
        TemporaryCurriculumMartinMilota,
        TemporaryCurriculumMartinBalek,
        TemporaryCurriculumPeterHrosso
    }

    /// <summary>
    /// Holds tasks that an agent should be trained with to gain new abilities
    /// </summary>
    public class SchoolCurriculum : IEnumerable<ILearningTask>
    {
        protected List<ILearningTask> Tasks = new List<ILearningTask>();
        private IEnumerator<ILearningTask> m_taskEnumerator;

        // for foreach usage
        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }

        public IEnumerator<ILearningTask> GetEnumerator()
        {
            return Tasks.GetEnumerator() as IEnumerator<ILearningTask>;
        }

        // for classic usage
        public ILearningTask GetNextLearningTask()
        {
            if (m_taskEnumerator == null)
                m_taskEnumerator = Tasks.GetEnumerator();
            if (m_taskEnumerator.MoveNext())
                return m_taskEnumerator.Current;
            return null;
        }

        public void ResetLearningProgress()
        {
            m_taskEnumerator.Reset();
        }

        public void AddLearningTask(ILearningTask task)
        {
            // TODO: if tasks are added by a caller in random order, insert the task after tasks that train the required abilities
            Tasks.Add(task);
        }
    }

    public class SchoolCurriculumPlanner
    {
        public static SchoolCurriculum GetCurriculumForWorld(AbstractSchoolWorld world)
        {
            SchoolCurriculum curriculum = new SchoolCurriculum();

            switch (world.TypeOfCurriculum)
            {
                //case CurriculumType.TrainingCurriculum:
                //    {

                //        // TODO: add more tasks to this curriculum:
                //        //curriculum.AddLearningTask(LearningTaskFactory.CreateLearningTask(LearningTaskNameEnum.DetectBlackAndWhite, world));
                //        curriculum.AddLearningTask(DeprecatedLearningTaskFactory.CreateLearningTask(LearningTaskNameEnum.ApproachTarget, world));
                //        break;
                //    }
                case CurriculumType.DebuggingCurriculum:
                    curriculum.AddLearningTask(LearningTaskFactory.CreateLearningTask(LearningTaskNameEnum.DebuggingTask, world));
                    break;
                case CurriculumType.TemporaryCurriculumSimon:
                    {
                        //curriculum.AddLearningTask(LearningTaskFactory.CreateLearningTask(LearningTaskNameEnum.DetectWhite, world));
                        //curriculum.AddLearningTask(LearningTaskFactory.CreateLearningTask(LearningTaskNameEnum.DetectColor, world));
                        curriculum.AddLearningTask(LearningTaskFactory.CreateLearningTask(LearningTaskNameEnum.ApproachTarget, world));
                        //curriculum.AddLearningTask(LearningTaskFactory.CreateLearningTask(LearningTaskNameEnum.SimpleSizeDetection, world));
                        break;
                    }
                case CurriculumType.TemporaryCurriculumOrest:
                    {
                        curriculum.AddLearningTask(LearningTaskFactory.CreateLearningTask(LearningTaskNameEnum.MovingTargetD, world));
                        break;
                    }
                //case CurriculumType.TemporaryCurriculumMichal:
                //    {
                //        curriculum.AddLearningTask(DeprecatedLearningTaskFactory.CreateLearningTask(LearningTaskNameEnum.CooldownAction, world));
                //        break;
                //    }
                //case CurriculumType.TemporaryCurriculumMartinPoliak:
                //    {
                //        curriculum.AddLearningTask(DeprecatedLearningTaskFactory.CreateLearningTask(LearningTaskNameEnum.SimpleDistanceDetection, world));
                //        break;
                //    }
                case CurriculumType.TemporaryCurriculumMartinStransky:
                    curriculum.AddLearningTask(LearningTaskFactory.CreateLearningTask(LearningTaskNameEnum.CompareLayouts, world));
                    break;
                //case CurriculumType.TemporaryCurriculumMartinMilota:
                //    {
                //        curriculum.AddLearningTask(DeprecatedLearningTaskFactory.CreateLearningTask(LearningTaskNameEnum.SimpleDistanceDetection, world));
                //        break;
                //    }
                //case CurriculumType.TemporaryCurriculumMartinBalek:
                //    {
                //        curriculum.AddLearningTask(DeprecatedLearningTaskFactory.CreateLearningTask(LearningTaskNameEnum.SimpleSizeDetection, world));
                //        break;
                //    }
                //case CurriculumType.TemporaryCurriculumPeterHrosso:
                //    {
                //        curriculum.AddLearningTask(DeprecatedLearningTaskFactory.CreateLearningTask(LearningTaskNameEnum.SimpleDistanceDetection, world));
                //        break;
                //    }
            }

            return curriculum;
        }
    }
}
