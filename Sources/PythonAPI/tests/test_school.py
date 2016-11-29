import os
import pytest

from goodai.brainsim import load_brainsim
load_brainsim('D:/BrainSimInternal/Sources/Platform/BrainSimulator/bin/Debug')

# because default log writer is flawed when running multiple runners one after another
from GoodAI.Core.Utils import MyLog
log = MyLog.MyTextLogWriter()
MyLog.Writer = log


@pytest.fixture
def runner():
    from goodai.brainsim import MyProjectRunner
    runner = MyProjectRunner()
    yield runner
    runner.Shutdown()


def get_curriculum(runner):
    from GoodAI.Modules.School.Common import SchoolCurriculum, LearningTaskFactory
    from GoodAI.School.Common import CurriculumManager
    runner.OpenProject(os.path.abspath('tests/brains/twtest.brain'))
    school = runner[22]
    curr = SchoolCurriculum()

    for w in CurriculumManager.GetAvailableWorlds():
        for t in CurriculumManager.GetTasksForWorld(w):
            it = LearningTaskFactory.CreateLearningTask(t, school)
            it.RequiredWorldType = w
            curr.Add(it)

    return curr


def test_create_curriculum(runner):
    from GoodAI.Modules.School.Common import SchoolCurriculum, LearningTaskFactory
    from GoodAI.School.Common import CurriculumManager
    runner.OpenProject(os.path.abspath('tests/brains/twtest.brain'))

    school = runner[22]

    curr = SchoolCurriculum()

    for w in CurriculumManager.GetAvailableWorlds():
        for t in CurriculumManager.GetTasksForWorld(w):
            it = LearningTaskFactory.CreateLearningTask(t, school)
            it.RequiredWorldType = w
            curr.Add(it)

    school.Curriculum = curr

    runner.RunAndPause(1)
    assert True


def test_save_curriculum():  # private in SchoolRunForm_Ops
    assert False


def test_load_saved_curriculum(runner):
    from GoodAI.School.Common import CurriculumManager
    from GoodAI.Modules.School.Common import SchoolCurriculum

    plan = CurriculumManager.LoadPlanDesign(os.path.abspath('tests/curricula/test.xml'))
    runner.OpenProject(os.path.abspath('tests/brains/twtest.brain'))
    school = runner[22]
    curriculum = plan.AsSchoolCurriculum(school)

    school.Curriculum = curriculum

    runner.RunAndPause(1)
    assert True


def test_user_can_access_TU_result(runner):
    from GoodAI.School.Common import CurriculumManager
    from GoodAI.Modules.School.Common import SchoolCurriculum, TrainingResult

    plan = CurriculumManager.LoadPlanDesign(os.path.abspath('tests/curricula/test.xml'))
    runner.OpenProject(os.path.abspath('tests/brains/twtest.brain'))
    school = runner[22]
    curriculum = plan.AsSchoolCurriculum(school)

    school.Curriculum = curriculum

    runner.RunAndPause(10)

    assert TrainingResult.TUInProgress == school.TaskResult


def test_simulation_saves_run_log():    # tied to GUI
    assert False


def test_user_knows_saved_log_path():   # tied to GUI
    assert False


def test_pause_and_resume_curriculum(runner):
    from GoodAI.Core.Execution import MySimulationHandler

    curriculum = get_curriculum(runner)
    school = runner[22]
    school.Curriculum = curriculum
    runner.RunAndPause(1)
    assert runner.SimulationHandler.State == MySimulationHandler.SimulationState.PAUSED
    runner.RunAndPause(1)
    assert True


def test_stop_running_curriculum(runner):
    from GoodAI.Core.Execution import MySimulationHandler

    curriculum = get_curriculum(runner)
    school = runner[22]
    school.Curriculum = curriculum
    runner.RunAndPause(10)
    runner.Reset()
    assert runner.SimulationHandler.State == MySimulationHandler.SimulationState.STOPPED


def test_run_curriculum_after_stopping_another_curriculum(runner):
    from GoodAI.Modules.School.Common import SchoolCurriculum, LearningTaskFactory
    from GoodAI.School.Common import CurriculumManager

    runner.OpenProject(os.path.abspath('tests/brains/twtest.brain'))
    school = runner[22]
    i = 0

    curr = SchoolCurriculum()
    for w in CurriculumManager.GetAvailableWorlds():
        for t in CurriculumManager.GetTasksForWorld(w):
            it = LearningTaskFactory.CreateLearningTask(t, school)
            it.RequiredWorldType = w
            if i % 3:
                curr.Add(it)
            i += 1

    school.Curriculum = curr
    runner.RunAndPause(10)

    i = 0

    curr = SchoolCurriculum()
    for w in CurriculumManager.GetAvailableWorlds():
        for t in CurriculumManager.GetTasksForWorld(w):
            it = LearningTaskFactory.CreateLearningTask(t, school)
            it.RequiredWorldType = w
            if i % 2:
                curr.Add(it)
            i += 1

    school.Curriculum = curr
    runner.RunAndPause(10)
    assert True
