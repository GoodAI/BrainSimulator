import pdb
import random
import struct
from goodai.brainsim import load_brainsim
load_brainsim("C:\Users\michal.vlasak\Desktop\Debug\p")

from goodai.brainsim import MyProjectRunner, SchoolCurriculum, CurriculumManager, ToyWorldAdapterWorld, LearningTaskFactory

# Get Node
# node = runner.GetNode(22)
# or
# node = runner[22]

# Get Memory Block
# memblock = runner[22].Visual

# Get Values
# floatField = runner.GetValues(22, "Visual")
# or
# memblock = runner[22].Visual
# memblock.SafeCopyToHost()
# floatField = memblock.Host


runner = MyProjectRunner()
runner.OpenProject('C:/Users/michal.vlasak/Desktop/Debug/p/twtest.brain')
runner.DumpNodes()

school = runner[22]

curr = SchoolCurriculum()

for w in CurriculumManager.GetAvailableWorlds():
	if w.Name == ToyWorldAdapterWorld.__name__:
		for t in CurriculumManager.GetTasksForWorld(w):
			it = LearningTaskFactory.CreateLearningTask(t, school)
			it.RequiredWorldType = w
			curr.Add(it)

school.Curriculum = curr

actions = 13*[0]

runner.RunAndPause(1)


for _ in xrange(10):
	actions[0] = random.random()
	print(actions)
	runner.SetValues(24, actions, "Output")
	runner.RunAndPause(1)
	tWorld = school.CurrentWorld
	chosenActions = tWorld.ChosenActions
	tWorld.ChosenActions.SafeCopyToHost()
	print(chosenActions.Host[0])

runner.Shutdown()