
import clr
import types
from _winreg import OpenKey, QueryValueEx, HKEY_LOCAL_MACHINE


def _configureContainer():
	conf = CoreContainerConfiguration()
	conf.Configure(TypeMap.SimpleInjectorContainer)


def _configureModulesAndKernels(brainsim_path):
	for module_name in ["BasicNodes", "InternalNodes", "MNIST", "School", "ToyWorld"]:
		MyConfiguration.ModulesSearchPath.Add("{}\modules\GoodAI.{}.dll".format(brainsim_path, module_name))
	MyConfiguration.GlobalPTXFolder = "{}\module\GoodAI.BasicNodes\ptx".format(brainsim_path)


def _init(brainsim_path):
	_configureContainer()
	_configureModulesAndKernels(brainsim_path)


def _load_classes():
	global MyProjectRunner, MyConfiguration, CoreContainerConfiguration, TypeMap, SchoolCurriculum, CurriculumManager, ToyWorldAdapterWorld, \
		LearningTaskFactory

	from GoodAI.Core.Execution import MyProjectRunner as DefaultProjectRunner
	from GoodAI.Core.Configuration import MyConfiguration
	from GoodAI.Core.Nodes import MyWorkingNode
	from GoodAI.Platform.Core.Configuration import CoreContainerConfiguration
	from GoodAI.TypeMapping import TypeMap
	from GoodAI.Modules.School.Common import SchoolCurriculum, LearningTaskFactory
	from GoodAI.School.Common import CurriculumManager
	from GoodAI.School.Worlds import ToyWorldAdapterWorld

	# maybe it could be done dynamically straight with MyProjectRunner but it is not working
	# right now, so I am postponing it - this is much easier and under control
	class MyProjectRunner(DefaultProjectRunner):

		# you cannot use number as attrs - therefore this
		def __getitem__(self, idx):
			node = self.GetNode(idx)

			class TTT(type(node)):
				def __init__(self, node):
					self._node = node

				def __getattr__(self, name):
					print("Looking for "+name)

			pode = TTT(node)

			return node
			# any of these 3 work
			#type(node).__getattr__ = test
			#setattr(type(node), "__getattr__", types.MethodType(test, self))
			#type(node).__getattr__ = types.MethodType(test, self)



def _brainsim_path_from_registry():
	key = OpenKey(HKEY_LOCAL_MACHINE, "SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\BrainSimulator.exe")
	val = QueryValueEx(key, "Path")
	return val[0]


def load_brainsim(brainsim_path=None):
	if not brainsim_path:
		try:
			brainsim_path = _brainsim_path_from_registry()
			print("Using BrainSimulator path from registry: {}".format(brainsim_path))
		except WindowsError:
			pass
	else:
		print("Using user-specified path to BrainSimulator: {}".format(brainsim_path))
	if not brainsim_path:
		brainsim_path = "C:\Program Files\GoodAI\Brain Simulator"
		print("Using default BrainSimulator path: {}".format(brainsim_path))

	import clr

	clr.AddReference(brainsim_path + '\GoodAI.Platform.Core.dll')
	clr.AddReference(brainsim_path + '\GoodAI.School.dll')

	_load_classes()
	_init(brainsim_path)
