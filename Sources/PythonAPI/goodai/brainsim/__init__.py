import os
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

	class MyProjectRunner(DefaultProjectRunner):

		# you cannot use number as attrs - therefore this
		def __getitem__(self, idx):
			node = self.GetNode(idx)
			return node



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


	if not os.path.isdir(brainsim_path):
		print("Path {} does not exist. Unable to load Brain Simulator.".format(brainsim_path))
		return False

	for dllName in ["Platform.Core", "School"]:
		absPath = brainsim_path + '\GoodAI.' + dllName + '.dll'
		if not os.path.isfile(absPath):
			print("Unable to find file {}. Did you specify the path to the latest Brain Simulator installation?".format(absPath))
			return False
		clr.AddReference(absPath)

	_load_classes()
	_init(brainsim_path)
	return True
