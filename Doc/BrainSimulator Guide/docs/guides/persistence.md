# Persistence

You can activate automatic data loading after the simulation is started and you can also enable automatic data saving just before the simulation is stopped. There are global (project scope) and local (node scope) switches for that. Global tools are on the main toolbar and local ones are in the node property window.

## Data Loading
Selected nodes will be loaded with available data after the first simulation step is done. In this step, only `OneShot` tasks are executed (will be renamed to init tasks). Then, all persistable memory blocks will be overridden with loaded data.

There are three different places from where the data can be loaded:

* **Temporal location** - automatic, cannot be changed by the user, dependent on the project name and node id. On Windows, it is `%userprofile%\AppData\Local\Temp\bs_temporal`
* **User defined location** - node property `DataFolder`
* **Global location** - for all nodes, can be set through global loading option

According to this order, the first existing location is used for data loading. If no data is found, a validation error is thrown. Despite that, some error may appear during the loading process. For example, structure gets changed and the size of memory block as well. In such case, Brain Simulator will not load affected memory block and will throw a warning.


## Data Saving
If you activate data saving (global or local), selected nodes will be saved to the temporal location after the last simulation step is finished. In such way, you can start from user defined data and continue from the last stored simulation data. Any other behavior should be possible by combining save and load switches.

In order to store data outside of your Brain Simulator you have to export it from the temporal location. This is done globally (main toolbar button). All available data will be exported to user defined location and you can filter it out if needed.

Be aware that temporal data will outlive Brain Simulator and nodes with enabled loading will always load temporal data of the last finished simulation if available. If you want to start with different data then you have to clear temporal data first. There is a local (node only) and a global tool (whole project) for that.

## Controls
There are two toolbar groups for controlling which data to save and load. First of them is on the [simulation controls toolbar](../ui.md#simulation-controls)

![Global persistence](img/persistence-01.png)

*  **Global Load on Start** - toggle, loads data from specified folder
*  **Global Save on Stop** - toggle, saves data to temporary location
*  **Clear Stored Network State** - deletes all data in temporary location
*  **Export Stored Network State** - exports data from temporary location to specified folder
*  **Autosave During Simulation each X steps** - toggle, saves data to temporary location each X steps

## Order of data operations
After the first simulation step, all data loading is processed in following order:

* If **Global Load on Start** is active -- data are loaded from the folder defined when activating that option
* If **Load Data on Startup** is active for node -- node data are loaded from temporary location and overwrite previously loaded data
* Node data are loaded from **DataFolder** (node attribute) location and overwrite previously loaded data

After stopping the simulation, data are saved in following order:

* If **Save Data on Stop** is active for node -- node data are saved to **DataFolder** (node attribute) location
* If **Global Save on Stop** is active -- all data are saved to temporary location


## Validation
There are some mandatory validation rules applied when loading is enabled. Also, every node will inform you from which location the data is loaded and if the data will be saved at the end. Watch this information as it may change between simulations (1st simulation = user defined location, 2nd simulation = temporal location). Finally, when global loading is enabled, Brain Simulator will not check if there are data available for all nodes because different projects can vary a lot. It only throws a warning at the loading time when something is missing.
