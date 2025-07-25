.. _examples_modelling_cbed_distortion_estimation_combine_ml_datasets_for_training_and_validation_then_split_sec:

Combining then splitting machine learning datasets for training and validation
==============================================================================

In this example, we perform the "action" of taking as input the 10 machine
learning (ML) datasets generated from the action described in :ref:`this page
<examples_modelling_cbed_distortion_estimation_generate_ml_datasets_for_training_and_validation_sec>`,
combining said input ML datasets, and then subsequently splitting the resulting
ML dataset into two output ML datasets: one intended for training ML models, the
other for validating ML models.

NOTE: Users are advised to read the remainder of the current page in its
entirety before trying to execute this action.

To execute the action, first we need to change into the directory
``<root>/examples/modelling/cbed/distortion/estimation/scripts``, where
``<root>`` is the root of the ``emicroml`` repository. Then, we need to run the
Python script ``./execute_action.py`` via the terminal command::

  python execute_action.py --action=<action> --use_slurm=<use_slurm>

where ``<action>`` must be equal to
``combine_ml_datasets_for_training_and_validation_then_split``, and
``<use_slurm>`` is either ``yes`` or ``no``. If ``<use_slurm>`` equals ``yes``
and a SLURM workload manager is available on the server from which you intend to
run the script, then the action will be performed as a SLURM job. If
``<use_slurm>`` is equal to ``no``, then the action will be performed locally
without using a SLURM workload manager.

If the action is to be performed locally without using a SLURM workload manager,
then prior to executing the above Python script, a set of Python libraries need
to be installed in the Python environment within which said Python script is to
be executed. See :ref:`this page
<examples_prerequisites_for_execution_without_slurm_sec>` for instructions on
how to do so. If the action is being performed as a SLURM job, then prior to
executing any Python commands that do not belong to Python's standard library, a
customizable sequence of commands are executed that are expected to try to
either activate an existing Python virtual environment, or create then activate
one, in which the Python libraries needed to complete the action successfully
are installed. See :ref:`this page
<examples_prerequisites_for_execution_with_slurm_sec>` for instructions how to
customize the sequence of commands.

The action described at the beginning of the current page takes automatically as
input data output data generated by the action described in the page
:ref:`examples_modelling_cbed_distortion_estimation_generate_ml_datasets_for_training_and_validation_sec`,
hence one must execute the latter action first, prior to the former. Upon
successful completion of the former action, approximately 80 percent of the
input ML data instances are stored in the output ML dataset intended for
training ML models, and the remaining input ML data instances are stored in the
output ML dataset intented for validating ML models. Moreover, upon successful
completion, the files storing the input ML datasets are deleted. The output ML
datasets intended for training and validating the ML models are stored in the
HDF5 files at the file paths
``<top_level_data_dir>/ml_datasets/ml_datasets_for_training_and_validation/ml_dataset_for_training.h5``
and
``<top_level_data_dir>/ml_datasets/ml_datasets_for_training_and_validation/ml_dataset_for_validation.h5``
respectively, where ``<top_level_data_dir>`` is
``<root>/examples/modelling/cbed/distortion/estimation/data``. **Be advised that
the files storing the ML datasets intended for training and validating the ML
models are approximately 620 GB and 125 GB in size respectively. Moreover,
approximately 1500 GB of free temporary storage space is required to complete
successfully the action described at the beginning of the curent page**.

In executing the action described at the beginning of the current page, multiple
scripts are executed. The particular scripts that are executed depend on the
command line arguments of the parent Python script introduced at the beginning
of this page. If ``<use_slurm>`` equals ``yes``, then the following scripts are
executed in the order that they appear directly below:

:download:`<root>/examples/modelling/cbed/distortion/estimation/scripts/execute_action.py <../../../../../../examples/modelling/cbed/distortion/estimation/scripts/execute_action.py>`
:download:`<root>/examples/modelling/cbed/common/scripts/combine_ml_datasets_for_training_and_validation_then_split/execute_all_action_steps.py <../../../../../../examples/modelling/cbed/common/scripts/combine_ml_datasets_for_training_and_validation_then_split/execute_all_action_steps.py>`
:download:`<root>/examples/modelling/cbed/common/scripts/combine_ml_datasets_for_training_and_validation_then_split/prepare_and_submit_slurm_job.sh <../../../../../../examples/modelling/cbed/common/scripts/combine_ml_datasets_for_training_and_validation_then_split/prepare_and_submit_slurm_job.sh>`
:download:`<root>/examples/modelling/cbed/common/scripts/combine_ml_datasets_for_training_and_validation_then_split/execute_main_action_steps.py <../../../../../../examples/modelling/cbed/common/scripts/combine_ml_datasets_for_training_and_validation_then_split/execute_main_action_steps.py>`

Otherwise, if ``<use_slurm>`` equals ``no``, then the third script, i.e. the one
with the basename ``prepare_and_submit_slurm_job.sh`` is not executed. See the
contents of the scripts listed above for implementation details. The last script
uses the module :mod:`emicroml.modelling.cbed.distortion.estimation`. It is
recommended that you consult the documentation of said module as you explore
said script. Lastly, if the action is being performed as a SLURM job, then the
default ``sbatch`` options, which are specified in the file with the basename
``prepare_and_submit_slurm_job.sh``, can be overridden by following the
instructions in :ref:`this page <examples_overriding_sbatch_options_sec>`.
