# Combining Traditional Fine Tuning with Representational Fine Tuning

## File Descriptions:
`container.sif` is the built container that includes library installations. Note that you must build this after pulling by following the instructions below due to file size.

`image.def` holds the blueprint for building `container.sif`. It includes dependency installations.

`build.sub` is the default file for building a container using Apptainer.

<br>

`train_reft_and_lora.sub` is the submission file for submitting `reft_and_lora.py` to condor, you can use it to add required files, change output files, change the required memory and disk space, the number of runs, and other metadata.

`train_reft_only.sub` is the submission file for submitting `reft_only.py` to condor, you can use it to add required files, change output files, change the required memory and disk space, the number of runs, and other metadata.

`reft_only_exec.sh` primarily includes a single instruction to run `reft_only.py`.

`reft_and_lora_exec.sh` primarily includes a single instruction to run `reft_and_lora.py`.

<br>

`reft_only.py` trains a model on a demo database using only reft.

`reft_and_lora.py` trains a model on a demo database using lora then reft (note that it also prints the state after first training with just lora).

<br>


`.hf_token` is the file you will create with only your HuggingFace token.

## Running the Code
The following instructions are intended for running on chtc.

**Step by step guide to run the code (details explained below):**

1. Create your hf_token

2. Build the container

3. Run the program

4. Check the output

<br>

**Create your `.hf_token`**

Make a new file and name it `.hf_token`.

Put your HuggingFace token and nothing else in the file.

Make sure you have access to Llama 2 on HuggingFace.

<br>

**To build the container:**

If you have any dependencies to add, then they must be added to the container before you can import them from a Python file.

Add your dependencies to `image.def` through adding a new `conda install` or `pip install`. Follow the existing notation.

Write to terminal: `condor_submit -i build.sub`.

Write to terminal: `apptainer build container.sif image.def`.

Write to terminal: `exit`.

<br>

**To run:**

Write to terminal: `condor_submit train_reft_only.sub`.

OR

Write to terminal: `condor_submit train_reft_and_lora.sub`.

<br>

**To check if it is running:**

Write to terminal: `condor_q`.

<br>

**To see the reason it failed to run or additional information:**

Write to terminal: `condor_q -better-analyze`.

OR

Read the output files.

<br>

**To view output after a successful or erroneous run:**

Read the `.log`, `.out`, and `.err` files located in the directory `./condor_outs`.

To view when each file was last written to, write to terminal: `ls -l` .

## Demo Results
As a demonstration, LoRA and ReFT were both run on a model to train it on a demo database. The results can be found here: https://docs.google.com/document/d/1lV0LZI6_zt7l5Gy28PKKqJzqAcBXUT-JQG4EdkC_5PA/edit?usp=sharing.
