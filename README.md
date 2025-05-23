# Bone-segmentation 


## Task One
- All the code for task 1 is in the directory `task1` with the file names `task1_1.py` and so on for each sub task.

- Details of approaches are in the read me file inside `task1` dir

- The mask files with extension .nii.gz are saved in shared drive directory: https://drive.google.com/drive/folders/1V2-aYog73AeqZDZxYQScFmYen0s6CWw8?usp=sharing

- The result images are in `task1/results` with the file names begining with task1_1 and so on for each respective sub task.

- The output of sub task 1.4 is stored in a csv file in `task1/results/task_1_4_tibia_landmarks.csv`.

- `viz_seg.py` is used as helper module for visualizations.

## Usage

```bash
git clone https://github.com/thenaivekid/bone-segmentation.git
bash download_data.sh
conda create -n myenv python=3.13
conda activate myenv
conda install pip
pip install -r requirements.txt
cd task1
python3 task1_1.py # to run first sub task

```


- LLM tool claude was used to create the readme, understand the data and assist with some visualization code.