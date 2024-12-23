# Model-Centric Track

This is the official GitHub of the **Model-Centric Track** of the Wake Vision Challenge (TBD: link to the challenge).

It asks participants to **push the boundaries of tiny computer vision** by **innovating model architectures** to achieve **high test accuracy** while **minimizing resource usage**, leveraging the newly released [Wake Vision](https://wakevision.ai/), a person detection dataset.

Proposed model architectures will be evaluated over a **private test set**.

## To Get Started
Create a new environment.

```
python -m venv /path/to/new/virtual/environment
```

Activate the environment.

```
source /path/to/new/virtual/environment/bin/activate
```

Install requirements.

```
python -m pip install -r requirements.txt
```

In "model_centric_track.py" modify the value of the variable "data_dir", writing the path to the location in which you would like to save the dataset (239.25 GiB).

The first execution will require hours, since it has to download the entire dataset. It will train the [ColabNAS](https://github.com/harvard-edge/Wake_Vision/blob/main/experiments/comprehensive_model_architecture_experiments/wake_vision_quality/k_8_c_5.py) model, a state-of-the-art person detection model, on the Wake Vision dataset to get you started. Then you can modify the "model_centric_track.py" script as you like, and propose your own model architecture.

```
python model_centric_track.py
```
