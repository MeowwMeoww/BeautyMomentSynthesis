# Beautiful moments systhesis
A project aims to retain all the beautiful pictures of you and your love ones by synthesizing a video.

## Table of Contents
* [Team members](#team-members)
* [General info](#general-info)
* [Implement](#implement)
* [Setup](#setup)

## Team members
- Tran Cong Thinh
- Nguyen Hoang Phuc
- Vu Thi Quynh Nga

## General info
This project include four main modules:
- Face detection
- Face recognition
- Face image quality assessment
- Smile evaluation
- Video synthesis

Input: an input image folder + an anchor image folder. 
Output: a video comprised of the most beautiful, happy moments in the input image folder.
Link to our [presentation](https://drive.google.com/file/d/1ARZQUff1AB6bCPEb5_SQjpTgRus900VR/view?usp=sharing).

## Code structure
```bash
BeautyMomentSynthesis
├── SmileScore
│   └── smileScore.py
├── animations
│   ├── animations.py
│   └── make_video.py
├── face_reg
│   ├── detection.py  
│   └── read_video.py
├── flaskapp
│   ├── static
│   ├── templates
│   │   ├── index.html
│   ├── Beauty_Moment_Synthesis_API.ipynb // Python notebook for deploying the API in Google Colaboratory
│   ├── app.py
│   └── requirements.txt
├── misc
│   ├── extract_bbox.py
│   ├── log.py
│   ├── utils.py
│   └── visualize.py
├── model
│   └── model.py
├── README.md
├── config.py
├── SDD-FIQA.py
├── main.py
└── requirements.txt
```

## Dataset needed structure :

```bash
BeautyMomentSynthesis
├── Input folder
│   ├── img_1.jpg
│   ├── img_2.jpeg
│   ├── img_3.JPG
│   ......
│
├── Anchor folder
│   ├── People 1
│        ├── img_1.jpg
│        ├── img_2.jpg
│        ├── img_3.jpg
│        ...........
│   ├── People 2
│        ├── img_1.jpg
│        ├── img_2.jpg
│        ├── img_3.jpg
│        .........
│     ........
```

## Implement

- Install requirements:
```
pip install -r requirements.txt
```

- Use this command:

```
python main.py --anchor_dataset_path "path_to_anchor" --original_dataset_path "path_to_dataset" --output_path "path_to_output_vid.mp4" --auto_vid_params True --find_person "person_name1 person_name2" --number_of_images --log --visualization 
```

