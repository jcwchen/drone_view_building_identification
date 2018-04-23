# drone_view_building_identification
Drone-view building identification by cross-view Triplet deep neural network and relative spatial estimation

Demo video: https://www.youtube.com/watch?v=sdq31ep2zYk

1. Environment: 
tensorflow 0.10.0
numpy 1.11.0
python 2.7

2. Run code:

train triplet:
python train_cross.py --model_dir model/ig_21.61 
(fine-tuned from --model_dir model_dir09el/ig_21.61)

extract deep features:
python extract_triplet_cross.py --model_dir model/weatherB_best 
(extract fc features by model/weatherB_best )

retrieval:
python retrieval.py --f frame/all  
(retrieval in frame/all)

similarity.py: compute similiarty

3. Dataset (will release after paper acceptance)

(1) Drone-BR
80 drone-view images: frame/all/
Sensor information per image: location.txt
Building information frome Google Place API per image: poi/[name].txt

Building information from Web:
a. ground-level image: search/
b. street-view image: streetview_clean/
c. aerial image: aerial_clean/

(2) Drone-BD
bounding box per image: Drone-BD/[name].txt

(3) IG-City8
same buildings are in same directory

Download model:
https://www.dropbox.com/sh/t8q70qnz9njtytq/AAAE2Hbw159p7Dj86QL2Z9hia

