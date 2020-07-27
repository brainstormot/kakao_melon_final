# 1. Outline

We used ALS and Word2Vec for this competition.
About 20GB memory is necessary for inference.

Test result will be written in `./result/test/results.json`

Vaild result will be written in `./result/val/results.json`

# 2. Prerequisite

1. Put raw data json files in `./data/raw/`

- train.json
- val.json
- test.json
- genre_gn_all.json
- song_meta.json


2. Download trained Word2Vec model and put them in `./model/`

[DOWNLOAD LINK for './model/0007ky.p2v'](https://drive.google.com/file/d/1tbJffK-CZIC1dXW_8xdOoBh_6z1GqH8r/view?usp=sharing)

[DOWNLOAD LINK for './model/0007ky.w2v'](https://drive.google.com/file/d/19Lr2O73T1yk9-ctxb0yRIGocozzdPx72/view?usp=sharing)

# 3. How to run
To train Word2Vec model. 

    python train.py

But it is not necessary becuase we submit trained model `./model/0007ky.p2v` and `./model/0007ky.w2v`



To infer the result

    python inference.py