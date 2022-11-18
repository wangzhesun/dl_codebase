#python3 main/test.py --cfg configs/fs_incremental/coco_20i_split0_5shot_renorm.yaml --load /data/cvpr2022/coco20i_split0_BR_BS32.pt
#python3 main/test.py --cfg configs/fs_incremental/coco_20i_split1_5shot_renorm.yaml --load /data/cvpr2022/coco20i_split1_BR_BS32.pt
#python3 main/test.py --cfg configs/fs_incremental/coco_20i_split2_5shot_renorm.yaml --load /data/cvpr2022/coco20i_split2_BR_BS32.pt
#python3 main/test.py --cfg configs/fs_incremental/coco_20i_split3_5shot_renorm.yaml --load /data/cvpr2022/coco20i_split3_BR_BS32.pt


## few shot split 0
python main/test.py --cfg configs/fs_incremental/coco20i_split0_5shot.yaml --load GIFS_coco20i_split0_final.pt
#
## few shot split 1
#python main/test.py --cfg configs/fs_incremental/coco20i_split1_5shot.yaml --load GIFS_coco20i_split1_final.pt
#
## few shot split 2
#python main/test.py --cfg configs/fs_incremental/coco20i_split2_5shot.yaml --load GIFS_coco20i_split2_final.pt
#
## few shot split 3
#python main/test.py --cfg configs/fs_incremental/coco20i_split3_5shot.yaml --load GIFS_coco20i_split3_final.pt