#python3 main/test.py --cfg configs/fs_incremental/pascal5i_split0_5shot.yaml --load /data/cvpr2022/pascal5i_split0_BR_BS32.pt
#python3 main/test.py --cfg configs/fs_incremental/pascal5i_split1_5shot.yaml --load /data/cvpr2022/pascal5i_split1_BR_BS32.pt
#python3 main/test.py --cfg configs/fs_incremental/pascal5i_split2_5shot.yaml --load /data/cvpr2022/pascal5i_split2_BR_BS32.pt
#python3 main/test.py --cfg configs/fs_incremental/pascal5i_split3_5shot.yaml --load /data/cvpr2022/pascal5i_split3_BR_BS32.pt

#python main/test.py --cfg configs/non_fs_incremental/voc2012_5i_split3.yaml --load voc2012_split3_non_fs_final.pt



## few shot split 0
python main/test.py --cfg configs/fs_incremental/voc2012_5i_fs_split0_5shot.yaml --load voc2012_split0_non_fs_final.pt
#
## few shot split 1
#python main/test.py --cfg configs/fs_incremental/voc2012_5i_fs_split1_5shot.yaml --load voc2012_split1_non_fs_final.pt
#
## few shot split 2
#python main/test.py --cfg configs/fs_incremental/voc2012_5i_fs_split2_5shot.yaml --load voc2012_split2_non_fs_final.pt
#
## few shot split 3
#python main/test.py --cfg configs/fs_incremental/voc2012_5i_fs_split3_5shot.yaml --load voc2012_split3_non_fs_final.pt