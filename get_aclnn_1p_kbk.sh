export GLOG_v=1
export DEVICE_ID=5

python train.py --config ./configs/yolov7/yolov7-tiny.yaml --weight ./yolov7_ckpt_data/yolov7-tiny.ckpt --device_target Ascend --is_parallel False --epochs 1 2>&1 | tee ./1p_kbk.log
grep "select aclop" ./1p_kbk.log > 1p_kbk_aclop.log

