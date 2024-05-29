export GLOG_v=1

mpirun -n 8 --allow-run-as-root python train.py --config ./configs/yolov7/yolov7-tiny.yaml --weight ./yolov7_ckpt_data/yolov7-tiny.ckpt --device_target Ascend --is_parallel True --epochs 1 2>&1 | tee ./8p_kbk.log
grep "select aclop" ./8p_kbk.log > 8p_kbk_aclop.log

