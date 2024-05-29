export GLOG_v=2
export MS_SUBMODULE_LOG_v="{PYNATIVE:0}"

mpirun -n 8 --allow-run-as-root python train.py --config ./configs/yolov7/yolov7-tiny.yaml --weight ./yolov7_ckpt_data/yolov7-tiny.ckpt --device_target Ascend --is_parallel True --epochs 1 --ms_mode 1 2>&1 | tee ./8p_pynative.log
grep "RunOp name" ./8p_pynative.log > 8p_pynative_aclop.log

