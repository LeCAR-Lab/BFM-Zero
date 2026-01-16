
POLICY_CONFIG=config/policy/motivo_newG1.yaml # use the old version without damping arms
MODEL_ONNX_PATH=./model/model_example/exported/FBcprAuxModel.onnx
TASK=config/exp/tracking/walking.yaml

python rl_policy/bfm_zero.py \
    --robot_config config/robot/g1.yaml \
    --policy_config ${POLICY_CONFIG} \
    --model_path ${MODEL_ONNX_PATH} \
    --task  ${TASK}
